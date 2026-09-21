# =============================================================================
# net.py —— 视频去雨的整体网络结构（CoarseNet 粗网络 + FineNet 精网络）
#
# 项目采用"粗到精"(coarse-to-fine) 的两阶段框架，本文件定义了：
#   1. CoarseNet       ：以连续 d 帧作为输入，利用编码器-解码器(LapSRN 风格)
#                        提取多尺度特征，再送入时空交互的 R-CLSTM 精化网络，
#                        输出 d 帧的去雨结果（对应论文中的粗阶段 / 时空交互学习）。
#   2. FineNet / FineNet_npic ：把粗网络的输出与输入拼接，用 3D 卷积 + RRDB
#                        残差密集块进一步精修，输出单帧(中间帧)或多帧结果。
#   3. C_C3D_1          ：用于融合时间序列的 3D 卷积，属于精网络的一部分。
#
# 张量约定：本文件全程使用 5 维张量 (b, c, d, h, w) 表示一段视频/特征立方体，
#         其中 d = 时间维(帧数)，即一个 batch 里 b 段视频，每段 d 帧。
# =============================================================================
import torch
import torch.nn as nn
from models import modules
from models.refinenet_dict import refinenet_dict

from models import block as B


def cubes_2_maps(cubes):
    """将视频立方体张量 (b, c, d, h, w) 展平成批量的 2 维特征图 (b*d, c, h, w)。

    这样做的目的：把时间维"摊平"到 batch 维，从而可以复用 2D 卷积
    (比如 backbone / decoder) 一次性处理所有帧，加快计算。
    """
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)  # 维度重排为 (b, d, c, h, w)

    return cubes.contiguous().view(b*d, c, h, w), b, d  # 返回 (b*d,c,h,w)、b、d


def maps_2_cubes(maps, b, d):
    """cubes_2_maps  的逆操作：把 (b*d, c, h, w) 的特征图恢复为视频立方体 (b, c, d, h, w)。"""
    bd, c, h, w = maps.shape
    cubes = maps.contiguous().view(b, d, c, h, w)  # (b, d, c, h, w)

    return cubes.permute(0, 2, 1, 3, 4)  # 还原为 (b, c, d, h, w)


def rev_maps(maps, b, d):
    """反向时间序列：把特征图先恢复成立方体，再沿时间维 flip，然后重新展平。

    用于双向(BiLSTM)时空精化时，为反向 R-CLSTM 提供"倒叙"的输入序列。
    """
    cubes = maps_2_cubes(maps, b, d).flip(dims=[2])  # 沿时间维(dim 2)倒序

    return cubes_2_maps(cubes)[0]


class CoarseNet(nn.Module):
    """粗网络 CoarseNet：视频去雨的"粗阶段"主干。

    处理流程（对输入 d 帧视频）：
      编码器 E (ResNet/DenseNet) 提取多尺度特征 -> 解码器 D 逐级上采样融合
      -> 多层特征融合 MFF -> R-CLSTM 时空精化网络 R_fwd，逐帧得到去雨结果。
    若开启双向 spatial-temporal 交互，则额外用 R_bwd 反向精化，再用 1x1 卷积融合。
    """

    def __init__(self, encoder, decoder, block_channel, refinenet, bidirectional=False, input_residue=False):

        super(CoarseNet, self).__init__()

        self.use_bidirect = bidirectional      # 是否使用双向(R-CLSTM)时空精化
        self.input_residue = input_residue      # 是否在最终输出与输入之间做残差连接

        self.E = encoder                        # 编码器，输出 5 层不同尺度的特征
        self.D = decoder                        # 解码器，多尺度逐级上采样
        self.MFF = modules.MFF(block_channel)   # 多层特征融合模块，融合不同尺度的编码特征

        self.R_fwd = refinenet_dict[refinenet](block_channel)  # 前向 R-CLSTM 时空精化网络

        if self.use_bidirect:
            self.R_bwd = refinenet_dict[refinenet](block_channel)                       # 反向 R-CLSTM
            self.bidirection_fusion = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2, bias=True)  # 融合前后向结果 -> 3通道RGB

    def forward(self, x):
        x_cube = x  # 保留原始输入立方体 (b,3,d,h,w)，供后续使用/残差

        x, b, d = cubes_2_maps(x)  # (b,3,d,h,w) -> 展平为 (b*d,3,h,w)，b,d 用于还原
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x)  # 编码器提取 5 层不同尺度特征
        x_decoder = self.D(x_block0, x_block1, x_block2, x_block3, x_block4)  # 解码器逐级上采样，输出高分辨率去雨特征图
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])  # 多尺度特征融合，缩放至解码器大小
        fwd_out = self.R_fwd(torch.cat((x_decoder, x_mff), 1), b, d)  # 送入前向 R-CLSTM，逐帧精化，输出 (b,3,d,h,w)

        if self.use_bidirect:
            # 反向分支：将特征沿时间维倒序后输入反向 R-CLSTM，实现"过去-未来"双向时空交互
            bwd_out = self.R_bwd(torch.cat((rev_maps(x_decoder, b, d),
                                            rev_maps(x_mff, b, d)), 1),
                                 b, d)

            # 前向与(还原方向后的)反向结果在通道维拼接，用卷积融合成最终 3 通道输出
            concat_cube = cubes_2_maps(torch.cat((fwd_out, bwd_out.flip(dims=[2])), 1))[0]

            out = maps_2_cubes(self.bidirection_fusion(concat_cube), b=b, d=d)
        else:
            out = fwd_out

        # 处理奇数像素宽度：解码器两次 2 倍上采样可能多出 1 列，裁掉保持与原图尺寸一致（理解存疑）
        if x_cube.shape[-1] % 2 == 1:
            out = out[:, :, :, :, :-1]

        if self.input_residue:
            return out + x_cube  # 残差学习：输出 = 网络预测 + 输入，让网络只学"雨层/残差"
        else:
            return out


class FineNet(nn.Module):
    """精网络 FineNet（非 npic 版本）：输出中间单帧的去雨结果。

    粗网络输出 d 帧后，把"粗输出 + 原始输入"拼接成 6 通道，用 3D 卷积做时间融合，
    再经若干 RRDB 残差密集块精修，最终输出 (b,3,1,h,w) 即中间那一帧（或经卷积降为 3 通道）。
    """

    def __init__(self, num_features=64, num_blocks=9, out_nc=3, mode='CNA', act_type='relu', norm_type=None):
        super(FineNet, self).__init__()

        nb, nf = num_blocks, num_features

        # 3d convolution to fuse sequence.  # 用 3D 卷积融合时间维上的帧间信息
        c3d = C_C3D_1()

        # nb 个 RRDB 残差密集块用于特征精修；gc=32 为生长通道数
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)   # 精修后的特征卷积
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=norm_type, act_type=None)   # 输出 3 通道 RGB

        # 顺序：c3d(3D融合) -> 带 LR 卷积的 shortcut 残差块(RRDB链) -> 两个 HR 卷积
        self.net = B.sequential(c3d, B.ShortcutBlock_ZKH(B.sequential(*rb_blocks), LR_conv), HR_conv0, HR_conv1)

    def forward(self, c_out, c_in):
        # 6-channel = 3 (rgb) model_out_C + 3 (rgb) in_videos_C
        inp_6c_F = torch.cat((c_out, c_in), 1)  # 拼接 6 通道 (b,6,d,h,w)
        # 残差：网络输出 + 粗网络"中间帧"输出；粗输出时间维被卷积压成 1 帧，故取中间帧做残差
        return self.net(inp_6c_F) + c_out[:, :, c_out.shape[2] // 2, :, :]


class FineNet_npic(nn.Module):
    """精网络 FineNet_npic（npic 版本）：一次性输出全部 d 帧去雨结果。

    out_nc=15 来自 5 帧 * 3 通道 = 15，即把每个时间步的 3 通道结果按通道展开输出。
    """

    def __init__(self, num_features=64, num_blocks=9, out_nc=15, mode='CNA', act_type='relu', norm_type=None):
        super(FineNet_npic, self).__init__()

        nb, nf = num_blocks, num_features

        # 3d convolution to fuse sequence.  # 用 3D 卷积做帧间(时间)信息融合
        c3d = C_C3D_1()

        # nb 个 RRDB 残差密集块
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=norm_type, act_type=None)  # 输出 15 = 5帧*3通道

        self.net = B.sequential(c3d, B.ShortcutBlock_ZKH(B.sequential(*rb_blocks), LR_conv), HR_conv0, HR_conv1)

    def forward(self, c_out, c_in):
        b, c, d, h, w = c_out.shape

        # 6-channel = 3 (rgb) model_out_C + 3 (rgb) in_videos_C
        inp_6c_F = torch.cat((c_out, c_in), 1)  # 拼接 6 通道

        # 输出 reshape 回 (b,c,d,h,w) 并与粗网络结果做逐帧残差
        return self.net(inp_6c_F).reshape(b, c, d, h, w) + c_out
        # return self.net(inp_6c_F).reshape(b, c, d, h, w) + c_in


class C_C3D_1(nn.Module):
    """3D 卷积模块：对 6 通道视频做时间维积分，压缩时间维(seq_len)至 1 帧大小的特征。

    前接 conv1(5x5x5) -> conv2(3x3x3)+残差 -> conv3(3x3x3) -> conv4(3x3x3)+残差，
    最后 squeeze 掉时间维，把 (b,c,d,h,w) 压成 (b,c,h,w)，供后续 2D 的 RRDB 精修。
    """

    def __init__(self, out_channels=64, input_channels=6):
        self.inplanes = out_channels
        super(C_C3D_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, out_channels, kernel_size=5, stride=1, padding=[1, 2, 2], bias=False),
            # nn.BatchNorm3d(out_channels),  # 原作者在此注释掉了 BN
            nn.ReLU(inplace=False),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=[0, 1, 1], bias=False),
            # 时间维 padding=0，即此处开始压缩时间长度
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x  # 残差连接
        x = self.conv3(x)      # 时间维因 padding=0 而减为 d-2
        x = self.conv4(x) + x  # 残差连接

        x = x.squeeze(2)  # 将时间维压缩(after conv3+conv4 时间维=1)，得到 (b,c,h,w)

        return x


if __name__ == '__main__':
    s = b, c, d, h, w = 1, 6, 5, 224, 224
    t = torch.ones(s).cuda()

    net = FineNet().cuda()
    output = net(t)
    import pdb; pdb.set_trace()


