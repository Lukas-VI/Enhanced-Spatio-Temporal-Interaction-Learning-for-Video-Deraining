# =============================================================================
# modules.py —— 网络的编码器(Encoder)、解码器(Decoder)与多层特征融合(MFF)
#
# 编码器 E 取预训练 backbone(ResNet/DenseNet/SENet) 的卷积部分，输出多个分辨率的
# 特征(block0..block4)，作为"空间金字塔"多尺度表征。
# 解码器 D 采用 LapSRN 式的逐级上采样(_UpProjection + 跨层残差)，把深层小特征逐步
# 恢复到高分辨率；MFF 是把不同尺度的编码特征统一融合成固定通道的特征。
# 这些模块与 R-CLSTM 组成论文中的粗网络(coarse stage)。
# =============================================================================
import torch
import torch.nn.functional as F
import torch.nn as nn


class _UpProjection(nn.Sequential):
    """上采样-精化单元：先把特征双线性上采样到指定 size，
    再经"主分支(conv1->conv1_2)"与"捷径分支(conv2)"叠加融合，得到上采样后特征。

    类似 LapSRN / sub-pixel 中的上投影单元，用于解码器逐级恢复分辨率。
    """

    def __init__(self, num_input_features, num_output_features, use_bn=False):
        super(_UpProjection, self).__init__()

        # 主分支第一层 5x5 卷积
        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        # 主分支第二层 3x3 卷积
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        # 捷径分支：直接 5x5 卷积
        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

        self.use_bn = use_bn

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)  # 上采样到目标尺寸

        # 主分支与捷径分支（可选择是否接 BN）
        if not self.use_bn:
            x_conv1 = self.relu(self.conv1(x))
            bran1 = self.conv1_2(x_conv1)
            bran2 = self.conv2(x)
        else:
            x_conv1 = self.relu(self.bn1(self.conv1(x)))
            bran1 = self.bn1_2(self.conv1_2(x_conv1))
            bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)  # 两条分支求和后激活

        return out


class E_resnet(nn.Module):
    """ResNet 编码器：复用预训练 ResNet 的卷积主干，输出 5 层不同分辨率的特征。

    block0(原始分辨率) -> block1/2/3/4 依次下采样，供解码器的跨层残差与 MFF 使用。
    """

    def __init__(self, original_model, num_features=2048, use_bn=False):
        super(E_resnet, self).__init__()
        # 从预训练 ResNet 中抽取各阶段
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)

        if self.use_bn:
            x = self.bn1(x)

        x = self.relu(x)
        x_block0 = x           # block0：conv1 后的特征，分辨率最高
        x = self.maxpool(x)

        x_block1 = self.layer1(x)   # 各 stage 输出不同尺度
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block0, x_block1, x_block2, x_block3, x_block4


class E_densenet(nn.Module):
    """DenseNet 编码器：按索引手工拆解预训练 DenseNet 的 features 序列，
    分别取出 denseblock 与 transition 之间的特征，得到 5 层不同尺度的特征。
    """

    def __init__(self, original_model, num_features=2208):
        super(E_densenet, self).__init__()
        self.features = original_model.features

    def forward(self, x):
        # 前 4 个操作(conv0->norm0->relu0->pool0)得到的特征
        x01 = self.features[0](x)
        x02 = self.features[1](x01)
        x03 = self.features[2](x02)
        x04 = self.features[3](x03)

        # denseblock1: features[4] 为 denseblock 的整体，features[5] 含其后的 transition
        x_block1 = self.features[4](x04)
        x_block1 = self.features[5][0](x_block1)  # 逐层取 denseblock1 内各 _DenseLayer
        x_block1 = self.features[5][1](x_block1)
        x_block1 = self.features[5][2](x_block1)
        x_tran1 = self.features[5][3](x_block1)   # transition1（下采样）

        # denseblock2: 同理
        x_block2 = self.features[6](x_tran1)
        x_block2 = self.features[7][0](x_block2)
        x_block2 = self.features[7][1](x_block2)
        x_block2 = self.features[7][2](x_block2)
        x_tran2 = self.features[7][3](x_block2)

        # denseblock3: 同理
        x_block3 = self.features[8](x_tran2)
        x_block3 = self.features[9][0](x_block3)
        x_block3 = self.features[9][1](x_block3)
        x_block3 = self.features[9][2](x_block3)
        x_tran3 = self.features[9][3](x_block3)

        # denseblock4 + 最后 BN，作为最深层特征
        x_block4 = self.features[10](x_tran3)
        x_block4 = F.relu(self.features[11](x_block4))

        x_block0 = x03  # 用 relu0 输出作为最浅层(高分辨率)特征

        return x_block0, x_block1, x_block2, x_block3, x_block4


class E_senet(nn.Module):

    def __init__(self, original_model, num_features=2048):
        super(E_senet, self).__init__()
        # 取 SENet 除 avg_pool/dropout/fc 之外的卷积主干（layer0..layer4）
        self.base = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.base[0](x)      # layer0
        x_block1 = self.base[1](x)  # layer1
        x_block2 = self.base[2](x_block1)  # layer2
        x_block3 = self.base[3](x_block2)  # layer3
        x_block4 = self.base[4](x_block3)  # layer4

        return x_block1, x_block2, x_block3, x_block4


class D_densenet(nn.Module):
    """DenseNet 解码器：对最深层特征做 1x1 卷积(残差)后，用 5 个 _UpProjection
    逐级上采样，并在每级与对应编码层特征相加(跨层残差)，最终恢复到 2 倍分辨率。
    """

    def __init__(self, num_features=2048, use_bn=False):
        super(D_densenet, self).__init__()
        # self.conv = nn.Conv2d(num_features, num_features //
        #                       2, kernel_size=1, stride=1, bias=False)
        # num_features = num_features // 2
        # （保留作者注释：曾计划用 1x1 降维）

        self.conv = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, bias=False)
        # num_features = num_features

        self.bn = nn.BatchNorm2d(num_features)

        # 每级上采样输出通道减半
        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up5 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)

        self.use_bn = use_bn

    def forward(self, x_block0, x_block1, x_block2, x_block3, x_block4):
        # 最深层特征 1x1 卷积 + 残差
        if self.use_bn:
            x_d0 = F.relu(self.bn(self.conv(x_block4))) + x_block4
        else:
            x_d0 = F.relu(self.conv(x_block4)) + x_block4

        # 逐级上采样并叠加对应编码层特征（跨层跳跃连接）
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)]) + x_block3
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)]) + x_block2
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)]) + x_block1
        x_d4 = self.up4(x_d3, [x_block0.size(2), x_block0.size(3)]) + x_block0
        x_d5 = self.up5(x_d4, [x_block0.size(2) * 2, x_block0.size(3) * 2])  # 再放大 2 倍

        return x_d5


class D_resnet(nn.Module):
    """ResNet 解码器：与 D_densenet 类似，逐级上采样 + 跨层残差，
    最终恢复到 2 倍输入分辨率（最后一级输出通道为 num_features//4）。
    """

    def __init__(self, num_features=2048, use_bn=False):
        super(D_resnet, self).__init__()
        # self.conv = nn.Conv2d(num_features, num_features //
        #                       2, kernel_size=1, stride=1, bias=False)
        # num_features = num_features // 2
        # （保留作者注释）

        self.conv = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, bias=False)
        # num_features = num_features

        self.bn = nn.BatchNorm2d(num_features)

        # 前三级逐级减半输出通道
        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2, use_bn=use_bn)
        num_features = num_features // 2

        # 第 4 级保持通道数（因为此时已接近浅层，减少降维）
        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features, use_bn=use_bn)

        # 最后一级输出 num_features//4 通道，作为 R-CLSTM 的输入特征（理解存疑：最终通道数选择依据）
        self.up5 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 4)

        self.use_bn = use_bn

    def forward(self, x_block0, x_block1, x_block2, x_block3, x_block4):
        # 最深层 1x1 卷积 + 残差
        if self.use_bn:
            x_d0 = F.relu(self.bn(self.conv(x_block4))) + x_block4
        else:
            x_d0 = F.relu(self.conv(x_block4)) + x_block4

        # 逐级上采样 + 跨层残差
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)]) + x_block3
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)]) + x_block2
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)]) + x_block1
        x_d4 = self.up4(x_d3, [x_block0.size(2), x_block0.size(3)]) + x_block0
        x_d5 = self.up5(x_d4, [x_block0.size(2) * 2, x_block0.size(3) * 2])  # 放大 2 倍

        return x_d5


class MFF(nn.Module):
    """多层特征融合模块(Multi-scale Feature Fusion)：把编码器输出的多尺度特征
    (block1..block4) 各自上采样到同一尺寸(size)，每个缩放为固定 16 通道后再拼接
    回 64 通道(4x16)，经 5x5 卷积融合为 64 通道特征，作为 R-CLSTM 的输入之一。
    """

    def __init__(self, block_channel, num_features=64):
        super(MFF, self).__init__()

        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        # 4 路各 16 通道拼接 = 64 通道，再用 5x5 卷积融合
        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size, use_bn=False):
        # 各尺度上采样到统一尺寸并压缩到 16 通道
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        # 拼接后 5x5 卷积融合（可选 BN）
        if use_bn:
            x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        else:
            x = self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1))
        x = F.relu(x)

        return x
