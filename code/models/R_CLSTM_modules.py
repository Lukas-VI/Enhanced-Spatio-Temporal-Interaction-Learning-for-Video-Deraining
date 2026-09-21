# =============================================================================
# R_CLSTM_modules.py —— 时空交互精化模块（Recurrent ConvLSTM 系列）
#
# 这是论文"Enhanced Spatio-Temporal Interaction Learning for Video Deraining"
# 的核心组件。CoarseNet 得到解码器特征后，送入这里的循环卷积网络 R-CLSTM，
# 沿时间维逐帧处理，把上一帧的隐状态(h_state)递推传递到当前帧，从而
# 同时利用"空间结构"与"时间相关性"来做逐帧推断（论文中的 spatio-temporal interaction）。
#
# 教版本参考：
#   - R / R_2 / R_3：单步精化"单元"，负责把(状态+特征)映射为隐状态 h 和输出图。
#   - R_CLSTM_5：隐状态宽度 8（cell width = 8），采用 LSTM 式门控更新 C 状态，
#                再用 Q_t 控制 h_state = Refine(C_state, Q_t(input))。
#   - R_CLSTM_6：隐状态宽度与特征通道数相同（num_features），门控更复杂。
#
# 注意：文件下方有大量被注释掉的版本(R_CLSTM_6/7/8/9/10 等)是作者的实验残留，
#       仅供对比思路，不参与实际运行。
# =============================================================================
import torch.nn.functional as F
import torch.nn as nn
import torch
import time


def maps_2_cubes(x, b, d):
    """把 2D 特征图 (b*d, c, h, w) 恢复为视频立方体 (b, c, d, h, w)。"""
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d, x_c, x_h, x_w)  # (b, d, c, h, w)

    return x.permute(0, 2, 1, 3, 4)  # 还原为 (b, c, d, h, w)


def maps_2_maps(x, b, d):
    """把 (b*d, c, h, w) 按 d 个时间步合并为 (b, d*c, h, w)（沿通道拼接时间维）。"""
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d * x_c, x_h, x_w)

    return x


class R(nn.Module):
    """精化单元 R：输入为拼接后的特征，经两层 5x5 卷积得到隐状态 h，
    再经一层卷积输出预测图（在此早期版本中为深度图 pred_depth，输出 1 通道）。
    该版本输出通道为 1，属于开发早期原型，当前默认使用 R_CLSTM_5/6 而非它。
    """

    def __init__(self, block_channel):
        super(R, self).__init__()

        # 输入通道数 = 64(MFF输出) + 最后一块特征通道数/32
        num_features = 64 + block_channel[3] // 32
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)  # 输出 1 通道(深度图)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        h = F.relu(x1)  # 隐状态

        pred_depth = self.conv2(h)  # 预测图像

        return h, pred_depth


class R_2(nn.Module):
    """精化单元 R_2：R 的改进版，隐状态宽度扩展为 4 通道，
    并通过单独的 convh 卷积分支输出隐状态 h，分离出"图像预测"与"隐状态"两条支路。
    """

    def __init__(self, block_channel):
        super(R_2, self).__init__()

        num_features = 64 + block_channel[3] // 32 + 4  # 额外 +4 以容纳拼接进来的 4 通道隐状态
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)  # 输出图像(1通道)

        self.convh = nn.Conv2d(
            num_features, 4, kernel_size=3, stride=1, padding=1, bias=True)  # 输出 4 通道隐状态

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)          # 隐状态，用于 LSTM 式递推
        pred_depth = self.conv2(x1)  # 预测图像

        return h, pred_depth


# todo disable batchnorm before use
# class R_d(nn.Module):
#     def __init__(self, block_channel):
#         super(R_d, self).__init__()
#
#         num_features = 64 + block_channel[3] // 32 + 4
#         self.conv0 = nn.Conv2d(num_features, num_features,
#                                kernel_size=5, stride=1, padding=2, bias=False)
#         self.bn0 = nn.BatchNorm2d(num_features)
#
#         self.conv1 = nn.Conv2d(num_features, num_features,
#                                kernel_size=5, stride=1, padding=2, bias=False)
#         self.bn1 = nn.BatchNorm2d(num_features)
#
#         self.dropout = nn.Dropout2d(p=0.5)
#
#         self.conv2 = nn.Conv2d(
#             num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)
#
#         self.convh = nn.Conv2d(
#             num_features, 4, kernel_size=3, stride=1, padding=1, bias=True)
#
#     def forward(self, x):
#         x0 = self.conv0(x)
#         x0 = self.bn0(x0)
#         x0 = F.relu(x0)
#
#         x1 = self.conv1(x0)
#         x1 = self.bn1(x1)
#         x1 = F.relu(x1)
#
#         h = self.convh(x1)
#         x1 = self.dropout(x1)
#         pred_depth = self.conv2(x1)
#
#         return h, pred_depth


class R_3(nn.Module):
    """精化单元 R_3：R_CLSTM_5 的内部精化单元。

    输入通道 = 64 + block_channel[3]//32 + 8（+8 为拼接进来的 8 通道 c_state），
    输出 3 通道 RGB 图 out 与 8 通道隐状态 h。
    """

    def __init__(self, block_channel, use_bn=False):
        super(R_3, self).__init__()

        num_features = 64 + block_channel[3] // 32 + 8  # +8 容纳拼接的状态
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        # self.conv2 = nn.Conv2d(
        #     num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.conv2 = nn.Conv2d(
            num_features, 3, kernel_size=5, stride=1, padding=2, bias=True)  # 输出 3 通道 RGB

        self.convh = nn.Conv2d(
            num_features, 8, kernel_size=3, stride=1, padding=1, bias=True)  # 输出 8 通道隐状态

        self.use_bn = use_bn

    def forward(self, x):

        x0 = self.conv0(x)
        if self.use_bn:
            x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        if self.use_bn:
            x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)   # 隐状态（8 通道），回传给下一时间步
        out = self.conv2(x1) # 当前帧去除雨后的 RGB 输出

        return h, out


class R_CLSTM_5(nn.Module):
    """时空交互精化网络 R_CLSTM_5（cell width = 8，即隐/记忆状态为 8 通道）。

    采用类似 LSTM 的门控机制沿时间维循环：
      c_state = F_t * c_state_prev + I_t * C_t      (遗忘门 F_t + 输入门 I_t)
      h_state, frame = Refine(c_state, Q_t(input))
    其中 Q_t 用当前输入调制 c_state 得到最终隐状态，再由 R_3 输出当前帧的去雨图。
    逐帧收集所有时间步的输出，堆叠成 (b,3,d,h,w) 返回。
    """

    def __init__(self, block_channel, use_bn=False):
        super(R_CLSTM_5, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.Refine = R_3(block_channel, use_bn=use_bn)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,   # 输入 = 当前特征 + 8 通道隐状态
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()  # 遗忘门，值域 [0,1]，控制丢弃多少旧记忆
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()  # 输入门，控制写入多少新信息
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()  # 候选记忆(新内容)，值域 [-1,1]
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=num_features,   # 映射回 num_features 以匹配 R_3 输入
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()  # 输出调制门，用当前输入调制记忆，得到 h 的输入
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)  # 恢复为 (b,c,d,h,w)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, 8, h, w).to('cuda')  # 初始隐状态
        c_state_init = torch.zeros(b, 8, h, w).to('cuda')  # 初始记忆状态

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            # 拼接当前帧特征与上一时刻隐状态
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
            # LSTM 记忆更新：遗忘(旧记忆) + 写入(新内容)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            # 用 Q_t 调制后的记忆状态输入精化单元，得到当前帧输出与更新后的隐状态
            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)  # 收集当前帧输出

        layer_output = torch.stack(output_inner, dim=2)  # (b,3,d,h,w)

        return layer_output


class R_4(nn.Module):
    """精化单元 R_4：R_CLSTM_6 的内部单元（隐状态宽度 = num_features，即与特征通道一致）。

    与 R_3 不同，这里 conv0/conv1 都带残差连接(x0+x 与 x1+x0)，属于残差式精化单元。
    """

    def __init__(self, block_channel, use_bn=True):
        super(R_4, self).__init__()

        num_features = (64 + block_channel[3] // 32) * 1
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 3, kernel_size=5, stride=1, padding=2, bias=True)  # 输出 3 通道 RGB

        self.use_bn = use_bn

    def forward(self, x):

        # first conv layer (-> bn -> relu)  第一层卷积(带残差)
        x0 = self.conv0(x) + x
        if self.use_bn:
            x0 = self.bn0(x0)
        x0 = F.relu(x0)

        # second conv layer (-> bn -> relu)  第二层卷积(带残差)
        x1 = self.conv1(x0) + x0
        if self.use_bn:
            x1 = self.bn1(x1)
        x1 = F.relu(x1)

        # output conv layer  输出卷积层
        out = self.conv2(x1)

        return out


class R_CLSTM_6(nn.Module):
    """时空交互精化网络 R_CLSTM_6（隐/记忆状态宽度 = num_features）。

    与 R_CLSTM_5 的区别：
      - 门控输出通道数为 num_features（而非 8），状态容量更大；
      - 隐状态更新采用 h_state = tanh(c_state) * Q_t(input)（经典 LSTM 输出门写法）；
      - 精化单元为残差式 R_4。
    """

    def __init__(self, block_channel, use_bn=True):
        super(R_CLSTM_6, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.num_features = num_features
        self.Refine = R_4(block_channel, use_bn=use_bn)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,  # 当前特征 + 隐状态
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()  # 遗忘门
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()  # 输入门
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()  # 候选记忆
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()  # 输出门
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)  # 恢复为 (b,c,d,h,w)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, self.num_features, h, w).to('cuda')  # 初始隐状态
        c_state_init = torch.zeros(b, self.num_features, h, w).to('cuda')  # 初始记忆状态

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)  # 拼接当前帧特征+隐状态
            # LSTM 记忆更新
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            # 隐状态：tanh(c_state) 经输出门 Q_t 调制（经典 LSTM 输出门）
            h_state = torch.tanh(c_state) * self.Q_t(input_cat)

            # output image  由残差精化单元 R_4 输出当前帧去雨图
            o_state = self.Refine(h_state)
            output_inner.append(o_state)

        layer_output = torch.stack(output_inner, dim=2)  # (b,3,d,h,w)

        return layer_output

# class R_cell(nn.Module):
#     def __init__(self, block_channel, cell_width=16):
#         super(R_cell, self).__init__()
#
#         num_features = 64 + block_channel[3] // 32
#         self.conv0 = nn.Conv2d(num_features + cell_width, num_features,
#                                kernel_size=5, stride=1, padding=2, bias=False)
#         self.bn0 = nn.BatchNorm2d(num_features)
#
#         self.conv1 = nn.Conv2d(num_features, num_features,
#                                kernel_size=5, stride=1, padding=2, bias=False)
#         self.bn1 = nn.BatchNorm2d(num_features)
#
#         self.conv2 = nn.Conv2d(
#             num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)
#
#         self.convh = nn.Conv2d(
#             num_features, cell_width, kernel_size=3, stride=1, padding=1, bias=True)
#
#     def forward(self, x):
#         x0 = self.conv0(x)
#         x0 = self.bn0(x0)
#         x0 = F.relu(x0)
#
#         x1 = self.conv1(x0)
#         x1 = self.bn1(x1)
#         x1 = F.relu(x1)
#
#         h = self.convh(x1)
#         pred_depth = self.conv2(x1)
#
#         return h, pred_depth


# class R_CLSTM_6(nn.Module):
#     def __init__(self, block_channel):
#         super(R_CLSTM_6, self).__init__()
#         num_features = 64 + block_channel[3] // 32
#         self.cell_width = 8
#         self.Refine = R_cell(block_channel, self.cell_width)
#         self.F_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.I_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.C_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Tanh()
#         )
#         self.Q_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=num_features,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input_tensor, b, d):
#         input_tensor = maps_2_cubes(input_tensor, b, d)
#         b, c, d, h, w = input_tensor.shape
#         h_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
#         c_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
#
#         seq_len = d
#
#         h_state, c_state = h_state_init, c_state_init
#         output_inner = []
#         for t in range(seq_len):
#             input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
#             c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)
#
#             h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))
#
#             output_inner.append(p_depth)
#
#         layer_output = torch.stack(output_inner, dim=2)
#
#         return layer_output
#
#
# class R_CLSTM_7(nn.Module):
#     def __init__(self, block_channel):
#         super(R_CLSTM_7, self).__init__()
#         num_features = 64 + block_channel[3] // 32
#         self.cell_width = 12
#         self.Refine = R_cell(block_channel, self.cell_width)
#         self.F_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.I_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.C_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Tanh()
#         )
#         self.Q_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=num_features,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input_tensor, b, d):
#         input_tensor = maps_2_cubes(input_tensor, b, d)
#         b, c, d, h, w = input_tensor.shape
#         h_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
#         c_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
#
#         seq_len = d
#
#         h_state, c_state = h_state_init, c_state_init
#         output_inner = []
#         for t in range(seq_len):
#             input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
#             c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)
#
#             h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))
#
#             output_inner.append(p_depth)
#
#         layer_output = torch.stack(output_inner, dim=2)
#
#         return layer_output
#
#
# class R_CLSTM_8(nn.Module):
#     def __init__(self, block_channel):
#         super(R_CLSTM_8, self).__init__()
#         num_features = 64 + block_channel[3] // 32
#         self.cell_width = 16
#         self.Refine = R_cell(block_channel, self.cell_width)
#         self.F_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.I_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.C_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Tanh()
#         )
#         self.Q_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=num_features,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input_tensor, b, d):
#         input_tensor = maps_2_cubes(input_tensor, b, d)
#         b, c, d, h, w = input_tensor.shape
#         h_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
#         c_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
#
#         seq_len = d
#
#         h_state, c_state = h_state_init, c_state_init
#         output_inner = []
#         for t in range(seq_len):
#             input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
#             c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)
#
#             h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))
#
#             output_inner.append(p_depth)
#
#         layer_output = torch.stack(output_inner, dim=2)
#
#         return layer_output
#
#
# class R_CLSTM_9(nn.Module):
#     def __init__(self, block_channel):
#         super(R_CLSTM_9, self).__init__()
#         num_features = 64 + block_channel[3] // 32
#         self.cell_width = 32
#         self.Refine = R_cell(block_channel, self.cell_width)
#         self.F_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.I_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.C_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=self.cell_width,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Tanh()
#         )
#         self.Q_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + self.cell_width,
#                       out_channels=num_features,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input_tensor, b, d):
#         input_tensor = maps_2_cubes(input_tensor, b, d)
#         b, c, d, h, w = input_tensor.shape
#         h_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
#         c_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
#
#         seq_len = d
#
#         h_state, c_state = h_state_init, c_state_init
#         output_inner = []
#         for t in range(seq_len):
#             input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
#             c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)
#
#             h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))
#
#             output_inner.append(p_depth)
#
#         layer_output = torch.stack(output_inner, dim=2)
#
#         return layer_output
#
#
# class R_CLSTM_10(nn.Module):
#     def __init__(self, block_channel):
#         super(R_CLSTM_10, self).__init__()
#         num_features = 64 + block_channel[3] // 32
#         self.Refine = R_10(block_channel)
#         self.F_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + 8,
#                       out_channels=8,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.I_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + 8,
#                       out_channels=8,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#         self.C_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + 8,
#                       out_channels=8,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Tanh()
#         )
#         self.Q_t = nn.Sequential(
#             nn.Conv2d(in_channels=num_features + 8,
#                       out_channels=num_features,
#                       kernel_size=3,
#                       padding=1,
#                       ),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input_tensor, b, d):
#         input_tensor = maps_2_cubes(input_tensor, b, d)
#         b, c, d, h, w = input_tensor.shape
#         h_state_init = torch.zeros(b, 8, h, w).to('cuda')
#         c_state_init = torch.zeros(b, 8, h, w).to('cuda')
#
#         seq_len = d
#
#         h_state, c_state = h_state_init, c_state_init
#         output_inner = []
#         for t in range(seq_len):
#             input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
#             c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)
#
#             h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))
#
#             output_inner.append(p_depth)
#
#         layer_output = torch.stack(output_inner, dim=2)
#
#         return layer_output
