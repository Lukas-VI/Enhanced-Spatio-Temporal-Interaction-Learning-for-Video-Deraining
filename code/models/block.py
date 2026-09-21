# =============================================================================
# block.py —— 精网络(FineNet)用的基础卷积块 / 残差密集块
#
# 提供激活函数、归一化、padding 的工厂函数，以及卷积块 conv_block、
# 残差块 ResNetBlock、残差密集块 RDB(RRDB) 等，参照 EDSR / ESRGAN 的实现。
# FineNet 的 RRDB 链条即由本文件的 RRDB 展开。
# =============================================================================
from collections import OrderedDict
import torch
import torch.nn as nn

####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # 激活函数选择：relu / leakyrelu / prelu
    # neg_slope: for leakyrelu and init of prelu   # LeakyReLU 斜率 / PReLU 初值
    # n_prelu: for p_relu num_parameters   # PReLU 的参数量
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # 归一化层选择：batch / instance
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # padding 层选择：reflect / replicate；padding=0 或 zero 时返回 None（交给卷积层自带 zero padding）
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    """根据卷积核大小与膨胀率计算等效 padding，使输出尺寸不变。"""
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)  # 有效卷积核尺寸
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # 把子模块输出与输入在通道维拼接（dense 式连接）
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    # 残差块：输入 + 子模块输出（逐元素相加）
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr



class ShortcutBlock_ZKH(nn.Module):
    # 双模块残差块：out = x + sub2(sub1(x))，本项目中用于"RRDB 链(sub1) + LR_conv(sub2)"的残差
    def __init__(self, submodule1, submodule2):
        super(ShortcutBlock_ZKH, self).__init__()
        self.sub1 = submodule1
        self.sub2 = submodule2

    def forward(self, x):
        output = x + self.sub2(self.sub1(x))
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub1.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr



def sequential(*args):
    # 把多个模块展平拼接成一个 nn.Sequential（自动解开嵌套的 nn.Sequential）
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.  # 只有一个模块直接返回
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):  # 展平嵌套的 Sequential
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    卷积块：padding + 卷积 + 归一化 + 激活 的组合
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)  # 计算保持尺寸的 padding
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None  # 非 zero 才额外加 padding 层
    padding = padding if pad_type == 'zero' else 0  # zero 模式由卷积自带 padding

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)  # Conv -> Norm -> Act
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
            # 重要：NAC 中不能用 inplace ReLU，否则会原地修改输入导致残差出错
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)  # Norm -> Act -> Conv


####################
# Useful blocks
####################


class ResNetBlock(nn.Module):
    '''
    ResNet 残差块（3-3 两卷积结构），带 EDSR 的残差缩放 res_scale（默认 1）
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None  # CNA 模式下第二层不再重复激活(最后无激活)（理解存疑）
        if mode == 'CNAC':  # Residual path: |-CNAC-|  # CNAC 残差路径模式下末层无 norm/act
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)  # 残差路径经比例缩放
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    '''
    残差密集块 RDB（5 个卷积，密集连接），RDB 的"核心卷积单元"
    出处：Residual Dense Network for Image Super-Resolution, CVPR 18
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels  # 生长通道数(中间特征宽度)
        # 前 4 个卷积输入通道逐级累加 gc（密集连接）
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        # 第 5 个卷积把 4 路拼接压缩回输入通道 nc
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        # 每层都把前面所有输出拼接进来（Dense 连接）
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x  # 局域残差，缩放系数 0.2


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block（嵌套残差密集块，3 个 RDB 叠加）
    出处：ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        # 三个级联的 RDB
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x  # 残差中的残差，缩放 0.2


####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle 上采样层（亚像素卷积），放大多倍分辨率
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)  # r^2 通道重排为 r 倍分辨率

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # 上采样 + 卷积 的组合（Up conv）
    # 见 https://distill.pub/2016/deconv-checkerboard/ 关于反卷积棋盘效应的讨论
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)  # 先插值放大
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)
