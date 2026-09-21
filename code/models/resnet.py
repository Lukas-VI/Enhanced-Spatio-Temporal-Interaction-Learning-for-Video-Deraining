# ==============================================================================
# resnet.py —— 借用的标准 ResNet 骨干实现
# ------------------------------------------------------------------------------
# 本文件为从 torchvision 借用的经典 ResNet 实现（为适配本项目做了轻微改写：
# 每个 block 增加 use_bn 开关，决定是否使用 BatchNorm，供编码器外壳 E_resnet 复用）。
# 提供 resnet18/34/50/101/152 五个版本，可选 ImageNet 预训练权重。
# 本项目只用其「特征提取(卷积部分)」能力，分类头(fc)在 E_resnet 中被截断。
# 属借用实现，此处仅做概括性注释，不逐行展开。
# ==============================================================================
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


# 各版本预训练权重的下载地址
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    # 3x3、padding=1、无偏置的卷积，保持分辨率不变的常用卷积
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    # 两层 3x3 卷积的残差基本块（用于 resnet18/34）
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_bn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)          
        self.bn1 = nn.BatchNorm2d(planes)                       
        self.relu = nn.ReLU(inplace=True)       
        self.conv2 = conv3x3(planes, planes)                    
        self.bn2 = nn.BatchNorm2d(planes)                       
        self.downsample = downsample       # 当维度/分辨率改变时用于对齐短接的模块
        self.stride = stride

        self.use_bn = use_bn              # use_bn 开关决定是否启用 BatchNorm

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        if self.use_bn:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)

        # 输入需要降采样或改通道时，用 downsample 对齐后做残差相加
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 三层「1x1-3x3-1x1」的瓶颈残差块（用于 resnet50/101/152）
    expansion = 4      # 瓶颈块会把通道数放大 4 倍

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_bn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)              
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,             
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)            
        self.bn3 = nn.BatchNorm2d(planes * 4)                                            
        self.relu = nn.ReLU(inplace=True)                                                   
        self.downsample = downsample
        self.stride = stride

        self.use_bn = use_bn

    def forward(self, x):
        residual = x

        # 与 BasicBlock 相同的前向逻辑，只是多一层 1x1 卷积
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # 完整 ResNet 主干：conv1 + maxpool + 4 个 stage(layer1~4) + avgpool + fc
    def __init__(self, block, layers, num_classes=1000, use_bn=False):
        self.inplanes = 64
        self.use_bn = use_bn
        super(ResNet, self).__init__()
        # 输入层：7x7 大卷积 + BN + ReLU + 最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,                 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                    
        # 四个 stage，层数由 layers=[...] 指定，后三个 stage 步长为 2 逐级降分辨率
        self.layer1 = self._make_layer(block, 64, layers[0])                               
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化：卷积用 He 初始化，BN 的 weight 置 1、bias 置 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        # 构建一个 stage：首块可能带 1x1 downsample，用于对齐通道/分辨率
        downsample = None
        # stride=1, 64 != 256
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(                                  
                nn.Conv2d(self.inplanes, planes * block.expansion,                  
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),                           
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_bn=self.use_bn))
        self.inplanes = planes * block.expansion        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_bn=self.use_bn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if self.use_bn:
            x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化 -> 展平 -> 全连接分类
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(pretrained=True, use_bn=False, model_dir=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], use_bn=use_bn, **kwargs)
    if pretrained:
        # 加载 ImageNet 预训练，只取与当前模型形状兼容的层（本项目的 use_bn/fc 裁剪后需这样过滤）
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir)
        model_state = model.state_dict()
        selected_state = OrderedDict()
        for k, v in pretrained_state.items():
            if k in model_state and v.size() == model_state[k].size():
                #print('pretrain..',k)
                selected_state[k] = v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


def resnet34(pretrained=False, model_dir=None, *kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir=model_dir))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
