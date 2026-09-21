# ==============================================================================
# models/__init__.py —— 编码器(骨干)模型注册表与工厂函数
# ------------------------------------------------------------------------------
# 本文件把多种骨干编码器(ResNet / DenseNet / SENet)与对应的「编码器外壳」
# (modules 里的 E_resnet / E_densenet / E_senet) 组合起来，注册成字典。
# get_models(args) 按 --backbone 字符串实例化对应的编码器。
#
# 说明（理解存疑）：
#   - 此处使用 keys 为 'ResNet18'/'DenseNet121' 等驼峰命名，而 options/训练脚本
#     里 --backbone 默认值是 'resnet18' 等小写形式（见 backbone_dict.py 的键名）。
#     若命令行直接传入小写骨干名，这里会 KeyError；使用时需与 __models_small__
#     的键名保持一致。
#   - 引用了 args.pretrained_dir / os.getenv('TORCH_MODEL_ZOO')，该项参数在
#     trainopt.py / testopt.py 中并未定义，仅为保留逻辑，实际不影响主干运行。
# ==============================================================================
import os
import torch
import torchvision
from models.modules import E_resnet, E_densenet, E_senet
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.densenet import densenet161, densenet121, densenet169, densenet201
from models.senet import senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
import pdb

# 骨干编码器注册表：键为字符串名，值为「懒加载」的工厂函数，
# 调用时内部实例化对应骨干(带 ImageNet 预训练)并用 E_* 外壳包装成编码器。
__models_small__ = {
	'ResNet18': lambda :E_resnet(resnet18(pretrained = True)),
	'ResNet34': lambda :E_resnet(resnet34(pretrained = True)),
	'ResNet50': lambda :E_resnet(resnet50(pretrained = True)),
	'ResNet101': lambda :E_resnet(resnet101(pretrained = True)),
	'ResNet152': lambda :E_resnet(resnet152(pretrained = True)),
	'DenseNet121': lambda :E_densenet(densenet121(pretrained = True)),
	'DenseNet161': lambda :E_densenet(densenet161(pretrained = True)),
	'DenseNet169': lambda :E_densenet(densenet169(pretrained = True)),
	'DenseNet201': lambda :E_densenet(densenet201(pretrained = True)),
	'SENet154': lambda :E_senet(senet154(pretrained="imagenet")),
	'SE_ResNet50': lambda :E_senet(se_resnet50(pretrained="imagenet")),
	'SE_ResNet101': lambda :E_senet(se_resnet101(pretrained="imagenet")),
	'SE_ResNet152': lambda :E_senet(se_resnet152(pretrained="imagenet")),
	'SE_ResNext50_32x4d': lambda :E_senet(se_resnext50_32x4d(pretrained="imagenet")),
	'SE_ResNext101_32x4d': lambda :E_senet(se_resnext101_32x4d(pretrained="imagenet"))
}   


def get_models(args):
    """按 args.backbone 从注册表返回对应的编码器实例。"""
    backbone = args.backbone

    # 设置模型下载缓存的 TORCH_MODEL_ZOO 环境变量（保留逻辑，实际参数未定义）
    if os.getenv('TORCH_MODEL_ZOO') != args.pretrained_dir:
        os.environ['TORCH_MODEL_ZOO'] = args.pretrained_dir
    else:
        pass

    return __models_small__[backbone]()

