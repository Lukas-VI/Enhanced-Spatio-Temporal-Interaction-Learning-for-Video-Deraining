# ==============================================================================
# backbone_dict.py —— 骨干(编码器)基础网络注册表
# ------------------------------------------------------------------------------
# 提供小写 key -> 对应的 torchvision-style 骨干网络工厂函数 的映射。
# 供 modules 的 E_resnet / E_densenet 外壳包装骨干时使用，
# 例如 E_resnet 内部调用 backbone_dict['resnet18'] 来实例化基础 ResNet。
# 说明（理解存疑）：第一行 `from models import densenet121, densenet169`
# 会触发 models 包的 __init__，可能引起循环依赖/导入顺序问题；
# 但此处名义上只想引入已实例化的 densenet 网络函数，后被上面 dict 覆盖逻辑冗余。
# ==============================================================================
from models import densenet121, densenet169
from models.resnet import resnet18, resnet34, resnet50


backbone_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'densenet121': densenet121,
    'densenet169': densenet169
}