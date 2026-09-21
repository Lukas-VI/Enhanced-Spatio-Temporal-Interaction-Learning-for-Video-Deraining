# ==============================================================================
# options/__init__.py —— 参数解析统一入口
# ------------------------------------------------------------------------------
# 根据 mode 分发到 trainopt.py 或 testopt.py 解析命令行参数，
# 返回一个 argparse.Namespace 对象 args（脚本中通过 args.xxx 读取配置）。
# ==============================================================================
from .trainopt import _get_train_opt
from .testopt import _get_test_opt

def get_args(mode):
    """按模式返回命令行解析结果：'train' -> 训练配置，'test' -> 测试配置。"""
    args = None
    if mode == 'train':
        args =  _get_train_opt()
    elif mode == 'test':
        args = _get_test_opt()
    else:
        raise ValueError("Invalid mode selection!")

    return args
