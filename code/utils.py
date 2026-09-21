# =============================================================================
# utils.py —— 训练/测试的通用工具函数
#
# 提供：模型构建 build_model、数据加载 get_train/test_loader、断点保存/恢复
# load/save_checkpoint、优化器构建 build_optimizer、学习率调整、
# 图像指标 PSNR 计算、ImageNet 归一化的反归一化/裁剪等。
# 逻辑上把 backbone/build 的细节从 train/test 中抽离出来，多处复用。
# =============================================================================
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn.parallel
from PIL import Image
from torchvision import transforms

from dataset import ntu_dataset
from models import modules
from models import net
from models.backbone_dict import backbone_dict
from models.net import FineNet, FineNet_npic


class AverageMeter(object):
    """累加平均统计器：记录并维护 val/sum/count/avg，常用于 loss、PSNR、耗时等统计。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n   # 加权求和
        self.count += n
        self.avg = self.sum / self.count  # 加权平均


# ==== model utils ====
def build_model(args):
    """根据配置构建粗网络 net_C 与精网络 net_F。

    粗网络由 backbone(编码器) + 解码器 + CoarseNet(含 R-CLSTM 时空精化) 组成，
    精网络根据 args.F_npic 决定用 FineNet_npic(输出多帧) 或 FineNet(输出中间帧)。
    返回 (net_coarse, net_fine)。
    """
    if args.backbone.startswith('resnet'):
        backbone = backbone_dict[args.backbone](use_bn=args.use_bn, model_dir=args.torch_home)
        encoder = modules.E_resnet(backbone, use_bn=args.use_bn)

        if args.backbone in ['resnet50']:
            decoder = modules.D_resnet(num_features=2048)
            net_coarse = net.CoarseNet(encoder,
                                       decoder,
                                       block_channel=[256, 512, 1024, 2048],  # ResNet50 各 stage 通道数
                                       refinenet=args.refinenet,
                                       bidirectional=args.use_bilstm,
                                       input_residue=args.input_residue
                                       )
        elif args.backbone in ['resnet18', 'resnet34']:
            decoder = modules.D_resnet(num_features=512)
            net_coarse = net.CoarseNet(encoder,
                                       decoder,
                                       block_channel=[64, 128, 256, 512],  # ResNet18/34 各 stage 通道数
                                       refinenet=args.refinenet,
                                       bidirectional=args.use_bilstm,
                                       input_residue=args.input_residue
                                       )

    elif args.backbone.startswith('densenet'):
        backbone = backbone_dict[args.backbone]()
        encoder = modules.E_densenet(backbone)

        if args.backbone in ['densenet121']:
            decoder = modules.D_densenet(num_features=1024, use_bn=True)
            net_coarse = net.CoarseNet(encoder,
                                       decoder,
                                       block_channel=[128, 256, 512, 1024],
                                       refinenet=args.refinenet,
                                       bidirectional=args.use_bilstm,
                                       input_residue=args.input_residue
                                       )
        elif args.backbone in ['densenet169']:
            decoder = modules.D_densenet(num_features=1664, use_bn=True)
            net_coarse = net.CoarseNet(encoder,
                                       decoder,
                                       block_channel=[128, 256, 640, 1664],
                                       refinenet=args.refinenet,
                                       bidirectional=args.use_bilstm,
                                       input_residue=args.input_residue
                                       )

    else:
        raise ValueError('Unrecognized backbone model name {}.'.format(args.backbone))

    # discriminator = net.C_C3D_1()  # 此处未启用判别器（此前调过的生成对抗思路的残留）

    # 按 F_npic 选择精网络：True 则输出多帧，False 则输出中间单帧
    if args.F_npic:
        net_fine = FineNet_npic()
    else:
        net_fine = FineNet()

    return net_coarse, net_fine


# ==== data utilities =====
# ImageNet 归一化所用的均值/标准差（输入图像按此标准化）
__imagenet_stats = {'mean': torch.tensor([0.485, 0.456, 0.406]),
                    'std': torch.tensor([0.229, 0.224, 0.225])
                    }


def get_train_loader(args):
    """构造训练 DataLoader：ToTensor + ImageNet 归一化，且带随机裁剪增强。"""
    transform = transforms.Compose([transforms.ToTensor(),          # PIL -> Tensor [0,1]
                                    transforms.Normalize(__imagenet_stats['mean'],   # 标准化
                                                         __imagenet_stats['std'])
                                    ]
                                   )

    dataset = ntu_dataset.NTU_dataset(data_root=args.data_root,
                                      indexfile_path=args.train_file,
                                      transform=transform,
                                      crop_size=args.crop_size,
                                      is_testing=False
                                      )

    # dataset, dataloader
    train_dataloader = ntu_dataset.get_dataloader(dataset=dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,      # 训练打乱
                                                  num_workers=8
                                                  )

    return train_dataloader


def get_test_loader(args):
    """构造验证/测试 DataLoader：关闭所有增强，batch_size=1，不打乱。"""
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(__imagenet_stats['mean'],
                                                         __imagenet_stats['std'])
                                    ]
                                   )

    dataset = ntu_dataset.NTU_dataset(data_root=args.data_root,
                                      indexfile_path=args.eval_file,
                                      transform=transform,
                                      apply_crop=False,
                                      apply_horizontal_flip=False,
                                      apply_coloraug=False,
                                      apply_rotation=False,
                                      is_testing=True
                                      )

    # dataset, dataloader
    train_dataloader = ntu_dataset.get_dataloader(dataset=dataset, batch_size=1, shuffle=False,
                                                  num_workers=8)

    return train_dataloader


# ==== checkpoint utils ======
def save_checkpoint(state, filename):
    """将 state_dict 保存为断点文件。"""
    torch.save(state, filename)


def load_checkpoint(args, model, checkpoint_dir=None, pretrain_ckpt=None):
    """加载断点/预训练权重，返回应继续的起始 epoch。

    逻辑：
      - args.resume 为真：从 checkpoint_dir 中取"最新"(epoch 最大)的断点续训；
      - 否则若指定 pretrain_ckpt/-loadckpt：加载指定权重做初始化(测试/微调)；
      - 否则从第 0 epoch 开始。
    """
    start_epoch = 0

    if args.resume:
        all_saved_ckpts = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.endswith(".pth.tar")]

        if len(all_saved_ckpts) == 0:  # 目录为空则从头开始
            logging.info("start at epoch {}".format(start_epoch))
            return start_epoch

        logging.info(all_saved_ckpts)
        # 按文件名中的 epoch 数字排序，取最大
        all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        loadckpt = os.path.join(checkpoint_dir, all_saved_ckpts[-1])
        start_epoch = int(all_saved_ckpts[-1].split('_')[-1].split('.')[0])  # 续训从这个 epoch 开始
        logging.info("loading the lastest model in checkpoint_dir: {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict)
    elif args.loadckpt is not None:
        logging.info("loading model {}".format(args.loadckpt))

        start_epoch = pretrain_ckpt.split('_')[-1].split('.')[0]  # 从文件名解析 epoch（测试时通常不关心）
        state_dict = torch.load(pretrain_ckpt)
        model.load_state_dict(state_dict)
    else:
        logging.info("start at epoch {}".format(start_epoch))

    return start_epoch


# ==== optimization =====
def build_optimizer(model,
                    learning_rate,
                    optimizer_name='rmsprop',
                    weight_decay=1e-5,
                    epsilon=0.001,
                    momentum=0.9):
    """根据名称构建优化器（支持 sgd / rmsprop / adam）。"""
    if optimizer_name == "sgd":
        print("Using SGD optimizer.")
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    elif optimizer_name == 'rmsprop':
        print("Using RMSProp optimizer.")
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=learning_rate,
                                        eps=epsilon,
                                        weight_decay=weight_decay,
                                        momentum=momentum
                                        )
    elif optimizer_name == 'adam':
        print("Using Adam optimizer.")
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def adjust_learning_rate(optimizer, epoch, init_lr):
    """学习率调整：当前实现为恒定的 init_lr（分段衰减被注释掉）。
    # lr = init_lr * (0.1 ** (epoch // 5))   每 5 个 epoch 降为 1/10（被作者注释）
    #
    # for param_group in optimizer.param_groups:
    # 	param_group['lr'] = lr

    """
    return init_lr  # 直接返回初值，即训练全程学习率不变


# ==== metrics ====
def calculate_psnr(img1, img2):
    """计算两个图像之间的 PSNR（先反归一化到 [0,255] 再算）。"""
    if isinstance(img1, torch.Tensor):
        img1 = tensor2array(img1)  # 若是 Tensor 先转成 numpy 数组
    if isinstance(img2, torch.Tensor):
        img2 = tensor2array(img2)

    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)  # 均方误差

    if mse == 0:
        return float('inf')  # 完全相同则 PSNR 为无穷大
    return 20 * math.log10(255.0 / math.sqrt(mse))


def tensor2img(t):
    """把(反归一化后的)张量转为 PIL 图像，便于保存。"""
    unnormalize = transforms.Normalize((-__imagenet_stats['mean'] / __imagenet_stats['std']).tolist(),
                                       (1.0 / __imagenet_stats['std']).tolist()
                                       )  # 反归一化(把标准化还原回 [0,1])

    img = transforms.ToPILImage()(unnormalize(t).cpu())
    return img


def tensor2array(t):
    """把张量反归一化并转为 numpy 数组（数值范围回到 [0,255]）。"""
    transform_unnorm = transforms.Normalize((-__imagenet_stats['mean'] / __imagenet_stats['std']).tolist(),
                                            (1.0 / __imagenet_stats['std']).tolist()
                                            )

    t = np.array(transforms.ToPILImage()(transform_unnorm(t).detach().cpu()))
    return t


# ==== misc ====
def makedir(directory):
    """若目录不存在则递归创建。"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_image(t, filename):
    """把张量保存为图像文件。"""
    img = tensor2img(t)
    img.save(filename)


def calculate_psnr_on_tensors(t1, t2):
    """对两个张量直接计算 PSNR（内部先转数组）。"""
    a1 = tensor2array(t1)
    a2 = tensor2array(t2)

    return calculate_psnr(a1, a2)


# clamp  # 对 ImageNet 归一化后的 RGB 做上下限截断。
# 取值范围由 mean ± 3*std 大致推出（最小/最大允许像素值）。
min_rgb = torch.tensor([-2.1179, -2.0357, -1.8044]).unsqueeze(-1).unsqueeze(-1).cuda()
max_rgb = torch.tensor([2.2489, 2.4286, 2.6400]).unsqueeze(-1).unsqueeze(-1).cuda()


def clamp_on_imagenet_stats(t):
    """把张量的 RGB 值截断在 min_rgb 与 max_rgb 之间（保证角度反转时数值稳定/在合法域）。"""
    t = t.transpose(1, 2)  # (b,c,d,h,w)->(b,d,c,h,w)，便于逐通道广播截断

    t = torch.where(t < min_rgb, min_rgb, t)
    t = torch.where(t > max_rgb, max_rgb, t)

    t = t.transpose(1, 2)  # 还原 (b,c,d,h,w)

    return t


def set_random_seed(seed):
    """统一设置随机种子以保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)