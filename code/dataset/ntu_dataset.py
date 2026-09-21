# =============================================================================
# ntu_dataset.py —— 视频去雨序列数据集（NTU 数据集）
#
# 数据来自 indexfile(.json) 中描述的帧路径，格式是一个由若干视频实例组成的列表，
# 每个实例是一串帧的字典列表：{ 'rain': 含雨帧路径, 'gt': 干净帧路径 }。
#
# 每个样本"读入一段连续 window_size(默认5)帧"的含雨视频及其 GT，整段 clip 应用
# 相同的随机变换(裁剪/翻转/旋转)，最后返回张量为 (C, T, H, W)，其中 T = window_size。
# 该 T 维即模型 (b,c,d,h,w) 中的时间维 d，供 CoarseNet 的时空交互模块使用。
# =============================================================================
import random
import os
import json

import PIL
import torch
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torchvision.transforms.functional as TF


def pil_loader(path):
    """用 PIL 打开图像文件。"""
    img = Image.open(path)

    return img


def load_frames(root_dir, filenames):
    """按文件名列表读取一整套视频帧（返回 PIL.Image 列表）。"""
    video = []
    for fn in filenames:
        image_path = os.path.join(root_dir, fn)
        if os.path.exists(image_path):
            video.append(pil_loader(image_path))
        else:
            raise ValueError('File {} not exists.'.format(image_path))

    return video


class NTU_dataset(Dataset):
    """NTU 视频去雨序列数据集。

    一次取一个"时间窗口"(window_size 帧)的多张含雨帧及其干净 GT 帧，
    作为单个样本(batch 元素)。
    """

    def __init__(self,
                 data_root,
                 indexfile_path,
                 window_size=5,
                 transform=None,
                 crop_size=224,
                 apply_crop=True,
                 apply_rotation=False,
                 apply_coloraug=False,
                 apply_horizontal_flip=True,
                 is_testing=False
                 ):
        self.data_root = data_root       # 图像根目录（json 中的路径相对该目录）
        self.indexfile = indexfile_path  # 描述样本的 json 文件路径

        self.window_size = window_size   # 时间窗口长度（一段 clip 的帧数，默认 5）
        self.padding = window_size // 2  # 测试时在首尾对称 padding 的帧数
        self.is_testing = is_testing

        self.crop_size = crop_size
        self.apply_crop = apply_crop
        self.apply_rotation = apply_rotation
        self.apply_coloraug = apply_coloraug
        self.apply_hflip = apply_horizontal_flip

        self.transform = transform

        if self.is_testing:
            # 测试/验证时不得应用任何数据增强
            assert not self.apply_crop and \
                   not self.apply_hflip and \
                   not self.apply_coloraug and \
                   not self.apply_rotation, "No transform should be applied during validation / testing."

        self.data = self.create_dataset()

    def create_dataset(self):
        """根据 json 索引构建样本表：把每个视频实例滑窗切成若干个"窗口样本"，
        每个窗口样本包含 window_size 帧的 rain 路径与 gt 路径。
        """
        data = {'input': [], 'gt': []}

        instances = json.load(open(self.indexfile))  # 读入所有视频实例

        for inst in instances:

            if self.is_testing:
                # 测试时在序列首尾复制 padding 帧，使对首帧也能预测（保持窗口完整）
                inst = inst[:self.padding] + inst + inst[-self.padding:]

            total_frames = len(inst)

            start = 0
            end = total_frames - self.window_size  # 最后一个可行窗口起点

            ip_entries, gt_entries = [], []
            for l in range(start, end + 1):
                # 取 [l, l+window_size) 这一窗口内的 rain 与 gt 帧路径
                ip_path = [f['rain'] for f in inst[l: l + self.window_size]]
                gt_path = [f['gt'] for f in inst[l: l + self.window_size]]

                ip_entries.append(ip_path)
                gt_entries.append(gt_path)

            data['input'].extend(ip_entries)
            data['gt'].extend(gt_entries)

        return data

    def __getitem__(self, idx):
        ip_data = self.data['input'][idx]
        gt_data = self.data['gt'][idx]

        ip_frames = load_frames(self.data_root, ip_data)  # 读含雨帧
        gt_frames = load_frames(self.data_root, gt_data)  # 读干净帧

        ip_tensors, gt_tensors = [], []

        # the whole clip should apply same transform.  # 整段 clip 使用同一套增强参数(保持帧间一致性)
        transform_params = self.get_transform_params(ip_frames[0].size[0], ip_frames[0].size[1])
        for ip_frame, gt_frame in zip(ip_frames, gt_frames):
            ip_frame, gt_frame = self.apply_transform(ip_frame, gt_frame, transform_params)

            ip_tensors.append(ip_frame)
            gt_tensors.append(gt_frame)

        # 逐帧张量堆叠：先 (T,C,H,W) 再转置为 (C,T,H,W)
        ip_tensors = torch.stack(ip_tensors, 0).permute(1, 0, 2, 3)
        gt_tensors = torch.stack(gt_tensors, 0).permute(1, 0, 2, 3)

        # tensors shape: Channel (C) x temporal window size (T) x H x W
        return ip_tensors, gt_tensors

    def __len__(self):
        return len(self.data['input'])

    def get_transform_params(self, w, h):
        """随机采样一套增强参数（裁剪起点、翻转标志、旋转角度），
        供整段 clip 的所有帧共用，确保时空一致性。
        """
        x0 = random.randint(0, w - self.crop_size)  # 随机裁剪左上角 x
        y0 = random.randint(0, h - self.crop_size)  # 随机裁剪左上角 y

        hflip_rnd = random.uniform(0, 1)  # 水平翻转阈值
        vflip_rnd = random.uniform(0, 1)  # 垂直翻转阈值
        degree = random.choice([0, 90, 180, 270])  # 旋转角度（当前未实际使用）

        return {'x0': x0,
                'y0': y0,
                'hflip_rnd': hflip_rnd,
                'vflip_rnd': vflip_rnd,
                'degree': degree
                }

    def apply_transform(self, ip_frame, gt_frame, params):
        x0, y0, hflip_rnd, vflip_rnd, deg = params['x0'], params['y0'], params['hflip_rnd'], params['vflip_rnd'], params['degree']

        # random cropping  随机裁剪：含雨帧与 GT 帧用同一裁剪框(保证对齐)
        if self.apply_crop:
            x1 = x0 + self.crop_size
            y1 = y0 + self.crop_size

            ip_frame = ip_frame.crop((x0, y0, x1, y1))
            gt_frame = gt_frame.crop((x0, y0, x1, y1))

        # horizontal flip  水平/垂直翻转（输入与 GT 同步）
        if self.apply_hflip:
            if hflip_rnd < 0.5:
                ip_frame = ip_frame.transpose(Image.FLIP_LEFT_RIGHT)
                gt_frame = gt_frame.transpose(Image.FLIP_LEFT_RIGHT)

            if vflip_rnd < 0.5:
                ip_frame = ip_frame.transpose(Image.FLIP_TOP_BOTTOM)
                gt_frame = gt_frame.transpose(Image.FLIP_TOP_BOTTOM)

        # color augmentattion  颜色增强（当前被作者注释留空，直接 pass）
        if self.apply_coloraug:
            # gamma correction
            # ip_frame = TF.adjust_gamma(ip_frame, 1)
            # gt_frame = TF.adjust_gamma(gt_frame, 1)

            # saturation
            # sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            # ip_frame = TF.adjust_saturation(ip_frame, sat_factor)
            # gt_frame = TF.adjust_saturation(gt_frame, sat_factor)
            pass

        # other transforms, e.g. Normalize, ToTensor  # 其余变换(ToTensor + Normalize)
        if self.transform:
            ip_frame = self.transform(ip_frame)
            gt_frame = self.transform(gt_frame)

        return ip_frame, gt_frame


def get_dataloader(dataset, batch_size=8, shuffle=True, num_workers=8):
    """构造 DataLoader（视频去雨通常用多进程 num_workers 加速图像读取）。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)


if __name__ == '__main__':
    data_root = '/media/hdd/derain/NTU-derain'
    indexfile = '/home/dxli/workspace/derain/proj/data/Dataset_Training_Synthetic.json'
    # indexfile = '/home/dxli/workspace/derain/proj/data/Dataset_Testing_Synthetic.json'

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                        }

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(__imagenet_stats['mean'],
                                                         __imagenet_stats['std'])
                                    ]
                                   )

    dataset = NTU_dataset(data_root=data_root,
                          indexfile_path=indexfile,
                          transform=transform,
                          apply_rotation=False,
                          apply_coloraug=False
                          )

    ip, gt = dataset[-2]
    print(ip.shape, gt.shape)
