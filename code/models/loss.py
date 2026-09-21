# =============================================================================
# loss.py —— 损失函数定义
#
# 注意：当前训练主循环(train.py)实际使用的是 F.mse_loss(MSE)，
# 本文件的 Sobel / cal_spatial_loss / cal_temporal_loss 是作者调试时空/深度
# 辅助损失用的备用实现（类继承自深度估计相关工作，尚未在默认流程中启用）。
# =============================================================================
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def adjust_gt(gt_depth, pred_depth):
    """把不同尺度的 GT 图逐个插值到对应预测图的尺寸（多尺度监督用）。"""
    adjusted_gt = []
    for each_depth in pred_depth:
        adjusted_gt.append(F.interpolate(gt_depth, size=[each_depth.size(2), each_depth.size(3)],
                                         mode='bilinear', align_corners=True))
    return adjusted_gt


class Sobel(nn.Module):
    """Sobel 边缘算子卷积层：用固定(不可训练)的 Sobel 卷积核计算 x/y 方向梯度图像。"""

    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        # Sobel 算子核（x、y 方向）
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)  # (2,1,3,3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():  # 冻结参数，Sobel 核不参与学习
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)  # 输出 2 通道(即 x 方向梯度、y 方向梯度)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


def cal_spatial_loss(output, depth_gt):
    """空间损失（备用）：鼓励输出与 GT 的像素值、梯度(x/y)及法向量一致。

    综合了深度损失(loss_depth)、梯度损失(loss_dx/loss_dy)与法向量一致性损失(loss_normal)。
    注意该版本假定输入为单通道图(ones 为 1 通道)，用于单通道深度监督；默认流程未使用。
    """
    losses = []

    for depth_index in range(len(output)):
        cos = nn.CosineSimilarity(dim=1, eps=0)  # 余弦相似度，用于比较法向量
        get_gradient = Sobel().cuda()            # Sobel 求梯度
        ones = torch.ones(depth_gt.size(0), 1, depth_gt.size(2), depth_gt.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)     # 法向量的第 3 个分量(恒为 1)
        depth_grad = get_gradient(depth_gt)      # GT 的 x/y 梯度
        output_grad = get_gradient(output)       # 输出的 x/y 梯度
        # 分别取出 x、y 方向梯度
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth_gt)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth_gt)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth_gt)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth_gt)

        # 由梯度构造表面法向量 (-dx, -dy, 1)
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        cof = 0.5  # 平滑常数，防止 log(0)

        loss_depth = torch.log(torch.abs(output - depth_gt) + cof).mean()                       # 像素值损失
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + cof).mean()             # x 梯度损失
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + cof).mean()             # y 梯度损失
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()                    # 法向量一致性损失

        loss = loss_depth + loss_normal + (loss_dx + loss_dy)

        losses.append(loss)

    spatial_loss = sum(losses)

    return spatial_loss


def cal_temporal_loss(pred_cls, gt_cls):
    """时间/分类损失（备用）：二分类交叉熵，用于时间一致性相关的学习任务。"""
    return F.binary_cross_entropy_with_logits(pred_cls, gt_cls)
