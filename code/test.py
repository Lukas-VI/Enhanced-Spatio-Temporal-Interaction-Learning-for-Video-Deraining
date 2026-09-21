# ==============================================================================
# test.py —— 测试 / 推理脚本
# ------------------------------------------------------------------------------
# 作用：
#   加载训练好的粗网络(CoarseNet, net_C)和精网络(FineNet, net_F)，对测试集视频
#   进行前向推理，计算 PSNR 指标，并将去雨前后的结果(gt / C_out / F_out)保存为图片。
#
# 主干流程：
#   1. main()                  读取「test」配置
#   2. test()                  构建并加载粗/精网络权重
#   3. validate()              逐 batch 推理，统计 PSNR，保存图片
#
# 模型约定（与训练一致）：
#   - 粗网络输入 5 维张量 (b, c, d, h, w)，输出整段 d 帧粗去雨结果；
#   - 精网络输入 (粗输出, 原始输入)，输出精去雨结果；
#   - 本项目精网络使用 F_npic 模式，输出的是全部 d 帧（见代码分支）。
# ==============================================================================

import sys
import time

import torch.nn as nn
import torch.nn.parallel
from torchvision import transforms

import utils
from dataset import ntu_dataset
from models import modules
from models import net
from models.backbone_dict import backbone_dict
from options import get_args
from utils import *

import logging

# cudnn.benchmark = True
logging_root = './test_log'
if not os.path.exists(logging_root):
    os.mkdir(logging_root)

logging.basicConfig(filename=os.path.join(logging_root, '{}.log'.format(int(time.time()))), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# cudnn optimization
torch.backends.cudnn.benchmark = True


# =============================================================
# validate(args, net_C, net_F, out_dir)
# 对测试集做前向推理并统计 PSNR、保存去雨图像
# =============================================================
def validate(args, net_C, net_F, out_dir):
    # 分别建立 gt / 粗网络输出 / 精网络输出 三个结果子目录
    gt_dir = os.path.join(out_dir, 'gt')
    cout_dir = os.path.join(out_dir, 'C_out')
    fout_dir = os.path.join(out_dir, 'F_out')

    makedir(gt_dir)
    makedir(cout_dir)
    makedir(fout_dir)

    # manually release GPU memory.
    torch.cuda.empty_cache()

    # 读取测试数据加载器（DataLoader）
    test_loader = get_test_loader(args)

    # 测试阶段固定网络参数（不计算梯度、不影响 BN/dropout 统计）
    net_C.eval()
    net_F.eval()

    # 用 AverageMeter 统计各类 PSNR 与耗时（val 为当前 batch 值，avg 为累计均值）
    all_out_psnr_C = AverageMeter()   # 粗网络输出的 PSNR（相对 gt）
    all_inp_psnr_C = AverageMeter()   # 粗网络输入的 PSNR（即原始雨图相对 gt，作为参照）
    all_out_psnr_F = AverageMeter()   # 精网络输出的 PSNR
    all_inp_psnr_F = AverageMeter()   # 精网络输入的 PSNR（即粗网络输出相对 gt，作为参照）
    time_C = AverageMeter()           # 粗网络单次前向耗时
    time_F = AverageMeter()           # 精网络单次前向耗时
    time_batch = AverageMeter()       # 单个 batch 总耗时

    end = time.time()

    for batch_idx, sample in enumerate(test_loader):
        inp_videos, gt_videos = sample[0].cuda(), sample[1].cuda()  # (b,c,d,w,h)
        batch_size, _, nf, _, _ = inp_videos.shape     # nf = 时间维帧数（窗口大小）

        # model forward
        with torch.no_grad():
            model_out_C = net_C(inp_videos)  # (b, c, d, h, w)  粗网络输出整段 d 帧
            c_now = time.time()
            time_C.update(c_now - end)

            model_out_F = net_F(model_out_C, inp_videos)  # (b, c, 1, h, w)  精网络输出（F_npic 下实际为全部帧）
            time_F.update(time.time() - c_now)

            # batch time
            time_batch.update(time.time() - end)

        # validation range for output of coarse network
        # 决定对粗网络输出的哪几帧计算 PSNR：
        #   args.val_mode == 'all'   -> 对所有 nf 帧都算
        #   args.val_mode == 'mid'   -> 只算中间一帧 (nf//2)
        if args.val_mode == 'all':
            val_range = range(nf)
        elif args.val_mode == 'mid':
            val_range = range(nf // 2, nf // 2 + 1)
        else:
            raise ValueError("invalid validation mode, must be 'all' or 'mid', got {}".format(args.val_mode))

        # 把网络输出恢复（反归一化）到正常的像素范围，方便计算 PSNR 和保存图片
        model_out_C = clamp_on_imagenet_stats(model_out_C)
        # model_out_F = clamp_on_imagenet_stats(model_out_F.unsqueeze(2)).squeeze(2)
        model_out_F = clamp_on_imagenet_stats(model_out_F)

        for i in range(batch_size):
            # validate for output of coarse network
            for j in val_range:
                # 取出第 i 个样本第 j 帧的：粗输出 / gt / 原始输入（形状均为 (c,h,w)）
                out_C, gt_C, inp_C = model_out_C[i, :, j, :, :], gt_videos[i, :, j, :, :], inp_videos[i, :, j, :, :]

                # 计算粗网络输出与输入的 PSNR（都以 gt 为基准）
                out_psnr_C, inp_psnr_C = calculate_psnr(gt_C, out_C), calculate_psnr(gt_C, inp_C)

                all_out_psnr_C.update(out_psnr_C)
                all_inp_psnr_C.update(inp_psnr_C)

                if args.F_npic:
                    # validate for output of fine network
                    # F_npic 模式下精网络逐帧输出，可逐帧评估与保存
                    out_F, gt_F, inp_F = model_out_F[i, :, j, :, :], gt_videos[i, :, j, :, :], model_out_C[i, :, j, :,
                                                                                                          :]
                    out_psnr_F, inp_psnr_F = calculate_psnr(gt_F, out_F), calculate_psnr(gt_F, inp_F)

                    all_out_psnr_F.update(out_psnr_F)
                    all_inp_psnr_F.update(inp_psnr_F)

                    # 保存第 batch_idx 个 batch、第 i 个样本、第 j 帧的图像
                    save_image(gt_C, os.path.join(gt_dir, '{}_{}_{}.png'.format(batch_idx, i, j)))
                    save_image(out_C, os.path.join(cout_dir, '{}_{}_{}.png'.format(batch_idx, i, j)))
                    save_image(out_F, os.path.join(fout_dir, '{}_{}_{}.png'.format(batch_idx, i, j)))

            if not args.F_npic:
                # 注意：本项目只支持 F_npic 模式，非 npic 分支被主动 raise 掉，下面内容不会执行
                raise ValueError('F_npic = False.')
                # validate for output of fine network
                # （以下为非 npic 模式遗留代码：只有中间帧被评估并保存）
                out_F, gt_F, inp_F = model_out_F[i, :, :, :], gt_videos[i, :, nf // 2, :, :], model_out_C[i, :, nf // 2,
                                                                                            :, :]
                out_psnr_F, inp_psnr_F = calculate_psnr(gt_F, out_F), calculate_psnr(gt_F, inp_F)

                all_out_psnr_F.update(out_psnr_F)
                all_inp_psnr_F.update(inp_psnr_F)

        # 每个 batch 输出一次当前与累计的 PSNR 和耗时日志
        if batch_idx % 1 == 0:
            logging.info('[{batch}/{total_batch}]\tbth. out_C PSNR: {boc.val:.3f},\t inp_C PSNR: {binpc.val:.3f}\t'
                         'out_F PSNR: {bof.val:.4f},\tC time: {tc.val:.3f}\tF time: {tf.val:.3f}\t batch time: {tb.val:.3f}\n'
                         '[{batch}/{total_batch}]\tavg. out_C PSNR: {boc.avg:.3f},\t inp_C PSNR: {binpc.avg:.3f}\t'
                         'out_F PSNR: {bof.avg:.3f},\tC time: {tc.avg:.3f}\tF time: {tf.avg:.3f}\t batch time: {tb.avg:.3f}'.format(batch=batch_idx,
                                                                                                                      boc=all_out_psnr_C,
                                                                                                                      binpc=all_inp_psnr_C,
                                                                                                                      bof=all_out_psnr_F,
                                                                                                                      binpf=all_inp_psnr_F,
                                                                                                                      tc=time_C,
                                                                                                                      tf=time_F,
                                                                                                                      tb=time_batch,
                                                                                                                                    total_batch=len(test_loader)
                                                                                                                      ))

        # 释放显存，避免测试过程中显存累积
        torch.cuda.empty_cache()

        end = time.time()

    # 返回粗、精网络的平均输出 PSNR
    return all_out_psnr_C.avg, all_out_psnr_F.avg


def test(args):
    # 依据配置构建并初始化粗、精网络
    net_C, net_F = build_model(args)

    # 用 DataParallel 包装以便在可用 GPU 上并行
    net_C = nn.DataParallel(net_C).cuda()
    net_F = nn.DataParallel(net_F).cuda()

    # 从指定 checkpoint 路径加载粗、精网络权重
    load_checkpoint(args, net_C, pretrain_ckpt=args.loadckpt_C)
    load_checkpoint(args, net_F, pretrain_ckpt=args.loadckpt_F)

    # out_dir = '/home/dxli/workspace/derain/proj/out/ntu'
    out_dir = args.out_dir    # 推理结果输出目录

    validate(args, net_C=net_C, net_F=net_F, out_dir=out_dir)


def main():
    # 「test」模式解析命令行配置
    args = get_args('test')
    test(args)


if __name__ == '__main__':
    main()
