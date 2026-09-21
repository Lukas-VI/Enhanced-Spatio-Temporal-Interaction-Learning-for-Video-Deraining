# =============================================================================
# train.py —— 视频去雨训练主程序
#
# 训练采用"粗到精"两阶段：每个 batch 先优化粗网络 net_C(含时空交互 R-CLSTM，
# 输出整段 d 帧去雨结果)，再优化精网络 net_F(在粗结果 + 原输入上进一步精修)。
# 训练过程中按设定间隔运行 validate() 在验证集上计算 PSNR，并用 PSNR 保存最优断点。
#
# 注：实际使用的损失为 MSE(F.mse_loss)；loss.py 中的空间/时间损失为备用。
# =============================================================================
import shutil
import sys
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

from options import get_args
from utils import *

logging_root = './train_log'
if not os.path.exists(logging_root):
    os.mkdir(logging_root)

# 创建日志：同时输出到文件与标准输出
logging.basicConfig(filename=os.path.join(logging_root, '{}.log'.format(int(time.time()))), filemode='w',
                    level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# cudnn optimization  # 启用 cudnn 自动选择最快的卷积算法
torch.backends.cudnn.benchmark = True


def validate(args, net_C, net_F):
    """在验证集上评估粗网络(C)与精网络(F)的 PSNR，用于训练中的在线评估。

    只评估前 eval_num_batch 个 batch 以节省训练时间；返回 (PSNR_C_avg, PSNR_F_avg)。
    """
    # manually release GPU memory.  # 手动释放显存
    torch.cuda.empty_cache()

    test_loader = get_test_loader(args)

    net_C.eval()  # 切换到评估模式(关闭 dropout 等)
    net_F.eval()

    end = time.time()

    # 各项统计容器
    all_out_psnr_C = AverageMeter()   # 粗网络输出相对 GT 的 PSNR
    all_inp_psnr_C = AverageMeter()   # 输入(含雨)相对 GT 的 PSNR(基线)
    all_out_psnr_F = AverageMeter()   # 精网络输出相对 GT 的 PSNR
    all_inp_psnr_F = AverageMeter()   # 输入(粗输出)相对 GT 的 PSNR
    time_C = AverageMeter()
    time_F = AverageMeter()
    time_batch = AverageMeter()

    for batch_idx, sample in enumerate(test_loader):
        if batch_idx > args.eval_num_batch:
            # only test first a couple of batches to save validation time during training.
            break  # 只测前几个 batch 以节省训练时的验证时间

        inp_videos, gt_videos = sample[0].cuda(), sample[1].cuda()  # (b,c,d,w,h)
        batch_size, _, nf, _, _ = inp_videos.shape

        # model forward  前向推理（关闭梯度）
        with torch.no_grad():
            model_out_C = net_C(inp_videos)  # (b, c, d, h, w)  粗网络输出 d 帧
            c_now = time.time()
            time_C.update(c_now - end)

            model_out_F = net_F(model_out_C, inp_videos)  # (b, c, 1, h, w)  精网络输出
            time_F.update(time.time() - c_now)

            # batch time  # 记录整批耗时
            time_batch.update(time.time() - end)

        # validation range for output of coarse network  # 决定对粗网络哪些帧算 PSNR
        if args.val_mode == 'all':
            val_range = range(nf)          # 所有帧
        elif args.val_mode == 'mid':
            val_range = range(nf // 2, nf // 2 + 1)  # 仅中间帧
        else:
            raise ValueError("invalid validation mode, must be 'all' or 'mid', got {}".format(args.val_mode))

        # 反归一化截断，保证像素在合法 RGB 范围
        model_out_C = clamp_on_imagenet_stats(model_out_C)
        # model_out_F = clamp_on_imagenet_stats(model_out_F.unsqueeze(2)).squeeze(2)
        model_out_F = clamp_on_imagenet_stats(model_out_F)

        for i in range(batch_size):
            # validate for output of coarse network  # 计算粗网络各帧 PSNR
            for j in val_range:
                out_C, gt_C, inp_C = model_out_C[i, :, j, :, :], gt_videos[i, :, j, :, :], inp_videos[i, :, j, :, :]

                out_psnr_C, inp_psnr_C = calculate_psnr(gt_C, out_C), calculate_psnr(gt_C, inp_C)

                all_out_psnr_C.update(out_psnr_C)
                all_inp_psnr_C.update(inp_psnr_C)

                if args.F_npic:
                    # validate for output of fine network  # npic 模式下精网络输出多帧，逐帧计算
                    out_F, gt_F, inp_F = model_out_F[i, :, j, :, :], gt_videos[i, :, j, :, :], model_out_C[i, :, j, :, :]
                    out_psnr_F, inp_psnr_F = calculate_psnr(gt_F, out_F), calculate_psnr(gt_F, inp_F)

                    all_out_psnr_F.update(out_psnr_F)
                    all_inp_psnr_F.update(inp_psnr_F)

            if not args.F_npic:
                # validate for output of fine network  # 非 npic 模式精网络只输出中间帧
                out_F, gt_F, inp_F = model_out_F[i, :, :, :], gt_videos[i, :, nf // 2, :, :], model_out_C[i, :, nf // 2,
                                                                                              :, :]
                out_psnr_F, inp_psnr_F = calculate_psnr(gt_F, out_F), calculate_psnr(gt_F, inp_F)

                all_out_psnr_F.update(out_psnr_F)
                all_inp_psnr_F.update(inp_psnr_F)

        if batch_idx % 2 == 0:
            # 每 2 个 batch 打印一次当前与累计 PSNR
            logging.info('[{batch}/{total_bt}]\tbth. out_C PSNR: {boc.val:.3f},\t inp_C PSNR: {binpc.val:.3f}\t'
                         'out_F PSNR: {bof.val:.4f},\t inp_F PSNR: {binpf.val:.3f}\t batch time: {tb.val:.3f}\n'
                         '[{batch}/{total_bt}]\tavg. out_C PSNR: {boc.avg:.3f},\t inp_C PSNR: {binpc.avg:.3f}\t'
                         'out_F PSNR: {bof.avg:.3f},\t inp_F PSNR: {binpf.avg:.3f}\t batch time: {tb.val:.3f}'.format(batch=batch_idx,
                                                                                                                      boc=all_out_psnr_C,
                                                                                                                      binpc=all_inp_psnr_C,
                                                                                                                      bof=all_out_psnr_F,
                                                                                                                      binpf=all_inp_psnr_F,
                                                                                                                      tc=time_C,
                                                                                                                      tf=time_F,
                                                                                                                      tb=time_batch,
                                                                                                                      total_bt=len(test_loader)
                                                                                                                      ))
        end = time.time()

        torch.cuda.empty_cache()

    return all_out_psnr_C.avg, all_out_psnr_F.avg


def train(args):
    """主训练循环：按 epoch 迭代，逐 batch 先更新粗网络再更新精网络，
    每 epoch 保存断点，并按间隔评估与保存最优 PSNR 断点。
    """
    train_loader = get_train_loader(args)
    net_C, net_F = build_model(args)

    net_C = nn.DataParallel(net_C)   # 多卡并行
    net_F = nn.DataParallel(net_F)

    logging.info(args)

    # 从断点恢复(若 resume)，取 C/F 中较大的起始 epoch
    start_epoch = max(load_checkpoint(args, net_C, args.checkpoint_dir_C),
                      load_checkpoint(args, net_F, args.checkpoint_dir_F)
                      )

    net_C.cuda()
    net_F.cuda()

    # 为粗、精网络分别建优化器（学习率各自独立）
    optimizer_C = build_optimizer(model=net_C,
                                  learning_rate=args.lr_C,
                                  optimizer_name=args.optimizer_name,
                                  weight_decay=args.weight_decay,
                                  epsilon=args.epsilon,
                                  momentum=args.momentum
                                  )

    optimizer_F = build_optimizer(model=net_F,
                                  learning_rate=args.lr_F,
                                  optimizer_name=args.optimizer_name,
                                  weight_decay=args.weight_decay,
                                  epsilon=args.epsilon,
                                  momentum=args.momentum
                                  )

    best_psnr_C, best_psnr_F = 0., 0.
    for epoch in range(start_epoch, args.epochs):
        # train
        adjust_learning_rate(optimizer_C, epoch, args.lr_C)
        adjust_learning_rate(optimizer_F, epoch, args.lr_F)
        logging.info('Epoch {} learning rate {} for C, {} for F'.format(epoch, optimizer_C.param_groups[0]['lr'], optimizer_F.param_groups[0]['lr']))

        batch_time = AverageMeter()
        batch_time_C = AverageMeter()
        batch_time_F = AverageMeter()
        losses_C = AverageMeter()
        losses_F = AverageMeter()

        # mutate model train / eval states  # 依据 freeze 标志切换 C/F 的 train/eval 状态
        if args.freeze_net_C:
            logging.info('Freezing coarse network.')
            net_C.eval()   # 冻结粗网络(不更新)
        else:
            logging.info('Training coarse network.')
            net_C.train()

        if args.freeze_net_F:
            logging.info('Freezing fine network.')
            net_F.eval()
        else:
            logging.info('Training fine network.')
            net_F.train()

        end = time.time()
        for batch_idx, sample in enumerate(train_loader):

            in_videos_C, gt_videos_C = sample[0].cuda(), sample[1].cuda()  # (b,c,d,w,h)
            batch_size, _, nf, _, _ = in_videos_C.shape

            # ===== optimize coarse network =====  优化粗网络
            if args.freeze_net_C:
                # 冻结时不计算梯度，仅前向获得输出与损失（用于给精网络提供输入）
                with torch.no_grad():
                    model_out_C = net_C(in_videos_C)
                    loss_C = F.mse_loss(model_out_C, gt_videos_C)
            else:
                optimizer_C.zero_grad()

                model_out_C = net_C(in_videos_C)  # (b, c, d, h, w)
                # compute loss of coarse net and update
                loss_C = F.mse_loss(model_out_C, gt_videos_C)  # 粗网络对整段 d 帧做 MSE 监督
                loss_C.backward()
                optimizer_C.step()

            losses_C.update(loss_C.item(), batch_size)
            # timing for coarse net
            c_end = time.time()
            batch_time_C.update(c_end - end)

            # ===== optimize fine network =====  优化精网络
            # take the frame in the middle as ground truth.  # 对精网络取中间帧作为监督
            if args.F_npic:
                gt_frames_F = gt_videos_C            # npic 模式：用全部 d 帧监督
            else:
                gt_frames_F = gt_videos_C[:, :, gt_videos_C.shape[2] // 2, :, :]  # 否则只用中间帧

            if args.freeze_net_F:
                with torch.no_grad():
                    model_out_F = net_F(model_out_C.detach(), in_videos_C)
                    loss_F = F.mse_loss(model_out_F, gt_frames_F)
            else:
                optimizer_F.zero_grad()

                # compute loss of finenet and update
                # 用 .detach() 切断精网络到粗网络的反向传播（两阶段各自独立训练）
                model_out_F = net_F(model_out_C.detach(), in_videos_C)
                loss_F = F.mse_loss(model_out_F, gt_frames_F)
                loss_F.backward()
                optimizer_F.step()

            losses_F.update(loss_F.item(), batch_size)
            # timing for fine net
            batch_time_F.update(time.time() - c_end)

            # logistics  计时/统计
            batch_time.update(time.time() - end)
            end = time.time()

            global_step = len(train_loader) * epoch + batch_idx
            if batch_idx % args.log_interval == 0:
                # 定期打印损失与时耗
                logging.info(('Epoch: [{0}][{1}/{2}]\t'
                              'iters: {3}\t'
                              'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                              'C_net time: {batch_time_C.val:.3f} ({batch_time_C.sum:.3f})\t'
                              'F_net time: {batch_time_F.val:.3f} ({batch_time_F.sum:.3f})\t'
                              'Loss_C {loss_C.val:.4f} ({loss_C.avg:.4f})\t'
                              'Loss_F {loss_F.val:.4f} ({loss_F.avg:.4f})'
                              .format(epoch, batch_idx, len(train_loader), global_step,
                                      batch_time=batch_time, batch_time_C=batch_time_C, batch_time_F=batch_time_F,
                                      loss_C=losses_C, loss_F=losses_F)
                              ))

        if (epoch + 1) % 1 == 0:
            # 每个 epoch 都保存一次常规断点
            save_checkpoint(net_C.state_dict(),
                            filename=args.checkpoint_dir_C + "checkpoints_small_" + str(epoch + 1) + ".pth.tar")
            save_checkpoint(net_F.state_dict(),
                            filename=args.checkpoint_dir_F + "checkpoints_small_" + str(epoch + 1) + ".pth.tar")

        # validate  # 按间隔评估
        if args.eval_on_the_fly and (epoch + 0) % args.eval_interval == 0:
            logging.info('Starting Evaluation on Epoch {}'.format(epoch))
            avg_psnr_C, avg_psnr_F = validate(args, net_C, net_F)

            # save checkpoint with best PSNR  # 把该 epoch 断点复制为 best(若 PSNR 更高)
            if avg_psnr_C > best_psnr_C:
                best_psnr_C = avg_psnr_C

                shutil.copyfile(src=args.checkpoint_dir_C + "checkpoints_small_" + str(epoch + 1) + ".pth.tar",
                                dst=args.checkpoint_dir_C + "checkpoints_best_" + str(epoch + 1) + ".pth.tar"
                                )
                logging.info('Best C PSNR {} found at {} epoch'.format(best_psnr_C, epoch))

            if avg_psnr_F > best_psnr_F:
                best_psnr_F = avg_psnr_F
                shutil.copyfile(src=args.checkpoint_dir_F + "checkpoints_small_" + str(epoch + 1) + ".pth.tar",
                                dst=args.checkpoint_dir_F + "checkpoints_best_" + str(epoch + 1) + ".pth.tar"
                                )
                logging.info('Best F PSNR {} found at {} epoch'.format(best_psnr_F, epoch))


def main():
    args = get_args('train')  # 解析命令行参数

    # Create folder
    makedir(args.checkpoint_dir_C)  # 创建粗网络断点目录
    makedir(args.checkpoint_dir_F)  # 创建精网络断点目录
    makedir(args.logdir)            # 创建日志目录

    train(args)


if __name__ == '__main__':
    main()
