# ==============================================================================
# options/testopt.py —— 测试 / 推理相关命令行参数定义
# ------------------------------------------------------------------------------
# 用 argparse 定义测试阶段参数：网络结构、验证模式、输入/输出路径与权重路径。
# 与 trainopt.py 相比少了训练专属参数，多了 out_dir / loadckpt_C / loadckpt_F。
# ==============================================================================
import argparse


def _get_test_opt():
    parser = argparse.ArgumentParser(description='Evaluate performance on test set')

    # ===== 网络结构（需与训练一致才能正确加载权重）=====
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
    parser.add_argument('--use_bilstm', action='store_true')
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--input_residue', action='store_true')
    parser.add_argument('--compress_channels', type=int, default=8, help='number of channels in R_CLSTM.')
    parser.add_argument('--F_npic', action='store_true', default=False, help='if True, output n images from fine net; '
                                                                             'otherwise output the one in the middle.')
    #   --F_npic: 决定精网络输出全部帧或仅中间帧
    parser.add_argument('--torch_home', type=str, default=None)

    # ===== 验证模式 =====
    parser.add_argument('--val_mode', type=str, default='all', help='validation on the frame in the middle (mid) or '
                                                                    'all the frames (all) of coarse network.')
    # ===== 数据与路径 =====
    parser.add_argument('--eval_file', type=str, required=True, help='the path of indexfile',
                        default='/home/dxli/workspace/derain/proj/data/Dataset_Testing_Synthetic.json')
    parser.add_argument('--checkpoint_dir_C', required=False, help="the directory to save the checkpoints of coarse net",
                        default='./checkpoint/C')
    parser.add_argument('--checkpoint_dir_F', required=False, help="the directory to save the checkpoints of finenet",
                        default='./checkpoint/F')
    parser.add_argument('--data_root', type=str, required=True, help="the root path of image data.",
                        default='/media/hdd/derain/NTU-derain')
    parser.add_argument('--out_dir', type=str, required=True, help="output dir root.")

    # ===== 杂项 =====
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--loadckpt', action='store_true')
    parser.add_argument('--loadckpt_C', type=str, help='pretrain weights of coarse net.')
    parser.add_argument('--loadckpt_F', type=str, help='pretrain weights of fine net.')
    parser.add_argument('--use_cuda', type=bool, default=True)

    return parser.parse_args()

