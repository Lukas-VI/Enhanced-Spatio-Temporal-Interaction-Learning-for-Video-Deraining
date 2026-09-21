# ==============================================================================
# gen_json_deblur.py —— 为视频去模糊数据生成索引文件(.json)
# ------------------------------------------------------------------------------
# 作用：扫描视频去模糊测试数据集，生成逐帧索引。
# 此脚本是针对「去模糊(deblur)」任务的变体，改进：gt 在 <序列>/GT 目录，
# 模糊输入(blur)在 <序列>/input 目录，键名为 blur / gt。
# ==============================================================================
import os
import json

# num_instances = 8
# out_dir = '/home/dxli/workspace/derain/proj/data'
out_dir = '/home/dxli/workspace/videodeblur/data'

# root = '/media/hdd/derain/NTU-derain/Dataset_Training_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_RealRain'
root = '/media/hdd/videodeblur/videodeblurring/quantitative_datasets/test'

out_filepath = os.path.join(out_dir, os.path.split(root)[-1] + '.json')

all_dirs = sorted([d for d in [os.path.join(root, x) for x in os.listdir(root)]])

entry_list = list()

for d in all_dirs:
    entry = []

    prefix, filename = os.path.split(d)

    # 每个序列含 GT(清晰) 与 input(模糊) 两个子目录
    gt_dirpath = os.path.join(d, 'GT')
    blur_dirpath = os.path.join(d, 'input')

    # 以 GT 目录的文件名为基准（模糊与清晰帧同名）
    frame_names = sorted([f for f in os.listdir(gt_dirpath) if f.endswith('jpg')])

    for fn in frame_names:
        blur_filepath = os.path.join(blur_dirpath, fn)
        gt_filepath = os.path.join(gt_dirpath, fn)

        entry.append({'blur': blur_filepath, 'gt': gt_filepath})

    entry_list.append(entry)

json.dump(entry_list, open(out_filepath, 'w'))



