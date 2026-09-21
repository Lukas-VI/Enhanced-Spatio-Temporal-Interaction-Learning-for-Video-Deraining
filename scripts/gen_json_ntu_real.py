# ==============================================================================
# gen_json_ntu_real.py —— 为 NTU 真实雨数据生成索引文件(.json)
# ------------------------------------------------------------------------------
# 作用：扫描真实雨数据目录，生成与 gen_json_ntu.py 相同结构的 .json。
# 注意差异：真实数据没有独立 GT 目录，gt 键与 rain 键指向同一张雨图
# （真实雨图没有配对真值，仅用于测试观测）。
# ==============================================================================
import os
import json

num_instances = 8
out_dir = '/home/dxli/workspace/derain/proj/data'

root = '/media/hdd/derain/NTU-derain/Dataset_Testing_RealRain'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_RealRain'

out_filepath = os.path.join(out_dir, os.path.split(root)[-1] + '.json')


all_dirs = sorted([d for d in [os.path.join(root, x) for x in os.listdir(root)] if os.path.isdir(d)])


entry_list = list()

for d in all_dirs:
    entry = []

    prefix, filename = os.path.split(d)
    prefix = os.path.split(prefix)[-1]

    gt_dirname = filename
    # gt_dirname = filename[:-4] + 'GT'
    gt_dirpath = os.path.join(prefix, gt_dirname)

    # 按字典序排列该序列的所有 jpg 帧
    frame_names = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))

    for fn in frame_names:
        rain_filepath = os.path.join(prefix, filename, fn)
        gt_filepath = os.path.join(gt_dirpath, fn)

        # 真实数据：gt 与 rain 相同（无真值）
        entry.append({'rain': rain_filepath, 'gt': rain_filepath})

    entry_list.append(entry)

json.dump(entry_list, open(out_filepath, 'w'))



