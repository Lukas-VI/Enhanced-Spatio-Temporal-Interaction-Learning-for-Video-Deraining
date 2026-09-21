# ==============================================================================
# gen_json_ntu.py —— 为 NTU 训练数据生成索引文件(.json)
# ------------------------------------------------------------------------------
# 作用：扫描 NTU-Derain 训练目录下每个视频序列，把雨图(rain)与清晰图(GT)
# 的逐帧绝对路径配对，写入一个 .json 文件，供 dataset/ntu_dataset.py 加载。
#
# json 结构：外层 list，每项对应一个视频序列；序列内为 list，每项是
#   {'rain': 雨图(输入)绝对路径, 'gt': 清晰图(标签)绝对路径}
# 输入目录命名约定：形如 <path>/xxx_rain 的序列目录，GT 目录为 <path>/xxx_GT。
# 说明（理解存疑）：保存的 json 中路径相对路径(只含序列名)，需与 data_root 配合。
# ==============================================================================
import os
import json

num_instances = 8
out_dir = '/home/dxli/workspace/derain/proj/data'

root = '/media/hdd/derain/NTU-derain/Dataset_Training_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_RealRain'

out_filepath = os.path.join(out_dir, os.path.split(root)[-1] + '.json')


# 只保留序列子目录，排除 GT 结尾的目录
all_dirs = sorted([d for d in [os.path.join(root, x) for x in os.listdir(root)] if os.path.isdir(d) and not d.endswith('GT')])

entry_list = list()

for d in all_dirs:
    entry = []

    prefix, filename = os.path.split(d)
    prefix = os.path.split(prefix)[-1]

    # 由序列名推导 GT 目录名：去掉末尾 '_rain' 等 7 字符再加 'GT'
    gt_dirname = filename[:-7] + 'GT'
    # gt_dirname = filename[:-4] + 'GT'
    gt_dirpath = os.path.join(prefix, gt_dirname)

    # 按字典序排列该序列的所有 jpg 帧
    frame_names = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))

    for fn in frame_names:
        rain_filepath = os.path.join(prefix, filename, fn)
        gt_filepath = os.path.join(gt_dirpath, fn)

        # 每帧一个【rain 输入, gt 标签】键值对
        entry.append({'rain': rain_filepath, 'gt': gt_filepath})

    entry_list.append(entry)

# 写入 json 文件
json.dump(entry_list, open(out_filepath, 'w'))



