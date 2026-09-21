# Enhanced Spatio-Temporal Interaction Learning for Video Deraining 中文导读（新人学习用）

> 以下中文介绍是本仓库注释补充阶段为方便新人快速上手而**前置**插入的导读，
> 原始 README 内容完整保留在下方。所有描述均依据本仓库的代码与注释真实整理，不做虚构。

## 一句话定位

这是 IEEE TPAMI 2022 的一篇**视频去雨（Video Deraining）**工作
（*Enhanced Spatio-Temporal Interaction Learning for Video Deraining: A Faster and Better Framework*，
作者 Kaihao Zhang, Dongxu Li, Wenhan Luo, Wenqi Ren, Wei Liu）。
它解决的问题是：从一段**含雨的视频帧序列**中恢复出干净的帧，重点在同时利用**空间（单帧画面）**与
**时间（相邻帧的关联）**信息去除雨线；相比单张图像去雨，视频去雨能借助时间冗余获得更稳定、更一致的结果。
属于**有监督的视频帧级图像复原**任务。

## 方法与核心思想（时空建模是重点）

采用**“粗到精”（coarse-to-fine）两阶段框架**，全程张量约定为 5 维 `(b, c, d, h, w)`，
其中 `d = 时间维(帧数)`，一段视频即一个特征立方体（见 `net.py` 顶部注释）。

**阶段一：粗网络 CoarseNet（含时空交互）**
1. `cubes_2_maps`：把视频立方体展平成 `(b*d, c, h, w)`，复用 2D 卷积一次处理所有帧；
2. **编码器 E**（`ResNet/DenseNet/SENet` 预训练 backbone 的卷积主干）提取 5 层不同尺度特征 `block0..4`（空间金字塔）；
3. **解码器 D**（LapSRN 风格的逐级上采样 + 跨层残差，`_UpProjection`）恢复高分辨率特征；
4. **多层特征融合 MFF**：把 block1..4 各自上采样到统一尺寸、压缩到 16 通道再拼接融合成 64 通道；
5. **时空交互精化 R-CLSTM**（`R_CLSTM_modules.py`，论文核心）：沿时间维逐帧循环，
   用类 LSTM 门控（遗忘门 `F_t`、输入门 `I_t`、候选 `C_t`、调制门 `Q_t`）更新记忆状态 `c_state`，
   隐状态 `h_state` 递推传递到下一帧，从而把前一帧信息回灌当前帧，实现“空间结构 + 时间相关性”联合推断
   （`c_state = F_t*c_prev + I_t*C_t`，再经 `Refine` 输出当前帧去雨图）。
   默认用 `R_CLSTM_5`（隐状态宽 8）；开启 `--use_bilstm` 时额外用反向 R-CLSTM（`rev_maps` 倒序送入）
   做“前向 + 反向”双向时空交互，再用 5x5 卷积融合前后向结果。
6. 可选 `input_residue` 残差学习（输出 + 输入）。

**阶段二：精网络 FineNet / FineNet_npic（3D 卷积 + RRDB 精修）**
- 把粗输出 `c_out` 与原始输入 `c_in` 拼成 6 通道，经 `C_C3D_1`（3D 卷积，5x5x5 + 3x3x3 残差块、
  时间维 padding=0 压缩）在时间维上融合，再经若干 `RRDB` 残差密集块精修，最后输出（残留学习加粗结果）；
- `FineNet`（非 npic）只输出**中间那 1 帧**；`FineNet_npic`（`out_nc = d帧*3`，如 5 帧=15）**一次性输出全部 d 帧**。

**损失与训练**
- 实际损失是 **MSE**（`F.mse_loss`）：粗网络对整段 d 帧做 MSE 监督，精网络按 `--F_npic`
  监督全部帧或仅中间帧；`loss.py` 中的 Sobel/时空/分类损失为**备用**（默认未启用）。
- 两阶段**独立优化**（粗、精各一个优化器，`model_out_C.detach()` 切断到粗网络的反向传播）；
  `train.py` 里可冻结 `net_C` 或 `net_F`，按 `validate()` 的 PSNR 保存 best checkpoint。

## 目录结构导读（建议按此顺序阅读）

- `code/models/net.py` —— **先看**（整体框架：CoarseNet / FineNet / C_C3D，先理解 5 维张量约定）
- `code/models/R_CLSTM_modules.py` —— **其次**（时空交互核心：R_CLSTM_5 类 LSTM 门控逐帧循环）
- `code/models/modules.py` —— 编码器/解码器/MFF（LapSRN 上采样、多尺度融合）
- `code/models/backbone_dict.py`、`block.py`、`resnet/densenet/senet` —— backbone 与 RRDB 等基础构件
- `code/options/trainopt.py` / `testopt.py` —— 命令行参数定义
- `code/train.py`（单卡）/ `train_ddp.py`（DDP 多卡）/ `test.py` —— 训练/测试主循环
- `code/dataset/ntu_dataset.py`—— 视频序列数据集（一次读入 window_size=5 帧的 clip 与整段随机变换）
- `code/run_scripts/*.sh` —— 训练/测试的一键 bash 启动脚本（最实用的入口）
- `data/*.json`、`scripts/gen_json_*.py` —— 数据集 json 索引与生成脚本
- `code/options` 里注意可能在 `models/__init__.py` 出现“参数未定义仅保留逻辑”的注释，不影响主干运行

建议顺序：`net.py → R_CLSTM_modules.py → modules.py → run_scripts 的一个训练脚本 → train.py → ntu_dataset.py`。

## 训练 / 测试怎么跑（入口 + 关键参数）

入口统一是 bash 脚本（先按自己数据路径修改路径），核心命令形如
`python train.py --epochs 120 --data_root <数据路径> --train_file <train.json> --eval_file <val.json>
 --batch_size 2 --input_residue --F_npic --val_mode all --backbone resnet18 --refinenet R_CLSTM_5
 --checkpoint_dir_C ./checkpoint/C --checkpoint_dir_F ./checkpoint/F --lr_C 0.0001 --lr_F 0.0001 --use_bilstm --resume`
并 `cd code/run_scripts/` 后执行。

- **训练**（见 README 与 `run_scripts/`）：
  1. NTU：`bash train_resnet18_5pic.sh`（数据放 `/$YOUR_ROOTPATH/derain/NTU-derain`）；
  2. RainSyn25 light/heavy：`bash train_resnet18_rainsys_light_5pic.sh` / `train_resnet18_rainsys_heavy_5pic.sh`
     （数据放 `/$YOUR_ROOTPATH/derain/RainSyn25`）。
  - 关键参数：`--backbone`（resnet18 等）、`--refinenet R_CLSTM_5`、`--F_npic`（输出全部帧）、
    `--use_bilstm`（双向时空交互）、`--input_residue`（残差学习）、`--lr_C/--lr_F`、`--epochs`。
  - checkpoint 存到 `--checkpoint_dir_C/F`（含 `checkpoints_best_*.pth.tar`），日志存 `--logdir`。
- **测试 / 评估**：
  1. 下载预训练权重放入 `code/best_checkpoints`（README 的 gdrive 链接）；
  2. 相应测试脚本（`test_ntu_npic.sh` / `test_light_npic.sh` / `test_heavy_npic.sh`）生成复原帧；
  3. 用 `scripts/calculate_PSNR_SSIM.py` 计算 PSNR/SSIM 数值指标。
- 数据集说明：`data/` 里的 json 是**帧索引**文件，`scripts/gen_json_*.py` 用于从原始帧目录生成这些索引；
  NTU 数据集样本由 `window_size=5` 帧的含雨/干净帧对组成，经随机裁剪(224)/翻转/旋转后返回 `(C, T, H, W)`。

## 学习方法建议 / 易踩坑（基于注释阶段发现，如实记录）

- **建议**：先吃透 5 维张量 `(b,c,d,h,w)` 与 `cubes_2_maps/maps_2_cubes/rev_maps` 的“时间维↔batch 维”转换，
  再看 R-CLSTM 的类 LSTM 门控如何逐帧递推；把 `R_CLSTM_5` 与 `R_CLSTM_6` 对比理解“隐状态宽度”的影响。
- **已知疑似 bug / 笔误 / 遗留（未修改，保持原样）**：
  1. `train_ddp.py` 的 freeze 逻辑**语义是反的**（注释明确标注）——冻结时调用 `.train()`、训练时调用 `.eval()`，
     常规语义应相反；注释判定为原作者笔误，因被冻结模型不参与反向传播，数值影响有限但逻辑确实反了。
  2. `backbone_dict.py` 顶部从 `models` 导入 `densenet121/densenet169` 等，注释标注“理解存疑/疑似缺漏”。
  3. `modules.py` 中 `D_resnet` 最后一级输出通道数 `num_features//4` 及 `senet.py` 的 `forward(x, x_)`
     第二个参数 `x_` 未使用，均为“理解存疑/遗留参数”。
- **踩坑**：默认 `--val_mode all`（粗网络对所有帧算 PSNR），非 npic 模式下精网络只评估/输出中间帧；
  训练需 GPU（`DataParallel`/DDP），`R_CLSTM` 隐状态申请 `.to('cuda')` 硬编码在 CUDA 上；
  bash 脚本里的数据/json 路径必须改成你自己的绝对路径，否则加载即报错。
- `R_CLSTM_modules.py` 下方有大量被注释掉的实验版本（R_CLSTM_6/7/8/9/10 等，仅供思路对比，不参与运行，别混淆。

---
## Enhanced Spatio-Temporal Interaction Learning for Video Deraining: A Faster and Better Framework
Kaihao Zhang, Dongxu Li, Wenhan Luo, Wenqi Ren, Wei Liu

### Installation
To replicate the environment:

```bash
cd code
conda install --file requirements.txt
```

### Training
**Please first modify bash files accordingly with your data folder path.**

```bash
cd code/run_scripts
```
(1) Train on NTU dataset:
Put data under /$YOUR_ROOTPATH/derain/NTU-derain
```bash
cd code/run_scripts/
bash train_resnet18_5pic.sh
```

(2) Train on RainSys25 light dataset:
Put data under /$YOUR_ROOTPATH/derain/RainSyn25
```bash
cd code/run_scripts/
bash train_resnet18_rainsys_light_5pic.sh
```

(3) Train on RainSys25 heavy dataset:
Put data under /$YOUR_ROOTPATH/derain/RainSyn25
```bash
cd code/run_scripts/
bash train_resnet18_rainsys_heavy_5pic.sh
```

### Testing with pre-trained weights
**Please first modify bash files accordingly with your data folder path.**

Download checkpoints and put in ```code/best_checkpoints```  (https://drive.google.com/drive/folders/19PSF-slyB_m_0dWWo-VRe_uvGOlDiCPf?usp=sharing)

(1) Test on NTU dataset:
```bash
cd code/run_scripts/
bash test_ntu_npic.sh
```

(2) Test on RainSys25 light dataset:
```bash
cd code/run_scripts/
bash test_light_npic.sh
```

(3) Test on RainSys25 heavy dataset:
```bash
cd code/run_scripts/
bash test_heavy_npic.sh
```

### Citation
```bibtex
  @article{zhang2022enhanced,
    title={Enhanced Spatio-Temporal Interaction Learning for Video Deraining: A Faster and Better Framework},
    author={Zhang, Kaihao and Li, Dongxu and Luo, Wenhan and Ren, Wenqi and Liu, Wei},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
    year={2022}
  }
```
