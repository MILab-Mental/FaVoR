# CONFIGS 配置说明

本目录存放 FAVOR 项目的全部训练配置。所有训练都由 `app/main.py` 驱动：读取一个
**任务入口 YAML**，按其中的 `yamls` 字段递归合并若干**配置片段**（数据 / 优化器 /
模型 / 掩码损失），得到一份完整参数，再根据 `app` 字段分发到对应的 `train` 模块。

## 1. 目录结构

```
CONFIGS/
├── tasks/        # 任务入口（每个可运行任务一个文件，app/main.py --fname 指向这里）
├── datas/        # 数据配置片段（预训练 / 微调）
├── opt/          # 优化器与调度器片段
├── models/       # 模型配置片段（预训练 / 微调）
├── mask_loss/    # 掩码 + 损失配置（仅预训练）
├── test.yaml     # 预训练调试入口
├── test-finetune.yaml  # 微调调试入口
├── root.txt      # 数据集清单：split CSV ↔ 根目录 ↔ 预训练用途（v/a/va）
└── bk/           # 历史 / 参考配置备份（不再使用）
```

## 2. 配置加载与合并机制

入口在 `app/main.py::load_config`：

1. 读入口 YAML，取出 `yamls` 字段（可为 dict / list / 单路径）。
2. 按 `yamls` 的出现顺序依次加载片段文件，**后加载的片段覆盖先加载的**。
3. 最后把入口文件自身的键（除 `yamls` 外）叠加到合并结果上，**入口配置优先级最高**。

每个片段文件内部只有一个顶层键（`data:` / `optimization:` / `model:` / `loss:` + `mask:`），
因此片段间不会互相覆盖；`yamls` 里的 dict key（`data` / `opt` / `model` / `mask`）只是
标签，真正起作用的是 `.values()` 的顺序。

合并后得到的完整配置顶层键为：

| 键 | 来源 | 说明 |
|---|---|---|
| `app` | 入口 | 分发目标模块：`pretrain_v` / `finetune_v` |
| `folder` | 入口 | 输出目录（日志、checkpoint、参数快照） |
| `meta` | 入口 | 运行级参数（精度、seed、ckpt 路径等） |
| `data` | datas/ 片段 | 数据集与 dataloader |
| `data_aug` | datas/ 片段 | 数据增强 |
| `optimization` | opt/ 片段 | 优化器 / 调度器 |
| `model` | models/ 片段 | 模型结构 |
| `loss` + `mask` | mask_loss/ 片段 | 预训练损失与掩码（仅预训练） |

> 每次启动时，`app/main.py` 会把合并后的完整参数快照写入
> `{folder}/params-{app}.yaml`，便于复现与对比（`OUTPUT/` 下的那些
> `params-*.yaml` 就是自动生成的）。

## 3. 任务入口配置（tasks/）

一个典型的微调入口：

```yaml
app: finetune_v
folder: /home/data/sdc/FAVOR/OUTPUT/finetune_v/.../RAVDESS-emotion/
meta:
  dtype: bfloat16
  eval_freq: 1
  load_checkpoint: true
  read_checkpoint: /home/data/sdc/FAVOR/OUTPUT/pretrain_v/.../latest.pt
  reset_epoch: true
  save_every_freq: 10
  seed: 239
  frozen_encoder: true          # 现已统一注释掉，见第 8 节
yamls:
  data: /home/data/sdc/FAVOR/CONFIGS/datas/vfinetune-RAVDESS-emotion-cls8.yaml
  opt: /home/data/sdc/FAVOR/CONFIGS/opt/vfinetune-opt-250-40.yaml
  model: /home/data/sdc/FAVOR/CONFIGS/models/vfinetune-vit-l.yaml
```

### 3.1 `meta` 字段

| 字段 | 默认 | 预训练 | 微调 | 说明 |
|---|---|:---:|:---:|---|
| `dtype` | `float32` | ✓ | ✓ | 混合精度类型：`bfloat16` / `float16` / `float32` |
| `seed` | `0` | ✓ | ✓ | 随机种子 |
| `load_checkpoint` | — | ✓ | ✓ | 是否加载 checkpoint |
| `read_checkpoint` | `null` | ✓ | ✓ | 加载的 checkpoint 路径（预训练用 `CKPT/vjepa2/*.pt`，微调用预训练产物） |
| `reset_epoch` | `false` | ✓ | ✓ | 只加载权重、丢弃优化器状态并从头开始 epoch |
| `save_every_freq` | `-1` | ✓ | ✓ | 每 N 个 epoch 额外存一份 `e{epoch}.pt` |
| `eval_freq` | `1` | ✓ | ✓ | 微调：每 N 个 epoch 评测一次；预训练：每 N 个 epoch 触发一次「表征质量 event」（见下方 rankme / hessian_trace） |
| `rankme` | — | ✓ | — | 预训练：RankMe（encoder 特征有效秩）评测块，见下 |
| `hessian_trace` | — | ✓ | — | 预训练：损失 Hessian 迹（Hutchinson）评测块，见下 |
| `frozen_encoder` | `false` | — | ✓ | 是否冻结 encoder 只训 task head（当前已注释，见第 8 节） |
| `skip_batches` | `-1` | ✓ | — | 预训练：跳过前 N 个 batch |
| `sync_gc` | `false` | ✓ | — | 预训练：周期性手动 GC |

### 3.2 预训练表征质量指标（`eval_freq` / `rankme` / `hessian_trace`）

预训练的 `meta` 下可选择性开启两个 **checkpoint 表征质量**指标。二者都基于**固定的
数据子集 / 固定缓存 batch** 计算，因此不同 checkpoint（甚至不同进程重启）间可比。
实现见 `utils/eval_metrics.py`，接入见 `app/pretrain_v/train.py`。

| 字段 | 默认 | 说明 |
|---|---|---|
| `eval_freq` | `-1` | 每 N 个 epoch 构成一次 evaluation event（`-1` = 全关）。只有 `>0` 时指标才可能运行 |
| `rankme.enabled` | `false` | 是否计算 RankMe（在线 encoder 的全局池化特征做列去均值 SVD 后的有效秩 `exp(-Σ p log p)`） |
| `rankme.every_events` | `1` | 每隔几个 event 算一次 RankMe（event 计数自训练开始递增，最后一次 epoch 必算） |
| `rankme.subset_frac` | `0.01` | 固定 eval 子集占全量比例（种子抽样，跨 checkpoint 同一批视频） |
| `rankme.max_samples` | `4096` | 子集样本数上限（越小越省） |
| `rankme.feature` | `global` | 特征源：`global` 全局池化 / `tokens` 逐 clip 的 token 矩阵 / `both` |
| `rankme.seed` | `meta.seed` | 子集抽样种子 |
| `hessian_trace.enabled` | `false` | 是否计算训练损失 Hessian 的迹（Hutchinson 估计，无显式 Hessian） |
| `hessian_trace.every_events` | `1` | 每隔几个 event 算一次（Hessian 双反向很贵，建议调大） |
| `hessian_trace.n_hutchinson` | `5` | Hutchinson Rademacher 向量数 |
| `hessian_trace.n_clips` | `1` | 固定 batch 的 clip 数（迹与 batch 线性，固定以保证可比） |
| `hessian_trace.seed` | `meta.seed` | Hutchinson 随机向量种子（固定 -> 可比） |

**checkpoint 保存语义**（`app/pretrain_v/train.py`）：

- 指标全关（默认）：每个 epoch 按**最小平均 loss** 保存 `best_loss.pt`（原有行为不变）。
- 开了任一指标：除 `best_loss.pt` 照常保存外，另在 `{folder}/log_eval.csv` 逐 event 记录
  `epoch, train_loss, rankme_global, var_global, top_sv_energy, rankme_tokens, hessian_trace, hessian_std`，
  并按 **RankMe 最大** 保存 `best_rankme.pt`、按 **Hessian 迹最小** 保存 `best_trace.pt`。

**RankMe**：特征取在线 encoder 无 mask 输出的 token，做时间+空间全局池化得 `Z∈[N,D]`
（`feature: tokens` 时另对每个 clip 的 token 矩阵算逐 clip RankMe 后取均值）。列去均值后
在 fp64 CPU 上做 SVD：`p_i = σ_i²/Σσ²`，`RankMe = exp(−Σ p log p)`。辅助量 `var_global`
（每维方差均值）与 `top_sv_energy`（最大奇异值能量占比，接近 1 表示表征坍塌）。
RankMe 由所有 rank 各算自己的固定切片后 `all_gather`，SVD 仅在聚合后的完整矩阵上做一次。

**Hessian 迹**：只在 rank 0 做（其余 rank 以 barrier 同步）。取固定小 batch（clips +
masks 首次生成后缓存），`_pretrain_loss_on_sample` 忠实复刻 train.py 的
`forward_context / forward_target / loss_fn`，再
`g=∇_θ L (create_graph=True)`、对 Rademacher `v` 计算 `Hv=∇_θ(gᵀv)`、
`Tr(H)≈mean_v(vᵀHv)`，返回均值与标准差。Hutchinson 通过
flash / memory-efficient attention **没有二阶导**，故该 pass 会临时强制 SDPA 走 math 后端
（`utils/eval_metrics.py::_force_math_sdpa`）；同样会临时关闭 activation checkpointing
并切到 eval 模式。二阶 pass 本身在 fp32（bf16/fp16 autocast 下也只用 fp32 前向）。

> 实现规模注意：Hessian 双反向内存随模型参数增长，建议用 `n_clips=1`、低 `every_events`
> 观察显存；RankMe 成本主要在视频解码（`subset_frac`/`max_samples` 可调）。

## 4. 数据配置片段（datas/）

### 4.1 预训练 data（`vpretrain-data*.yaml`）

| 字段 | 说明 |
|---|---|
| `datasets` | manifest 列表，空格分隔的 `视频路径 标签` 文本文件 |
| `datasets_weights` | 每个数据集的采样权重（长度需与 `datasets` 一致） |
| `batch_size` / `num_workers` / `pin_mem` / `persistent_workers` | dataloader 参数 |
| `crop_size` / `patch_size` | 空间裁剪 / patch 大小 |
| `dataset_fpcs` | 每数据集每 clip 帧数列表（`frames per clip`），须与 `datasets` 一一对应 |
| `tubelet_size` | 时间维度 tubelet（3D patch 的时间深度） |
| `fps` | 抽帧帧率 |

预训练 `data_aug`：

| 字段 | 说明 |
|---|---|
| `random_resize_aspect_ratio` | 随机缩放长宽比范围 |
| `random_resize_scale` | 随机缩放比例范围 |
| `motion_shift` / `auto_augment` / `reprob` | 运动偏移 / 自动增强 / 随机 erase 概率 |

### 4.2 微调 data（`vfinetune-*.yaml`）

在预训练字段基础上新增：

| 字段 | 说明 |
|---|---|
| `datasets` | split CSV 列表（带表头，含 `video_path` + 标签列 + `{label}_split` 列） |
| `rootpaths` | 每个 CSV 对应的视频根目录（列表，与 `datasets` 一一对应） |
| `num_clips` | 每个样本采样的 clip 数 |
| `task` | `classification` / `regression`（`multi_label_classification` 未实现） |
| `num_class` | 分类类别数 |
| `label_column` | CSV 中作为标签的列名 |

> **split 约定**：`VideoCSVDataset` 读取 `{label_column}_split` 列，`split=0` 为训练集、
> `split=1` 为验证集。**分类标签是 1-based**（代码内部 `int(label) - 1` 转 0-based），
> 回归标签为浮点值。详见 `datasets/video_finetune_dataset.py`。

微调 `data_aug`：

| 字段 | 说明 |
|---|---|
| `random_resize_aspect_ratio` / `random_resize_scale` | 同预训练 |
| `random_horizontal_flip` | 水平翻转概率 |
| `reprob` | 随机 erase 概率 |

## 5. 优化配置片段（opt/）

### 5.1 预训练（`vpretrain-opt-*.yaml`）

| 字段 | 说明 |
|---|---|
| `ema` | `[起始, 结束]` 目标编码器动量（余弦从起始升到结束） |
| `epochs` | 训练 epoch 数 |
| `ipe` / `ipe_scale` | 每 epoch 迭代数 / 缩放系数 |
| `lr` / `start_lr` / `final_lr` | 峰值 / 初始 / 最终学习率（cosine + warmup） |
| `weight_decay` / `final_weight_decay` | 权重衰减 / 最终权重衰减 |
| `warmup` | warmup epoch 数 |
| `is_anneal` / `anneal_ckpt` / `resume_anneal` | 退火（cooldown）相关，仅退火配置使用 |

### 5.2 微调（`vfinetune-opt-*.yaml`）

| 字段 | 说明 |
|---|---|
| `epochs` / `warmup` | 同预训练 |
| `lr` / `start_lr` / `final_lr` | 同预训练 |
| `weight_decay` / `final_weight_decay` | 同预训练 |
| `betas` | Adam betas（默认 `[0.9, 0.999]`） |
| `eps` | Adam epsilon（默认 `1e-8`） |

## 6. 模型配置片段（models/）

### 6.1 预训练（`vpretrain-vit-*.yaml`）

| 字段 | 说明 |
|---|---|
| `model_name` | `vit_large` / `vit_huge` / `vit_giant_xformers`（见 `models/vision_transformer.py`） |
| `uniform_power` | 是否用均匀（无 CLS）patch 编码 |
| `use_rope` / `use_sdpa` / `use_silu` / `wide_silu` | RoPE / SDPA attention / SiLU 激活开关 |
| `use_activation_checkpointing` | 梯度检查点 |
| `pred_depth` / `pred_embed_dim` / `pred_num_heads` | predictor 结构 |
| `use_mask_tokens` / `zero_init_mask_tokens` | 可学习 mask token 及其零初始化 |

### 6.2 微调（`vfinetune-vit-*.yaml`）

| 字段 | 说明 |
|---|---|
| `model_name` | 同预训练，须与 `read_checkpoint` 的 backbone 一致 |
| `uniform_power` / `use_rope` / `use_sdpa` / `use_silu` / `wide_silu` / `use_activation_checkpointing` | 同预训练 |
| `out_layers` | 取哪些 transformer block 输出做多层级聚合（`ClipEncoder`） |
| `classifier_depth` / `classifier_num_heads` | 分类 / 回归头（`AttentiveClassifier`/`AttentiveRegressor`）深度与头数 |

`model_name` → `out_layers` 对应关系：

| model_name | out_layers |
|---|---|
| `vit_large` | `[17, 19, 21, 23]` |
| `vit_huge` | `[23, 25, 27, 31]` |
| `vit_giant_xformers` | `[31, 33, 35, 39]` |

## 7. 掩码与损失配置片段（mask_loss/）

仅预训练使用（`vjepa-pretrain.yaml`）：

| 字段 | 说明 |
|---|---|
| `loss.loss_exp` | JEPA 损失指数 `p`（`\|z−h\|^p / p`） |
| `mask` | 掩码块列表，支持多组掩码（multiblock）。每组字段：`num_blocks`（块数）、`spatial_scale` / `temporal_scale`（时空尺度范围）、`aspect_ratio`、`max_keep` / `max_temporal_keep`（保留上限）、`full_complement`（是否取补集） |

当前配置包含两组掩码：8 个小块（`spatial_scale 0.15`）+ 2 个大块（`spatial_scale 0.7`），
`temporal_scale` 均为 `[1.0, 1.0]`（保留完整时间维度）。

## 8. 微调任务总览（tasks/）

共 15 个微调入口，对应 7 个视频数据集：

| # | 任务入口 | 数据集 CSV | task | 输出 | label_column | 备注 |
|---|---|---|---|---|---|---|
| 1 | `finetune_v_AVEC2014-PHQ` | `3_AVEC2014_LV.csv` | regression | 标量 | `PHQ` | 抑郁评分 0~45 |
| 2 | `finetune_v_CREMA-D-emotion` | `2_CREMA-D_SV.csv` | classification | 6 类 | `emotion` | |
| 3 | `finetune_v_CREMA-D-intensity` | `2_CREMA-D_SV.csv` | regression | 标量 | `intensity` | 序数 0/1/2 |
| 4 | `finetune_v_EmotionTalk-emotion` | `2_EmotionTalk_SV.csv` | classification | 7 类 | `emotion` | |
| 5 | `finetune_v_IEMOCAP-activation` | `2_IEMOCAP_SV.csv` | regression | 标量 | `activation` | 1.0~5.0 |
| 6 | `finetune_v_IEMOCAP-dominance` | `2_IEMOCAP_SV.csv` | regression | 标量 | `dominance` | 0.5~5.0 |
| 7 | `finetune_v_IEMOCAP-emotion` | `2_IEMOCAP_SV.csv` | classification | 9 类 | `emotion` | |
| 8 | `finetune_v_IEMOCAP-valence` | `2_IEMOCAP_SV.csv` | regression | 标量 | `valence` | 1.0~5.5 |
| 9 | `finetune_v_MER2023-emotion` | `2_MER2023_SV.csv` | classification | 6 类 | `emotion` | |
| 10 | `finetune_v_MER2023-pos_intensity` | `2_MER2023_SV.csv` | regression | 标量 | `pos_intensity` | 0.0~9.25 |
| 11 | `finetune_v_MER242526-26openset` | `2_MER242526_SV.csv` | multi_label_classification | 23 类 | `26openset` | **占位，未实现** |
| 12 | `finetune_v_MER242526-emotion` | `2_MER242526_SV.csv` | classification | 6 类 | `emotion` | |
| 13 | `finetune_v_MER242526-pos_intensity` | `2_MER242526_SV.csv` | regression | 标量 | `pos_intensity` | 0~37 |
| 14 | `finetune_v_RAVDESS-emotion` | `2_RAVDESS_SV.csv` | classification | 8 类 | `emotion` | |
| 15 | `finetune_v_RAVDESS-intensity` | `2_RAVDESS_SV.csv` | classification | 2 类 | `intensity` | 标签已重编码为 {1,2}（1=normal / 2=strong） |

## 9. 预训练任务（tasks/）

| 任务入口 | 数据 | 帧数 | 说明 |
|---|---|---|---|
| `pretrain_v_FaVoR-112px-48f.yaml` | `pretrain_videos_0901.csv` | 48f | 主训练，`batch_size 64`，`epochs 150` |
| `pretrain_v_FaVoR-112px-48f-rep.yaml` | `pretrain_videos_0901.csv` | 48f | 主训练 + 表征质量监控（RankMe 每 2 event、Hessian 每 10 event），输出到 `.../FaVoR-112px-48f-rep`，存 `best_rankme.pt` / `best_trace.pt` |
| `pretrain_v_FaVoR-cooldown.yaml` | `pretrain_videos_0901.csv`（64f） | 64f | 退火：从主训练 `latest.pt` 继续，LR 退到 ~0，`epochs 40`，无 warmup |

退火通过 `opt/vpretrain-opt-cooldown.yaml` 的 `is_anneal: true` + `anneal_ckpt` + `resume_anneal: true`
与 `datas/vpretrain-data-cooldown.yaml`（`dataset_fpcs: [64]`、`batch_size 32`）实现。

## 10. 调试入口

多卡训练统一用 torchrun 启动：`app/main.py` 检测到 torchrun 注入的 `RANK/WORLD_SIZE` 后
以单进程身份直接运行（不再自行 fork）。等价写法二选一：

```bash
torchrun --nproc_per_node=2 -m app.main --fname <CONFIG.yaml>            # 用当前可见 GPU 前 2 张
torchrun --nproc_per_node=2 -m app.main --fname <CONFIG.yaml> --devices cuda:0 cuda:1  # 显式挑卡
```

单进程调试（`--debugmode True`，便于打断点，与生产同一套入口）：

```bash
python -m app.main --fname CONFIGS/test.yaml          --devices cuda:0 --debugmode True
python -m app.main --fname CONFIGS/test-finetune.yaml --devices cuda:0 --debugmode True
```

- `test.yaml` → 预训练调试（读 `CKPT/vjepa2/vitl.pt`，输出到 `OUTPUT/test`）。
- `test-finetune.yaml` → 微调调试（读预训练 `latest.pt`，跑 RAVDESS emotion，输出到 `OUTPUT/test_finetune`）。

## 11. 其他文件

- **`split_root_paths.csv`** — 数据集清单 TSV：`split_csv, root_path, original_csv_path, duration_mean, duration_median, duration_std, duration_var, duration_min, duration_max, sampled_mean_fps`。
  记录每个数据集 CSV 对应的媒体根目录（供 `merge_csv.py` 解析相对路径）及各时长统计列。
- **`bk/`** — 历史备份：
  - `bk/pretrain_v/vit{g,h,l}16/*.yaml`：早期 V-JEPA 标准分辨率（256px/384px）与 FaVoR 112px 的预训练/退火配置。
  - `bk/finetune_v/fintune-RAVDESS-*.yaml`：早期 RAVDESS 微调配置（旧字段名，如 `csv_path`/`freeze_backbone`，已废弃）。

## 12. 已知问题与注意事项

1. **`multi_label_classification` 未实现** —— `datasets/video_finetune_dataset.py` 与
   `app/finetune_v/train.py` 目前只支持 `classification` / `regression`。
   `finetune_v_MER242526-26openset` 及其 data 片段均为占位，不能直接运行。

2. **`frozen_encoder` 已统一注释** —— 15 个微调任务入口中的 `frozen_encoder: true` 现已
   注释掉，实际走默认值 `false`（解冻 encoder、端到端训练）。

3. **`use_sdpa` 位置** —— 必须写在 `model` 片段下（`app/pretrain_v/train.py` 从
   `cfgs_model` 读取）；早期 `bk/` 配置曾误写到 `meta` 下，会被忽略并回退到默认 `false`。

4. **`yamls` 路径** —— 片段路径可用绝对路径，也可用相对入口文件的相对路径（`load_config`
   会自动相对入口目录解析）。
