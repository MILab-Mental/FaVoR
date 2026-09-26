# CONFIGS 配置说明

本目录存放 FAVOR 项目的全部训练配置。所有训练都由 `app/main.py` 驱动：读取一个
**任务入口 YAML**，按其中的 `yamls` 字段递归合并若干**配置片段**（数据 / 优化器 /
模型 / 掩码损失），得到一份完整参数，再根据 `app` 字段分发到对应的 `train` 模块。

## 1. 目录结构

```text
CONFIGS/
├── tasks/                     # 启动训练时 --fname 指向这里
│   ├── pretrain/
│   │   ├── audio/{jepa,lejepa}/
│   │   ├── video/{jepa,lejepa}/
│   │   └── audio-video/lejepa/
│   └── finetune/
│       └── {audio,video,audio-video}/
│           └── {classification,regression,multilabel,mental}/
├── data/                      # 与 tasks 使用相同的阶段、模态、方法分组
│   ├── pretrain/
│   └── finetune/              # 数据集片段 + sampling/ 采样预设
├── models/                    # 按阶段、模态、方法分组的模型片段
├── optimization/              # 按阶段、模态、方法分组的优化片段
├── losses/                    # JEPA / LeJEPA 掩码与损失片段
├── archive/                   # 历史配置、参考配置和调试入口
├── datasets.csv               # 数据集清单（原 root.csv）
├── path-mapping.csv           # 完整的旧路径 → 新路径对照，用于兼容加载
└── README.md
```

### 命名与兼容约定

- 目录使用完整英文词；文件名用 `-` 分隔，数据集名称保留官方大小写。
- 微调入口和数据集片段使用 `<数据集>-<标签>.yaml`，任务类型由父目录说明。
- 采样预设明确写出单位，如 `48frames-8fps.yaml`、`audio-4s-4clips.yaml`。
- 优化配置按实际参数命名，如 `epochs80-warmup10-lr0.0004.yaml`。
  实验变体中的 `newopt/newnewopt` 也改为对应的 epoch、warmup 和学习率。
- 预训练入口由目录说明阶段和方法，文件名只保留实验信息；消融实验保留编号。
- `app:`、`folder:`、checkpoint 路径、数据路径和全部训练参数保持原值。
  `OUTPUT/` 的目录、日志、checkpoint 和已有参数快照不参与重命名。
- `app/main.py` 在路径不存在时查询 `path-mapping.csv`，兼容旧 `--fname` 和旧 `yamls` 引用。
  新命令应使用新路径；已有文件优先于映射，不改变正常配置加载行为。


**视频 / 音频两套微调的命名对应**（刻意保持一一平行）：

| | 视频 | 音频 |
|---|---|---|
| 任务入口 | `tasks/finetune/video/{classification,regression,multilabel}/<任务名>.yaml` | `tasks/finetune/audio/{classification,regression,multilabel}/<任务名>.yaml` |
| 多入口任务 | 保留 `tasks/finetune/video/<分组>/<任务名>/` 目录 | —（目前无） |
| 入口 `app:` | `finetune_v` | `finetune_a` |
| 数据片段 | `data/finetune/video/<类型>/<任务>.yaml` | `data/finetune/audio/<类型>/<任务>.yaml` |
| 采样片段 | `data/finetune/video/sampling/<帧数>frames-<帧率>fps.yaml`（由入口 `sampling:` 引用） | `data/finetune/audio/sampling/audio-4s-4clips.yaml` |
| 输出目录 | `OUTPUT/finetune_v/` | `OUTPUT/finetune_a/` |
| 代码模块 | `app/finetune_v/` ✅ 已实现 | `app/finetune_a/` ✅ 已实现 |

> 视频与音频侧任务目录的分组**完全一致**：`classification/` = `classification`、`regression/` = `regression`、
> `multilabel/` = `multi_label_classification`，与 `data/` 下的同名分组一一对应。

> **职责划分**：`classification/`、`regression/`、`multilabel/` 下的片段**只描述数据集是谁**（`datasets` / `datasets_weights` /
> `rootpaths` / `task` / `num_class` / `label_column`），**不含任何采样或 dataloader 参数**；
> `batch_size` / `crop_size` / `patch_size` / `dataset_fpcs` / `tubelet_size` / `fps` / `num_workers` /
> `persistent_workers` / `pin_mem` / `num_clips` 与 `data_aug` 全部来自顶层**采样预设片段**，
> 由入口的 `sampling:` 键指定。
>
> 因此采样预设**不能单独使用**（缺 `datasets` / `rootpaths` / `label_column` 会在
> `app/finetune_v/train.py:208-214` 处 KeyError），必须与一个数据集片段叠加。
>
> 这一划分对视频和音频微调入口都生效。视频采样预设提供帧数、fps 和图像增强；
> 音频采样预设提供采样率、crop 时长、clip 数和 dataloader 参数。任务数据片段不再复制这些键。

### VA-LeJEPA 预训练的数据异常处理

`CONFIGS/tasks/pretrain/audio-video/lejepa/` 的入口直接使用 canonical paired manifest，
不要求预先运行全量 `manifest_tools validate`。公共数据配置默认
`data.on_decode_error: skip`：读取样本时检查媒体与同步，失败则跳过整对 AV，
不独立替换任一模态。设为 `error` 可恢复遇错立即退出的行为。

DDP 每个 batch 取各 rank 有效样本数的最小值；任一 rank 没有有效样本时，所有 rank
一起跳过该 batch，optimizer step 和梯度累积计数不增加。异常详情写入
`{folder}/logs/skipped_pairs_rank{rank}.jsonl`，本次启动以来的全局异常数量、
为对齐 DDP 数量而丢弃的有效样本数和跳过的 batch 数写入 `logs/data_skips.csv`。
若一整轮没有可用 batch，则报错结束，避免无进展循环。

### 临床 mental 微调任务

临床数据源为 `/home/data/sdb/基座模型数据集/3心理数据集/临床/split-0901.csv`，
项目内保留文件名 `DATASET/splits-0901/3_私有临床数据_LV.csv`，内容已替换为新数据源。
`CONFIGS/datasets.csv` 中的 `original_csv_path` 同时指向该新文件。

38 个任务分别提供 audio、video、audio-video 三种入口，共 114 份任务配置和 114 份数据片段：

| 内容 | 路径 |
|---|---|
| 任务入口 | `CONFIGS/tasks/finetune/{audio,video,audio-video}/mental/<任务名>.yaml` |
| 数据片段 | `CONFIGS/data/finetune/{audio,video,audio-video}/mental/<任务名>.yaml` |
| 音频输出 | `OUTPUT/finetune_a/mental/<任务名>/` |
| 视频输出 | `OUTPUT/finetune_v/mental/<任务名>/` |
| 音视频输出 | `OUTPUT/finetune_va/mental/<任务名>/` |
| 音视频配对清单 | `DATASET/merged/mental/clinical_canonical.csv` |

任务包含 DSM 13 项、SCL 10 项、DSMxSCL 9 项、PAYKEL、MINI、MOAS，以及 RESULT 3 项。
全部使用 `classification`：PAYKEL 为 4 分类，MINI / MOAS 为 3 分类，其余为 2 分类。
`label_column` 与源 CSV 的列名一致；各任务使用自己的 `<任务名>_split`，0 为训练、1 为验证，
空值 / -1 不参与该任务。类别保留 CSV 的从 1 开始的编码，由加载器转换为从 0 开始的目标。
配对清单保留全部标签和 split 列，增加唯一 `pair_id`、`source_id`，媒体路径转为绝对路径。

采样、模型、优化器和 checkpoint 继承各模态现有分类默认配置。音视频配置中的
`finetune.pretrained_checkpoint` 默认为 `OUTPUT/pretrain_va_lejepa/I3-C1/latest.pt`，
运行时需要该文件，或通过 `--set finetune.pretrained_checkpoint=/实际路径/checkpoint.pt` 指定。

```bash
# 重新同步临床 CSV、生成配对清单和全部 mental 配置（会覆盖已有同名 mental 配置）
python -m utils.prepare_clinical_mental

# 以 DSM-Depression 为例启动音频微调；视频 / 音视频改用对应模态目录
torchrun --nproc_per_node=2 -m app.main \
  --fname CONFIGS/tasks/finetune/audio/mental/DSM-Depression.yaml
```

数据与配对清单位于被 Git 忽略的 `DATASET/`；其他机器使用前需执行上述同步命令。

### 音视频下游任务配置

`CONFIGS/tasks/finetune/audio-video/` 现有 53 个具体任务入口：普通分类 7 项、多标签 1 项、
回归 7 项，以及独立 `mental/` 下的 38 项临床分类。普通任务与视频侧的同名任务对应：

| 分组 | 任务 |
|---|---|
| classification | CREMA-D-emotion、EmotionTalk-emotion、IEMOCAP-emotion、MER2023-emotion、MER242526-emotion、RAVDESS-emotion、RAVDESS-intensity |
| multilabel | MER242526-26openset |
| regression | AVEC2014-PHQ、CREMA-D-intensity、IEMOCAP-activation、IEMOCAP-dominance、IEMOCAP-valence、MER2023-pos-intensity、MER242526-pos-intensity |

每个入口引用对应的 `CONFIGS/data/finetune/audio-video/<分组>/<任务名>.yaml`、公共采样、模型
和优化器片段。输出为 `OUTPUT/finetune_va/<分组>/<任务名>/`；类别数与原任务一致，回归
`num_class: 1`，分类 / 多标签以 `f1_macro` 最大值选择最佳模型，回归以 `mae` 最小值选择。
重复的三个 `default.yaml` 入口和三个 `va.yaml` 数据片段已删除，直接使用具体任务配置。
旧路径通过 `path-mapping.csv` 分别兼容映射到 RAVDESS-emotion、MER242526-26openset、
AVEC2014-PHQ；更早的 `vafinetune` 路径也直接映射到对应具体任务。

普通配对清单位于 `DATASET/merged/finetune_va/<原split文件名>.csv`，临床清单仍位于
`DATASET/merged/mental/clinical_canonical.csv`。清单保留所有原标签 / split 列，仅纳入同时有
音频和视频路径的样本，并验证音视频身份和配对唯一性。RAVDESS 的 `02-` 视频与 `03-` 音频
使用现有录制身份规则配对。采样预设只含采样参数，避免覆盖任务自己的清单。

```bash
# 生成 / 更新全部 53 个 VA 任务、配对清单和训练 / 验证样本数报告
# 会覆盖已有同名 VA 入口和数据片段，不生成通用占位配置
python -m utils.prepare_va_finetune

torchrun --nproc_per_node=2 -m app.main \
  --fname CONFIGS/tasks/finetune/audio-video/classification/RAVDESS-emotion.yaml \
  --set finetune.pretrained_checkpoint=/实际路径/va-pretrain.pt
```

各任务有效样本数写入 `DATASET/merged/finetune_va/tasks.csv`。数据清单被 Git 忽略，其他机器
需运行生成命令。默认 checkpoint 仍为 VA 预训练入口产出的
`OUTPUT/pretrain_va_lejepa/I3-C1/latest.pt`，需要先完成 VA 预训练或指定已有兼容 checkpoint；
单模态 checkpoint 不能直接替代。生成过程验证清单和目标值，不执行全量媒体解码 / 同步校验。

## 2. 配置加载与合并机制

入口在 `app/main.py::load_config`：

1. 读入口 YAML，取出 `yamls` 字段（可为 dict / list / 单路径）。
2. 按 `yamls` 的出现顺序依次加载片段文件，**后加载的片段覆盖先加载的**。
3. 最后把入口文件自身的键（除 `yamls` 外）叠加到合并结果上，**入口配置优先级最高**。
4. 合并完成后再应用命令行的 `--set 键=值` 覆盖，**优先级比入口文件还高**（见 10.1）。

微调入口的 `yamls` 采用固定的四键布局，把「数据集是谁」与「怎么采样」拆成两个片段：

```yaml
yamls:
  data:     CONFIGS/data/finetune/video/classification/CREMA-D-emotion.yaml
  sampling: CONFIGS/data/finetune/video/sampling/48frames-8fps.yaml
  opt:      CONFIGS/optimization/finetune/video/epochs80-warmup5-lr5e-05.yaml
  model:    CONFIGS/models/finetune/video/vit-l-4layers.yaml
```

`yamls` 路径支持三种形式：

- `CONFIGS/...`：相对于仓库根目录，推荐写法，不依赖入口 YAML 的目录深度；
- `../../../data/...`：相对于入口 YAML 所在目录，保留兼容；
- `/absolute/path/...`：绝对路径，保留兼容。

若相对路径不存在，启动时的 `FileNotFoundError` 会列出所有尝试过的完整路径。

`data` 与 `sampling` 都指向 `data/` 片段、都写进合并结果的同一个 `data` 顶层键，但两者的键集
**互不重叠**（数据集标识 vs 采样参数），因此谁先谁后都不影响结果。`sampling` 是可换的旋钮：
改一行就能把一个任务从 48 帧/8fps 换成 64 帧/16fps，不必碰数据集定义。

每个片段文件内部只有一个顶层键（`data:` / `optimization:` / `model:` / `loss:` + `mask:`），
因此片段间不会互相覆盖；`yamls` 里的 dict key（`data` / `opt` / `model` / `mask`）只是
标签，真正起作用的是 `.values()` 的顺序。

合并后得到的完整配置顶层键为：

| 键 | 来源 | 说明 |
|---|---|---|
| `app` | 入口 | 分发目标模块：`pretrain_v` / `finetune_v` / `pretrain_a` / `finetune_a` |
| `folder` | 入口 | 输出目录（日志、checkpoint、参数快照） |
| `meta` | 入口 | 运行级参数（精度、seed、ckpt 路径等） |
| `data` | `data/` 片段 | 数据集与 dataloader |
| `data_aug` | `data/` 片段 | 数据增强 |
| `optimization` | optimization/ 片段 | 优化器 / 调度器 |
| `model` | models/ 片段 | 模型结构 |
| `loss` + `mask` | losses/ 片段 | 预训练损失与掩码（仅预训练） |

> 每次启动时，`app/main.py` 会把合并后的完整参数快照写入
> `{folder}/params-{app}.yaml`，便于复现与对比（`OUTPUT/` 下的那些
> `params-*.yaml` 就是自动生成的）。

## 3. 任务入口配置（tasks/）

一个典型的微调入口：

```yaml
app: finetune_v
folder: OUTPUT/finetune_v/.../RAVDESS-emotion/
meta:
  dtype: bfloat16
  eval_freq: 1
  load_checkpoint: true
  read_checkpoint: OUTPUT/pretrain_v/.../latest.pt
  reset_epoch: true
  save_every_freq: 10
  seed: 239
  frozen_encoder: true          # 现已统一注释掉，见第 8 节
yamls:
  data: CONFIGS/data/finetune/video/classification/RAVDESS-emotion.yaml
  sampling: CONFIGS/data/finetune/video/sampling/48frames-8fps.yaml
  opt: CONFIGS/optimization/finetune/video/epochs250-warmup40-lr0.000525.yaml
  model: CONFIGS/models/finetune/video/vit-l.yaml
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

## 4. 数据配置片段（data/）

### 4.1 预训练 data（`data/pretrain/video/jepa/*.yaml`）

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

### 4.2 微调 data（`data/finetune/video/{classification,regression,multilabel}/*.yaml`）

在预训练字段基础上新增：

| 字段 | 说明 |
|---|---|
| `datasets` | split CSV 列表（带表头，含 `video_path` + 标签列 + `{label}_split` 列） |
| `rootpaths` | 每个 CSV 对应的视频根目录（列表，与 `datasets` 一一对应） |
| `num_clips` | 每个样本采样的 clip 数 |
| `task` | `classification` / `regression` / `multi_label_classification` |
| `num_class` | 分类类别数（`multi_label_classification` 必填且须 ≥2） |
| `label_column` | CSV 中作为标签的列名 |

> **split 约定**：`VideoCSVDataset` 读取 `{label_column}_split` 列，`split=0` 为训练集、
> `split=1` 为验证集。**分类标签是 1-based**（代码内部 `int(label) - 1` 转 0-based），
> 回归标签为浮点值，**多标签是竖线分隔的 1-based 索引**（如 `2|11|20`，解析为 0-based
> multi-hot 向量）。详见 `datasets/video_finetune_dataset.py`。

> **`video_path` 不能为空**：`VideoCSVDataset` 逐行构造样本时要求 `video_path` 非空
> （`_load_data_path` 会抛 `ValueError: Empty video_path`）——**空值不是「跳过」，是报错**。
> 音频数据集的 CSV（`2_*_SA`、`3_*_LA` 等）整列 `video_path` 为空，因此无法用于本流水线，
> 见 8.2 与 12.2。

微调 `data_aug`：

| 字段 | 说明 |
|---|---|
| `random_resize_aspect_ratio` / `random_resize_scale` | 同预训练 |
| `random_horizontal_flip` | 水平翻转概率 |
| `reprob` | 随机 erase 概率 |

## 5. 优化配置片段（optimization/）

### 5.1 预训练（`optimization/pretrain/video/jepa/*.yaml`）

| 字段 | 说明 |
|---|---|
| `ema` | `[起始, 结束]` 目标编码器动量（余弦从起始升到结束） |
| `epochs` | 训练 epoch 数 |
| `ipe` / `ipe_scale` | 每 epoch 迭代数 / 缩放系数 |
| `lr` / `start_lr` / `final_lr` | 峰值 / 初始 / 最终学习率（cosine + warmup） |
| `weight_decay` / `final_weight_decay` | 权重衰减 / 最终权重衰减 |
| `warmup` | warmup epoch 数 |
| `is_anneal` / `anneal_ckpt` / `resume_anneal` | 退火（cooldown）相关，仅退火配置使用 |

### 5.2 微调（`optimization/finetune/video/*.yaml`）

| 字段 | 说明 |
|---|---|
| `epochs` / `warmup` | 同预训练 |
| `lr` / `start_lr` / `final_lr` | 同预训练 |
| `weight_decay` / `final_weight_decay` | 同预训练 |
| `betas` | Adam betas（默认 `[0.9, 0.999]`） |
| `eps` | Adam epsilon（默认 `1e-8`） |

## 6. 模型配置片段（models/）

### 6.1 预训练（`models/pretrain/video/jepa/vit-*.yaml`）

| 字段 | 说明 |
|---|---|
| `model_name` | `vit_large` / `vit_huge` / `vit_giant_xformers`（见 `models/video_jepa/vision_transformer.py`） |
| `uniform_power` | 是否用均匀（无 CLS）patch 编码 |
| `use_rope` / `use_sdpa` / `use_silu` / `wide_silu` | RoPE / SDPA attention / SiLU 激活开关 |
| `use_activation_checkpointing` | 梯度检查点 |
| `pred_depth` / `pred_embed_dim` / `pred_num_heads` | predictor 结构 |
| `use_mask_tokens` / `zero_init_mask_tokens` | 可学习 mask token 及其零初始化 |

### 6.2 微调（`models/finetune/video/vit-*.yaml`）

| 字段 | 说明 |
|---|---|
| `model_name` | 同预训练，须与 `read_checkpoint` 的 backbone 一致 |
| `uniform_power` / `use_rope` / `use_sdpa` / `use_silu` / `wide_silu` / `use_activation_checkpointing` | 同预训练 |
| `out_layers` | 取哪些 transformer block 输出做多层级聚合（`ClipEncoder`） |
| `classifier_depth` / `classifier_num_heads` | 分类 / 回归头（`AttentiveClassifier`/`AttentiveRegressor`）深度与头数 |
| `backbone_dropout` | backbone attention 输出投影与 MLP dropout，默认 `0.0` |
| `backbone_attention_dropout` | backbone attention probability dropout，默认 `0.0` |
| `backbone_drop_path` | backbone stochastic-depth 最大概率（各层从 0 线性递增），默认 `0.0` |
| `classifier_dropout` | attentive pooler 输出与最终 linear 之间的 dropout，默认 `0.0` |

`finetune_v` 的 `optimization` 还可启用 Grokfast EMA 梯度滤波：

| 字段 | 默认 | 说明 |
|---|---:|---|
| `grokfast` | `false` | 是否在反向传播与 optimizer step 之间应用 Grokfast |
| `grokfast_alpha` | `0.98` | 慢梯度 EMA 系数，须位于 `[0, 1)` |
| `grokfast_lambda` | `2.0` | EMA 慢梯度的放大系数，须非负 |

`model_name` → `out_layers` 对应关系：

| model_name | out_layers |
|---|---|
| `vit_large` | `[17, 19, 21, 23]` |
| `vit_huge` | `[23, 25, 27, 31]` |
| `vit_giant_xformers` | `[31, 33, 35, 39]` |

## 7. 掩码与损失配置片段（losses/）

仅预训练使用（`video-jepa.yaml`）：

| 字段 | 说明 |
|---|---|
| `loss.loss_exp` | JEPA 损失指数 `p`（`\|z−h\|^p / p`） |
| `mask` | 掩码块列表，支持多组掩码（multiblock）。每组字段：`num_blocks`（块数）、`spatial_scale` / `temporal_scale`（时空尺度范围）、`aspect_ratio`、`max_keep` / `max_temporal_keep`（保留上限）、`full_complement`（是否取补集） |

当前配置包含两组掩码：8 个小块（`spatial_scale 0.15`）+ 2 个大块（`spatial_scale 0.7`），
`temporal_scale` 均为 `[1.0, 1.0]`（保留完整时间维度）。

## 8. 微调任务总览（tasks/）

`DATASET/splits-0901/tasks.json` 定义了 **33 条**下游任务，覆盖 **20 个**数据集。本节为全部
33 个语义任务都建立了音频入口；其中 15 条同时具有视频入口。

- **视频（15 条）** → `tasks/finetune/video/{classification,regression,multilabel}/<任务名>.yaml`（`app: finetune_v`，已实现）
- **音频（33 条）** → `tasks/finetune/audio/{classification,regression,multilabel}/<任务名>.yaml`（`app: finetune_a`）

> `app/finetune_v` 解码 `video_path`，`app/finetune_a` 解码 `audio_path`；两边共用
> `<label_column>_split`、1-based 分类标签、任务类型和统一启动器的配置契约。

数据片段按任务类型分目录：`classification/` = `classification`、`regression/` = `regression`、
`multilabel/` = `multi_label_classification`。消融 / 复现实验**不单独建目录**，它们的差异
全部落在入口引用的采样预设上（见 8.4）。

### 8.1 视频数据集（15 条）

入口均为 `tasks/finetune/video/<分组>/<下表中的名字>.yaml`。分组目录取该行 `task` 列的
对应项：`classification` → `classification/`、`regression` → `regression/`、`multi_label_classification` → `multilabel/`。
本节 14 条按此规则分布为 **`classification/` 7 条、`regression/` 7 条**；第 15 条可运行任务在 `multilabel/` 下，
见 8.3。

| # | 任务入口 | 数据集 CSV | task | 输出 | `label_column` | 备注 |
|---|---|---|---|---|---|---|
| 1 | `CREMA-D-emotion` | `2_CREMA-D_SV.csv` | classification | 6 类 | `emotion` | |
| 2 | `CREMA-D-intensity` | `2_CREMA-D_SV.csv` | regression | 标量 | `intensity` | 有序 0/1/2 |
| 3 | `EmotionTalk-emotion` | `2_EmotionTalk_SV.csv` | classification | 7 类 | `emotion` | |
| 4 | `IEMOCAP-emotion` | `2_IEMOCAP_SV.csv` | classification | 9 类 | `emotion` | |
| 5 | `IEMOCAP-valence` | `2_IEMOCAP_SV.csv` | regression | 标量 | `valence` | 1.0~5.5 |
| 6 | `IEMOCAP-activation` | `2_IEMOCAP_SV.csv` | regression | 标量 | `activation` | 1.0~5.0 |
| 7 | `IEMOCAP-dominance` | `2_IEMOCAP_SV.csv` | regression | 标量 | `dominance` | 0.5~5.0 |
| 8 | `MER2023-emotion` | `2_MER2023_SV.csv` | classification | 6 类 | `emotion` | |
| 9 | `MER2023-pos_intensity` | `2_MER2023_SV.csv` | regression | 标量 | `pos_intensity` | 0.0~9.25 |
| 10 | `MER242526-emotion` | `2_MER242526_SV.csv` | classification | 6 类 | `emotion` | 另有 18 个变体目录，见下注 |
| 11 | `MER242526-pos_intensity` | `2_MER242526_SV.csv` | regression | 标量 | `pos_intensity` | 0~37 |
| 12 | `RAVDESS-emotion` | `2_RAVDESS_SV.csv` | classification | 8 类 | `emotion` | 另有 3 个变体目录，见下注 |
| 13 | `RAVDESS-intensity` | `2_RAVDESS_SV.csv` | classification | 2 类 | `intensity` | 标签已重编码为 {1,2}（1=normal / 2=strong） |
| 14 | `AVEC2014-PHQ` | `3_AVEC2014_LV.csv` | regression | 标量 | `PHQ` | 抑郁评分 0~45 |

> **上表 14 条已全部扁平化为 `<任务名>.yaml`**（如 `classification/CREMA-D-emotion.yaml`）。其中两个任务
> 另带实验变体，因此**入口文件与同名变体目录并存**：
>
> - `classification/MER242526-emotion.yaml`（默认入口）+ `classification/MER242526-emotion/`（18 个
>   `favor-*.yaml` 变体）。默认入口是 `favor-e5-4layers-48frames-8fps-epochs80-warmup5-lr5e-5` 的
>   **等价配置**——它是这批实验里 acc/F1 双料第一（`0.5692` / `0.4907` @ epoch 10）。
>   注意它沿用该变体自己的 `folder:`，所以默认入口与原变体**输出到同一个目录**。
> - `classification/RAVDESS-emotion.yaml`（默认入口，由原 `finetune_v.yaml` 剪切而来）+
>   `classification/RAVDESS-emotion/`（3 个变体：`-vjepa` / `_reproduce` / `_reproduce_e5`）。
>
> 这 18 + 3 个变体入口的采样同样由 `sampling:` 预设给出（13 个历史 MER242526 变体用
> `48-4`，`-data48-8` / `-data64-16` 命名的用 `48-8` / `64-16`，两个 RAVDESS reproduce
> 用 `128-step2-RAVDESS-reproduce`），见 8.4。

### 8.2 仅音频数据集对应的单标签任务（16 条）

入口为 `tasks/finetune/audio/{classification,regression}/<下表中的名字>.yaml`（`app: finetune_a`）。
这些入口均由 `app/finetune_a` 使用 `audio_path` 运行，并共用确定性的验证集多裁剪和
AudioJEPA audio backbone。分类标签按 CSV 的 1-based 编码转为训练时的 0-based 编码。

| # | 任务入口 | 数据集 CSV | task | 输出 | `label_column` | 备注 |
|---|---|---|---|---|---|---|
| 1 | `ASVP-ESD-emotion` | `2_ASVP-ESD_SA.csv` | classification | 12 类 | `emotion` | 13,964 行 |
| 2 | `ASVP-ESD-intensity` | `2_ASVP-ESD_SA.csv` | classification | 2 类 | `intensity` | 1=normal / 2=high |
| 3 | `CASIA-emotion` | `2_CASIA_SA.csv` | classification | 6 类 | `emotion` | 1,200 行 |
| 4 | `CSEMOTIONS-emotion` | `2_CSEMOTIONS_SA.csv` | classification | 7 类 | `emotion` | 4,160 行 |
| 5 | `EMNS-emotion` | `2_EMNS_SA.csv` | classification | 8 类 | `emotion` | 1,205 行 |
| 6 | `EMNS-intensity` | `2_EMNS_SA.csv` | regression | 标量 | `intensity` | 0~10 |
| 7 | `ESD-Chinese-emotion` | `2_ESD-Chinese_SA.csv` | classification | 5 类 | `emotion` | 35,000 行 |
| 8 | `Androids-Corpus-label` | `3_Androids-Corpus_LA.csv` | classification | 2 类 | `label` | 1=HC / 2=PT |
| 9 | `CMDC-label` | `3_CMDC_LA.csv` | classification | 2 类 | `label` | 1=HC / 2=MDD |
| 10 | `CMDC-PHQ` | `3_CMDC_LA.csv` | regression | 标量 | `PHQ` | 0~25 |
| 11 | `CMDC-HAMD` | `3_CMDC_LA.csv` | regression | 标量 | `HAMD` | 9~24 |
| 12 | `MODMA-label` | `3_MODMA_SA.csv` | classification | 2 类 | `label` | 1=HC / 2=MDD |
| 13 | `CNSCED-intensity` | `2_CNSCED_SA.csv` | regression | 标量 | `intensity` | 0~3，多标签强度的逐行平均 |
| 14 | `EMOVIE-pos_intensity` | `2_EMOVIE_SA.csv` | regression | 标量 | `pos_intensity` | 0~4 |
| 15 | `EATD-Corpus-SDS` | `3_EATD-Corpus_SA.csv` | regression | 标量 | `SDS` | 20~66 |
| 16 | `PDCH-HAMD` | `3_PDCH_LA.csv` | regression | 标量 | `HAMD` | 1~35 |

> `Androids-Corpus/` 与 `CMDC/` 下有名为 `face_videos/` 的目录，但里面只有 `.wav`，**没有视频**。

### 8.3 `multi_label_classification`（3 条：音频侧全部可运行）

| # | 任务入口 | 所在目录 | 数据集 CSV | 输出 | `label_column` | 状态 |
|---|---|---|---|---|---|---|
| 1 | `MER242526-26openset` | `tasks/finetune/{video,audio}/multilabel/MER242526-26openset.yaml` | `2_MER242526_SV.csv` | 23 类，`\|` 分隔 | `26openset` | 视频 / 音频均可运行 |
| 2 | `CNSCED-emotion` | `tasks/finetune/audio/multilabel/CNSCED-emotion.yaml` | `2_CNSCED_SA.csv` | 7 类，`\|` 分隔 | `emotion` | 音频可运行 |
| 3 | `M3ED-emotion` | `tasks/finetune/audio/multilabel/M3ED-emotion.yaml` | `2_M3ED_SA.csv` | 7 类，`\|` 分隔 | `emotion` | 音频可运行 |

标签列写法如 `5|7`，是**竖线分隔的 1-based 类别索引**，由对应模态的 CSV dataset 解析为 0-based
multi-hot 向量（维度 = `data.num_class`，该键对多标签**必填且须 ≥2**，否则加载即报错）。
训练用逐类 `pos_weight = 负样本数 / 正样本数` 加权的 `BCEWithLogitsLoss`，sigmoid 后以
阈值 `0.5` 出预测，best 按 `val_f1_macro` 最大选取（与单标签一致）。音频侧 3 条任务都由
`app/finetune_a` 的 multi-hot 标签解析和 `BCEWithLogitsLoss` 支持。

### 8.4 采样预设

采样参数集中在 `data/finetune/video/sampling/`的 4 个预设片段，由入口的 `sampling:` 键引用（见第 2 节）。
这 4 个片段是**唯一**的采样来源——`classification/`、`regression/`、`multilabel/` 下的数据集片段只描述「数据集是谁」，
不含任何采样键（见第 1 节的职责划分）。

音频侧的 33 个入口共用 `data/finetune/audio/sampling/audio-4s-4clips.yaml`：16 kHz、4 秒 crop、
每条样本 4 个 clip，并在该片段统一设置 batch size 和 dataloader 参数。

| 预设 | `dataset_fpcs` | `fps` | `frame_step` | 引用它的入口 |
|---|---|---|---|---|
| `48frames-8fps.yaml` | `[48]` | 8 | — | **19 个**：8.1 的 15 个任务入口 + `RAVDESS-emotion/vjepa.yaml` + 3 个 `...-data48-8-...` 变体 |
| `48frames-4fps.yaml` | `[48]` | 4 | — | **14 个**：`MER242526-emotion` 的 13 个历史变体 + `CONFIGS/archive/test-finetune.yaml` |
| `64frames-16fps.yaml` | `[64]` | 16 | — | **2 个**：`...-data64-16-new{new,}opt.yaml` 变体 |
| `RAVDESS-128frames-step2-reproduce.yaml` | `[128]` | 无（`fps` 未设 → 源视频原帧率） | 2 | **2 个**：`RAVDESS-emotion/reproduce{,_e5}.yaml` |

消融 / 复现实验**不再有专属数据片段**：`dataset_fpcs` / `fps` / `frame_step` 的差异就是消融的
自变量本身，现在一律用「入口换一个 `sampling:` 预设」表达（例：MER242526 的 64 帧消融 =
同一份 `classification/MER242526-emotion.yaml` + `64frames-16fps.yaml`）。新增一档采样 = 加一个预设片段，
数据集片段与任务入口都不必动。

> **为什么 `MER242526-emotion` 的 13 个历史变体是 `48-4` 而不是 `48-8`**：这些入口历史上一直
> 吃数据片段里的 `fps: 4`（该 `fps` 键对 8.1 的任务入口是死键，因为那些入口一律覆盖成 8）。
> 改成 `48-8` 会把帧率翻倍，`run.sh:25-113` 记录的那些指标（如 `favor-e11-23out` 的
> `best acc 0.5219 @ epoch 15`）就不再可比。想统一到 8fps 的话，把这 13 个入口的 `sampling:`
> 改成 `48frames-8fps.yaml` 即可。

> 采样预设统一使用 `<帧数>frames-<帧率>fps.yaml`，直接按文件名中的单位读取。

> **`pin_mem` 已统一为 `false`**：4 个预设全部显式写 `data.pin_mem: false`，因此 36 个入口
> 合并后的 `pin_mem` 都是 `false`。该键只影响 dataloader 的锁页内存，不影响任何指标；
> 历史上片段写的是 `true`，改动后仅可能影响吞吐。

## 9. 预训练任务（tasks/pretrain/video/jepa/）

| 任务入口 | 数据 | 帧数 | 说明 |
|---|---|---|---|
| `favor-112px-48frames-4fps.yaml` | `pretrain_videos_0901.csv` | 48f | 主训练，`batch_size 64`，`epochs 150` |
| `favor-112px-64frames-8fps-rankme.yaml` | `pretrain_videos_0901.csv` | 64f | RankMe 监控，沿用原 checkpoint 和输出目录 |
| `favor-112px-64frames-8fps-rankme-resume-epoch32.yaml` | `pretrain_videos_0901.csv` | 64f | 从原 epoch 32 checkpoint 续训 |
| `favor-cooldown.yaml` | `pretrain_videos_0901.csv`（64f） | 64f | 退火：从主训练 `latest.pt` 继续，LR 退到 ~0，`epochs 40`，无 warmup |

退火通过 `optimization/pretrain/video/jepa/cooldown.yaml` 的 `is_anneal: true` + `anneal_ckpt` + `resume_anneal: true`
与 `data/pretrain/video/jepa/cooldown.yaml`（`dataset_fpcs: [64]`、`batch_size 32`）实现。

## 10. 调试入口

多卡训练统一用 torchrun 启动：`app/main.py` 检测到 torchrun 注入的 `RANK/WORLD_SIZE` 后
以单进程身份直接运行（不再自行 fork）。等价写法二选一：

```bash
torchrun --nproc_per_node=2 -m app.main --fname <CONFIG.yaml>            # 用当前可见 GPU 前 2 张
torchrun --nproc_per_node=2 -m app.main --fname <CONFIG.yaml> --devices cuda:0 cuda:1  # 显式挑卡
```

单进程调试（`--debugmode True`，便于打断点，与生产同一套入口）：

```bash
python -m app.main --fname CONFIGS/archive/test.yaml          --devices cuda:0 --debugmode True
python -m app.main --fname CONFIGS/archive/test-finetune.yaml --devices cuda:0 --debugmode True
```

- `test.yaml` → 预训练调试（读 `CKPT/vjepa2/vitl.pt`，输出到 `OUTPUT/test`）。
- `test-finetune.yaml` → 微调调试（读预训练 `latest.pt`，跑 RAVDESS emotion，输出到 `OUTPUT/test_finetune`）。

### 10.1 用 `--set` 覆盖配置（无需改 YAML）

只改一两个键时（最典型的是 `folder` 与 `meta.read_checkpoint`），不必新建 YAML 变体：
`app/main.py` 的 `--set` 在 **YAML 合并之后**对合并结果做覆盖，因此优先级高于所有片段
**和**入口文件本身。

```bash
# 同一个入口跑两组对比：只换输出目录与预训练 ckpt
torchrun --nproc_per_node=2 -m app.main \
  --fname CONFIGS/tasks/finetune/video/classification/RAVDESS-emotion.yaml --devices cuda:0 cuda:1 \
  --set folder=OUTPUT/finetune_v/vitl16/RAVDESS-emotion/e5 \
        meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-48f/e5.pt

torchrun --nproc_per_node=2 -m app.main \
  --fname CONFIGS/tasks/finetune/video/classification/RAVDESS-emotion.yaml --devices cuda:0 cuda:1 \
  --set folder=OUTPUT/finetune_v/vitl16/RAVDESS-emotion/e11 \
        meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-48f/e11.pt
```

规则：

| 项 | 行为 |
|---|---|
| 键写法 | 点号索引嵌套映射：`meta.read_checkpoint`、`optimization.epochs`、`data.batch_size` |
| 值解析 | 先按 YAML 标量解析，`false` → bool、`12` → int、`0.5` → float、`[a, b]` → list；解析结果不是标量/容器的（如路径）保留为字符串 |
| 优先级 | 最高——覆盖 `yamls` 片段与入口 YAML 里已写的值 |
| 生效范围 | 预训练 / 微调共用 `app/main.py`，因此两边都支持；`app=` 也能改，且会真的切换分发模块 |
| 启动方式 | `torchrun`、`python -m app.main --debugmode True`、手写 `mp.Process` 三条路径都透传 |
| 位置 | 建议放最后。实测放中间也能解析——argparse 遇到下一个 `--flag` 会终止值列表；唯一歧义是值本身以 `-` 开头且不是负数 |
| 键路径写错 | 中间层不存在或不是映射 → 启动即抛 `KeyError` / `ValueError`，不会静默忽略 |
| 可写类型 | 字符串 / int / float / bool / list / dict（整块覆盖也算，见下方第 3 点） |

### 10.2 `--set` 传不了什么

`--set` 作用于**合并后的参数字典**，所以「配置里存在的键」都能覆盖；下面这些是边界：

1. **`yamls` 引用键不能改**。`load_config` 在返回前已经 `pop` 掉 `yamls`，因此
   `--set yamls.sampling=/x.yaml` 会抛 `KeyError: yamls is not a config mapping`。
   想换采样就**直接覆盖采样键本身**（已实测可覆盖预设里的值）：

   ```bash
   # 等价于把入口的 sampling: 从 48frames-8fps.yaml 换成 64frames-16fps.yaml
   --set data.fps=16 data.dataset_fpcs='[64]'
   ```

   代价是绕过了 `sampling:` 这层抽象——§8.4 那套「消融 = 换预设」的记账方式在此失效，
   要在 `folder` 命名或实验记录里自己体现，否则回头对不上。

2. **没有 schema 校验：改一个不存在的叶子键会静默无效**。`--set yamls=/x.yaml`、
   `--set meta.seed_=7` 都不报错，只是往配置里塞了个没人读的键。
   所以：**覆盖已存在的键 = 一定生效；发明新键 = 可能什么都不发生**。
   尤其注意 §12.4 那个坑——`--set meta.use_sdpa=true` 会静默无效（该键读的是 `model`），
   而 `--set model.use_sdpa=true` 才有效。

3. **嵌套块不存在时不能逐键创建**。配置里没有 `meta.rankme` 时
   `--set meta.rankme.n_hutchinson=3` 抛 `KeyError`。变通是**一次塞整块**：

   ```bash
   --set meta.rankme='{enabled: true, every_events: 2}'
   ```

4. **列表不能按下标改**。`--set mask.0.num_blocks=2`、
   `--set data.datasets_weights.0=0.5` 都会抛 `KeyError`（报错明确，不会误改）。
   只能**整条替换**：`--set mask='[{num_blocks: 1, ...}, {...}]'`。

5. **纯数字 / `true` / `false` 值会被强制转类型，且无法强制回字符串**。
   `--set folder=123` 得到的是 int `123`（`Path(123)` 会报错），
   给值加引号也没用——`--set "folder='/123'"` 里的引号会成为字符串内容的一部分。
   实际影响极小：路径类值（`folder` / `read_checkpoint` / `label_column`）都不会是纯数字。
   值里带 `=` 和空格没问题（只按第一个 `=` 切分，shell 引号不会进入值）。

6. **不在配置里的东西管不着**：`--fname` / `--devices` / `--debugmode`（argparse 自己的
   参数）、torchrun 的 `--nproc_per_node` / `--master_port`、环境变量 `CUDA_VISIBLE_DEVICES`；
   以及代码里硬编码的行为（如 `app/finetune_v/train.py` 的 `torch.device("cuda:0")`、
   `drop_last=False`、`higher_is_better` 由 `data.task` 推导）。

7. **架构相关的覆盖要自己保证与 ckpt 兼容**：`--set model.model_name=vit_huge` 能设进去，
   但与 `read_checkpoint` 的 backbone 形状不匹配会在加载时报错；
   `--set data.num_class=...` 与 CSV 实际标签不符会在 `train.py` 的越界校验处显式报错。

> 一句话：**改已存在的键 = 可靠；改引用机制（`yamls`）、发明新键、动列表元素 = 别用 `--set`。**

### 10.3 两条通用注意

1. **override 会写进参数快照**。`{folder}/params-{app}.yaml` 是在覆盖之后 dump 的，
   所以快照与 `best.pt` / `latest.pt` 里存的 `args` 都忠实反映命令行实际生效的值
   （见 `app/main.py::process_main`），复现不受影响。
2. **换 `folder` 就等于换实验**。`latest.pt` 是按 `folder` 找的，改 `folder` 后不会续跑
   原目录的进度，而是从 `read_checkpoint` 重新开始（`meta.load_checkpoint` 仍为 `true` 时
   只会找新目录下的 `latest.pt`，找不到就当新实验）。想在原目录续跑就不要覆盖 `folder`。

## 11. 其他文件

- **`split_root_paths.csv`** — 数据集清单 TSV：`split_csv, root_path, original_csv_path, duration_mean, duration_median, duration_std, duration_var, duration_min, duration_max, sampled_mean_fps`。
  记录每个数据集 CSV 对应的媒体根目录（供 `merge_csv.py` 解析相对路径）及各时长统计列。
- **`bk/`** — 历史备份：
  - `bk/pretrain_v/vit{g,h,l}16/*.yaml`：早期 V-JEPA 标准分辨率（256px/384px）与 FaVoR 112px 的预训练/退火配置。
  - `bk/finetune_v/fintune-RAVDESS-*.yaml`：早期 RAVDESS 微调配置（旧字段名，如 `csv_path`/`freeze_backbone`，已废弃）。

## 12. 已知问题与注意事项

1. **音频 backbone 与 checkpoint 来源** —— 仅支持
   `model.backbone.mode: emotion2vec`，它包含相对位置卷积、extra tokens、4 层 modality
   context encoder、ALiBi 和 8 层 global blocks。`IEMOCAP-emotion.yaml` 与新的
   `pretrain_a` 均采用此路径；旧简化 AudioJEPA backbone checkpoint 已不再兼容。

2. **音频解码依赖** —— 优先使用 `torchaudio`，当当前 PyTorch/Torchaudio 组合要求但未安装
   TorchCodec 时会回退到 `soundfile`；对本地 libsndfile 不支持的 WebM / MP3 等格式，最后回退到 `ffmpeg`。

3. **`frozen_encoder` 默认解冻** —— 33 个音频微调任务入口都显式设为
   `frozen_encoder: false`（解冻 encoder、端到端训练）。

4. **`use_sdpa` 位置** —— 必须写在 `model` 片段下（`app/pretrain_v/train.py` 从
   `cfgs_model` 读取）；早期 `bk/` 配置曾误写到 `meta` 下，会被忽略并回退到默认 `false`。

5. **`yamls` 路径** —— 片段路径可用绝对路径，也可用相对入口文件的相对路径（`load_config`
   会自动相对入口目录解析）。**移动 / 重构 `data/` 或 `tasks/` 下的文件后，记得同步更新
   引用方**：`CONFIGS/tasks/**` 各入口的 `yamls`、`CONFIGS/test*.yaml`、以及 `run.sh` 的
   `--fname`。仓库没有覆盖这些引用的校验，路径写错只会在启动时以 `FileNotFoundError` 暴露。

6. **音频下游特征与 loss** —— `model.features.mode` 支持 `last` 和 `topk_average`，后者由
   `model.features.topk_layers` 控制层数；完整 emotion2vec 配置默认 `last`。微调 loss 统一放在
   `optimization.loss`：分类默认 `cross_entropy + class_weighting: none`，可选
   `inverse_frequency`、`sqrt_inverse_frequency`、`effective_number`；多标签支持是否启用
   `positive_weighting: inverse_frequency`；回归支持 `mse`、`mae`、`smooth_l1`。

7. **音频微调 dropout 与 Grokfast** —— dropout 可分别配置在
   `model.extractor.dropout`、`model.backbone.prenet_dropout`、
   `model.encoder.{dropout_input,dropout,attention_dropout,activation_dropout,post_mlp_dropout}`、
   `model.head.{attention_dropout,dropout}`；未配置项沿用模型片段中的值，新增的 head attention
   dropout 默认 `0.0`。`optimization.grokfast` 默认 `false`，开启时复用视频侧相同的
   `grokfast_alpha`（默认 `0.98`）和 `grokfast_lambda`（默认 `2.0`）。

VA 预训练默认允许时长差 0.5 秒（或两帧时长，取较大值），起点及声明的 AV offset
最多 0.1 秒；通过检查后按实际共同时间区间裁剪音视频，不拉伸或重复视频帧。
`data.sync.duration_tolerance_seconds`、`duration_tolerance_frames` 和
`offset_tolerance_seconds` 可调整这些限制；两帧容差设为 0 可使用固定秒数限制。
超过容差、没有共同区间或解码损坏的 pair 仍按 `on_decode_error` 处理。
控制台仅打印 `loss inv sigreg std rank lr wd grad clips/s`；多分支时 inv、sigreg、
std、rank 取分支均值，lr / wd 取参数组最大值。详细分支指标和耗时仍写入 history，
其中 `step_seconds` 为一次 optimizer step 的耗时，包含数据加载及梯度累积。
skip 计数是本次启动以来的累计值，写入 data_skips.csv。
