# FAVOR

**多模态语音 / 视频情感与心理数据集** —— 视频 V-JEPA 与音频 AEmo-JEPA 预训练 + 下游微调。

FAVOR（Facial / Audio / Video emotion & mental-health understanding）是一套面向
**语音、视频、音视频联合**的情感与心理健康理解训练框架。视频侧以 V-JEPA 对无标注
人脸视频做自监督预训练；音频侧以 emotion2vec_plus_large 初始化 AEmo-JEPA 并继续
预训练。两种模态都通过统一配置入口完成分类、回归和多标签下游微调。

> 本仓库 fork 自 [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
> （V-JEPA 2.1），保留了其骨干模型与预训练管线（MIT 许可），并在此之上重构了
> 配置系统、数据管线与下游微调流程。上游为 [MILab-Mental/FaVoR](https://github.com/MILab-Mental/FaVoR)。

## 目录结构

```
FAVOR/
├── app/
│   ├── main.py            # 统一启动器：读 YAML → 每 GPU 一个进程 → 按 app 字段分发
│   ├── pretrain_v/        # 视频 V-JEPA 自监督预训练
│   ├── finetune_v/        # 视频下游微调
│   ├── pretrain_a/        # 音频 AEmo-JEPA 自监督预训练
│   └── finetune_a/        # 音频分类 / 回归 / 多标签微调
├── CONFIGS/               # 全部训练配置（任务入口 + 片段），详见 CONFIGS/README.md
├── datasets/              # 视频与音频 dataset / dataloader / transforms
├── models/                # V-JEPA、AEmo-JEPA 骨干与任务头
├── optimization/          # optimizer / 调度器（warmup-cosine / wd-schedule / anneal）
├── utils/                 # 分布式 / 日志 / 指标 / checkpoint 加载
├── DATASET/               # 数据清单、split、合并与统计脚本（不纳入版本控制）
├── CKPT/                  # V-JEPA 2.1 官方预训练权重（不纳入版本控制）
├── OUTPUT/                # 训练产物：日志 / checkpoint / 参数快照（不纳入版本控制）
├── PLOT/                  # 论文统计、制表与可视化脚本，详见 PLOT/README.md
├── run.sh                 # 15 个微调任务的一键运行脚本（含 vjepaori 基线 15 条）
├── train.sh               # 元数据生成训练命令：dry-run 校验 / N 路并发 / 日志汇总
├── clean.py               # 递归清理 __pycache__ / .ipynb_checkpoints
└── requirements.txt
```

## 快速开始

```bash
# 调试（单进程，便于打断点）
python -m app.main --fname CONFIGS/test.yaml          --devices cuda:0 --debugmode True
python -m app.main --fname CONFIGS/test-finetune.yaml --devices cuda:0 --debugmode True

# 多卡训练统一用 torchrun 启动。`app/main.py` 检测到 torchrun 注入的 RANK/WORLD_SIZE
# 后以单进程身份直接运行（不再自行 fork）。等价写法二选一：
#   torchrun --nproc_per_node=2 -m app.main --fname <CONFIG.yaml>            # 用可见 GPU 前 N 张
#   torchrun --nproc_per_node=2 -m app.main --fname <CONFIG.yaml> --devices cuda:0 cuda:1  # 显式挑卡
# 旧的 `python -m app.main ... --devices` 多卡写法仍兼容，但新任务请用 torchrun。

# 预训练（48 帧 / 112px，vit_large）
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vpretrain/pretrain_v_FaVoR-112px-48f.yaml --devices cuda:0 cuda:1

# 音频 AEmo-JEPA 预训练（emotion2vec_plus_large 初始化，2～4 秒变长 crop）
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/apretrain/pretrain_a_FaVoR.yaml

# 音频微调（例：CASIA emotion 6 类；33 个任务均使用同一入口模式）
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/afinetune/cls/CASIA-emotion.yaml

# 退火 / cooldown（长片段 64f，LR 退到 ~0）
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vpretrain/pretrain_v_FaVoR-cooldown.yaml

# 微调（例：RAVDESS emotion 8 类）
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-emotion.yaml

# 不改 YAML，用 --set 覆盖任意配置项（点号表示嵌套键，值按 YAML 标量解析）
# 覆盖在 YAML 合并之后生效，优先级最高；会一并写进 {folder}/params-{app}.yaml 快照。
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-emotion.yaml \
  --set folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/RAVDESS-emotion/e5-vs-e11 \
        meta.read_checkpoint=/home/data/sdc/FAVOR/OUTPUT/pretrain_v/vitl16/FaVoR-112px-48f/e11.pt \
        meta.seed=7

# 一次性跑完 run.sh 里列的 15 个视频微调任务（内部已改为 torchrun）
# 注意：tasks.json 共 33 个语义任务；run.sh 当前只编排其中 15 条视频任务，全部
# 33 条任务的音频版本可从 CONFIGS/tasks/afinetune/ 单独启动。
bash run.sh

# train.sh 从实验元数据 + 15 条任务元数据自动生成 torchrun 命令；先预览再运行
bash train.sh --dry-run  # 只校验元数据并打印命令，不启动训练、不创建日志
bash train.sh          # 串行（1 个槽位，最安全）
bash train.sh 2        # 2 个任务同时跑（槽位轮流从列表领取，见脚本头部说明）
# 单任务失败只记录不中断，跑完打印汇总并以非 0 退出；日志在 OUTPUT/train_sh_logs/<时间戳>/
# 切换 checkpoint / scratch / 官方 V-JEPA 时只改 train.sh 的 EXPERIMENT_META；
# 增删任务只改 TASK_META，不再复制整条命令。EXTRA_SET 可添加所有任务共享的 --set 覆盖。

```

所有命令都通过 `--fname` 指向 `CONFIGS/` 下的任务 YAML，无需改动代码即可换数据 /
换模型 / 换超参。**只改一两个键（如换 `folder` / 换预训练 ckpt / 换 seed）时不必新建
YAML 变体**，用 `--set 键=值` 覆盖即可（点号索引嵌套键；值按 YAML 标量解析，
`false` / `12` 会得到 bool / int）；`--set` 建议放在命令行最后。
边界（`yamls` 引用键改不了、新键会被静默忽略、列表只能整条替换）见
[`CONFIGS/README.md`](CONFIGS/README.md) 10.2。

## 模型与训练流程

### 预训练（`app/pretrain_v/`）

V-JEPA 自监督：对视频帧做 **multiblock 3D 随机掩码**，context encoder（ViT）只看到
可见部分并预测被掩码位置的表征，与一个 **EMA 更新的 target encoder** 输出对齐：

```
loss = |z_context − h_target|^p / p     (p = loss_exp)
```

- 网络：`vit_large` 等骨干 + 独立 predictor（`vit_predictor`），target encoder 由
  encoder 动量更新（`ema 0.99925 → 0.9998`）。
- 掩码：两组块（8 个小块 + 2 个大块），支持多数据集多帧率（`dataset_fpcs`）各自配掩码。
- 优化：AdamW + warmup-cosine LR + cosine weight decay schedule；`bfloat16` 混合精度。
- 支持从任意 checkpoint 续训、`reset_epoch`、以及退火（anneal）阶段。

### 微调（`app/finetune_v/`）

从预训练 checkpoint **只加载 backbone**（`load_backbone` 自动剥 `module.` / `backbone.` /
`model.` 前缀），接一个**全新初始化的 attentive-pooler 任务头**：

- `ClipEncoder`：取 backbone 多个 transformer block 的输出（`out_layers`）做多层级聚合，
  多 clip 沿时间拼成 token 序列。
- 任务头：`AttentiveClassifier`（分类） / `AttentiveRegressor`（回归）。
- 分类用类别加权交叉熵 + `f1_macro` 选 best；回归用 MSE，按 `val_loss` 选 best。
- 支持 `frozen_encoder`（冻结编码器只训头，当前 15 个任务已统一解冻做端到端训练）。
- 分布式评测：多卡按样本分区、`all_gather` 汇总，输出混淆矩阵 / 预测 CSV / 指标曲线。

## 已覆盖数据集

7 个视频数据集，15 个微调任务（分类 + 回归）：

| 数据集 | 任务 |
|---|---|
| RAVDESS | emotion（8 类）/ intensity（2 类） |
| CREMA-D | emotion（6 类）/ intensity（回归） |
| IEMOCAP | emotion（9 类）/ valence / activation / dominance（回归） |
| EmotionTalk | emotion（7 类） |
| MER2023 | emotion（6 类）/ pos_intensity（回归） |
| MER242526 | emotion（6 类）/ pos_intensity（回归）/ 26openset（多标签 23 类） |
| AVEC2014 | PHQ 抑郁评分（回归） |

完整的任务 ↔ 配置 ↔ 标签列映射见 [`CONFIGS/README.md`](CONFIGS/README.md)。

## 配置系统

采用 **「任务入口 + 片段合并」** 的 YAML 配置模式：入口文件通过 `yamls:` 字段按序合并
`datas/`、`opt/`、`models/`、`mask_loss/` 下的片段，入口自身字段优先级最高。字段说明、
合并规则、以及 15 个微调任务总览均见 [`CONFIGS/README.md`](CONFIGS/README.md)。

## Checkpoint

- 预训练初始化权重：`CKPT/vjepa2/vitl.pt`（V-JEPA 2.1 官方 `vit_large`）。
- 预训练产物：`OUTPUT/pretrain_v/.../latest.pt`（及 `e{epoch}.pt`），微调入口的
  `read_checkpoint` 指向这里。
- 微调产物：`OUTPUT/finetune_v/.../best.pt` / `latest.pt` + `logs/`（指标曲线、混淆矩阵、
  最佳预测 CSV）。

---

## 设计哲学

### 1. 单一入口，配置即实验

所有训练共享一个启动器 `app/main.py`，实验差异**全部落在 YAML** 而不是代码里。加一个
新实验 = 写一个新的任务 YAML + 复用片段，不改任何 Python。配置片段（数据 / 优化器 /
模型 / 掩码）可自由组合复用，天然 DRY；每次启动自动把合并后的完整参数快照写入
`{folder}/params-{app}.yaml`，保证实验可复现、可对比。

### 2. 预训练与微调解耦，骨干可替换

预训练产出的只是一份「通用视频表征骨干」，微调阶段通过**前缀剥离的兼容加载器**只取
backbone 权重，任务头永远随机初始化。因此换预训练 checkpoint（vit_l / vit_h / vit_g、
甚至外部模型）只改一个路径，下游代码零改动。这层解耦让「预训练怎么训」和「下游怎么用」
两个问题可以独立演化。

### 3. 面向多数据集、多任务、多帧率的统一抽象

`VideoDataset` / `VideoCSVDataset` 一个类同时承载多数据源（各自不同的帧率 `dataset_fpcs`
、采样权重、根目录），掩码 collator 也按数据集维度广播；微调侧用同一套
`classification / regression` 接口覆盖 15 个异构任务（情绪分类、valence/arousal 回归、
抑郁评分回归）。**新增数据集或任务类型只需增配置 + 最小代码**，而非复制训练脚本。

### 4. 容错与显式失败并重

面向大规模、脏数据、长时训练的工程现实：

- **容错**：checkpoint 加载带指数退避重试（`robust_checkpoint_loader`）；视频解码失败
  自动重采样；加载 checkpoint 时形状不匹配的键自动跳过并告警；dataloader 用尽自动刷新。
- **显式失败**：label 越界、空 dataloader、多标签缺 `num_class` / 类别索引越界等**直接抛错**并给出可读信息，
  绝不静默跑出一个错误结果；预训练对 NaN/Inf loss 立即断言退出。

原则：**能自动恢复的自动恢复，不能恢复的尽早大声失败。**

### 5. 分布式为先，单进程可调试

多卡训练默认 DDP，启动方式统一为 `torchrun --nproc_per_node`（每 rank 一进程、独占一张
GPU；rank 崩溃会立即报错并终止整组，不用再干等 NCCL 默认 600s 超时）。`app/main.py` 同时
保留手写 `mp.Process` 本地启动与 `--debugmode True` 单进程调试两条路径；SLURM 与非 SLURM
的 rendezvous 均兼容、端口自动选空避免并发冲突。
微调评测在多卡下按样本分区、`all_gather` 汇总，保证指标在任意卡数下一致。

### 6. 保留上游血统，渐进式改造

骨干模型、attentive-pooler、调度器等直接继承 V-JEPA 2.1（保留 MIT 版权头），保证与官方
预训练权重无缝对接；同时在其之上**重构**了配置系统、数据管线与微调流程，并用 `bk/`
目录保留历史配置、用注释显式标记「未实现 / 占位」项，让每一步演进都留下痕迹。

### 7. 模态统一于「情感与心理」这一目标

`root.txt` 用统一清单管理 30+ 数据集的音频 / 视频 / 音视频三种预训练用途（`v/a/va`），
视频与音频共享同一套「配置片段 + 任务入口」范式，但数据加载、采样和模型实现按
模态分开。当前视频侧覆盖 15 条任务，音频侧已覆盖 `tasks.json` 的全部 33 条任务；
音视频联合（VA）仍保留为后续扩展。

---

## 音频训练状态

- `app/pretrain_a` 已接入 AEmo-JEPA：以 emotion2vec_plus_large 初始化 CNN 与 context
  encoder，使用独立 predictor 和 EMA target encoder 做音频 JEPA 继续预训练。默认入口为
  `CONFIGS/tasks/apretrain/pretrain_a_FaVoR.yaml`。
- `app/finetune_a` 已支持 `classification`、`regression` 和
  `multi_label_classification`；tasks.json 中全部 33 条任务的音频版本位于
  `CONFIGS/tasks/afinetune/{cls,reg,mlcls}/`。默认兼容读取原 AEmo-JEPA 的
  `checkpoint["model"]`，新 `pretrain_a` checkpoint 则直接读取显式的 `encoder`。
- 音视频联合（VA）训练仍未实现。

## 许可

骨干模型与预训练代码沿用上游 V-JEPA 2.1 的 MIT 许可；新增微调 / 配置 / 数据代码为项目自有。
