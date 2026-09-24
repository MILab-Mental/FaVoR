# 论文图表与统计脚本

本目录集中存放论文结果汇总、统计检验、LaTeX 制表和可视化脚本。脚本默认从仓库根目录的
`OUTPUT/` 读取训练结果，并把新生成的文件写入 `PLOT/output/`。`PLOT/output/` 是可重复
生成的产物目录，已加入 `.gitignore`。

## 目录

```text
PLOT/
├── ana.py             # 单任务内多配置的 bootstrap 分布、显著性检验和 ROC
├── ana_bench.py       # FaVoR 与 V-JEPA 2.1 的跨任务比较图和统计表
├── 增加radar.py       # 按任务主指标归一化的多配置平滑雷达图（SVG）
├── stat.py            # 从 logs/history.csv 汇总每个实验的最佳指标
├── latex_table.py     # 生成分类、多标签和回归 LaTeX 表格
├── pca.py             # 将视频 patch-token 的前三个 PCA 分量渲染为 RGB 视频
├── plot_train.py      # 汇总各 rank CSV 日志并绘制训练 loss 曲线
├── color.md           # 论文图表统一色卡与颜色语义约定
└── output/            # 默认输出目录，不纳入版本控制
    ├── ana/
    ├── bench/
    ├── stat/
    ├── pca/
    └── train/
```

以下命令均建议从仓库根目录执行。

所有论文图表应遵循 [`color.md`](color.md) 的统一色卡和语义映射；同一模型或方法在不同
图中保持同色，避免额外引入随机色相。

## 最佳指标汇总

`stat.py` 从匹配实验目录的 `logs/history.csv` 选择最佳 epoch，在终端打印汇总，并默认写入
`PLOT/output/stat/summary.csv`：

```bash
python PLOT/stat.py 'OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-e5'

# 按实验名排序、显示 checkpoint/seed/frames/epochs/lr，并指定 CSV 路径
python PLOT/stat.py 'OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-*' \
  --sort name --config --csv PLOT/output/stat/all.csv
```

## LaTeX 指标表

`latex_table.py` 复用 `stat.py` 的任务类型推断和最佳 epoch 选择口径。默认比较 FaVoR e5 与
V-JEPA 2.1 基线，把分类、多标签分类和回归排进同一个 `tabular`，写入
`PLOT/output/stat/tables/`：

```bash
python PLOT/latex_table.py --preview

# 改用其他实验通配符或输出目录
python PLOT/latex_table.py \
  --favor 'OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-e11' \
  --out PLOT/output/stat/e11-tables
```

输出包括 `table_all_tasks.tex` 和包含所需宏包的 `preamble.tex`。总表只有一个 `table*`、
一个 `tabular`、一个 caption 和一个 label；三种任务用分组标题行和各自的指标表头区分。
多标签与回归指标通过 `\multicolumn` 均匀铺满表宽，不会在右侧留下成片空列。

## 单任务配置分析

`ana.py` 比较同一任务下的多个配置，输出 bootstrap 分布图、相对主配置的配对显著性检验
及 micro ROC。任务路径默认相对于根目录 `OUTPUT/`：

```bash
python PLOT/ana.py \
  --tasks finetune_v/vitl16/RAVDESS-emotion \
  --main_result FaVoR-112px-48f-8fps-e5 \
  --n-bootstrap 1000
```

结果写入 `PLOT/output/ana/<task-name>/`。`--tasks` 可同时给多个任务；此时
`--main_result` 可给一个共用配置名，或按任务顺序提供同样数量的配置名。

## 跨任务基准图

`ana_bench.py` 复用 `ana.py` 的分类指标、配色和显著性定义，并复用 `stat.py` 的实验汇总
逻辑。默认比较全部可用任务，写入 `PLOT/output/bench/`：

```bash
python PLOT/ana_bench.py --n-bootstrap 1000 --per-task

# 只处理指定任务
python PLOT/ana_bench.py --tasks RAVDESS-emotion MER2023-emotion

# 只从已有缓存重画，不重算指标、bootstrap、显著性或 ROC
python PLOT/ana_bench.py --redraw-only --per-task
```

主要产物包括柱状图、箱线图、小提琴图、逐任务分面图、ROC、`significance.csv` 和
`metrics_bootstrap.csv`。可用 `--out` 改变输出目录。完整运行还会写入
`bootstrap_samples.npz`，之后 `--redraw-only` 可从该缓存精确重画三种图。如果是旧结果、
尚无 NPZ 缓存，则会利用 CSV 中的均值和标准差精确重画柱状图，并保留现有的箱线图、
小提琴图和 ROC。`--plot-only` 是 `--redraw-only` 的别名。

## 跨任务雷达图

`增加radar.py` 以任务为轴，默认选取分类/多标签分类的 `val_f1_macro` 和回归的
`val_rmse`。每根轴在给定配置内单独归一化，最优配置落在 100% 外圈；图中数字仍是原始
指标值。配色严格按 `color.md` 的 Primary palette 顺序对应 `--configs` 顺序。
输出 SVG 无图内标题、使用 Arial 字体，并按所有可见元素的实际包围盒自动裁切，四周不额外留白；
100% 最佳效果圆同时是雷达图的唯一外边界，图例位于右上角，可直接用于论文排版。

```bash
python PLOT/增加radar.py \
  --task OUTPUT/finetune_v/vitl16 \
  --configs FaVoR-112px-48f-8fps-e5 FaVoR-112px-48f-8fps-vjepaori \
  --labels FaVoR "V-JEPA 2.1"
```

默认输出 `PLOT/output/radar/radar.svg`。可用 `--tasks` 限定任务并指定雷达轴顺序，用
`--classification-metric`、`--multilabel-metric` 和 `--regression-metric` 替换三类主指标。

## PCA 视频可视化

`pca.py` 对视频滑窗提取 patch token，经 PCA 后把前三个分量映射为 RGB 视频。默认写入
`PLOT/output/pca/2_RAVDESS_SV_v01.mp4`：

```bash
python PLOT/pca.py \
  --input DATASET/splits-0901/demo_video/2_RAVDESS_SV/v01.mp4 \
  --ckpt OUTPUT/pretrain_v/vitl16/FaVoR-112px-48f/latest.pt

# 自定义输出文件
python PLOT/pca.py --input path/to/video.mp4 --ckpt path/to/checkpoint.pt \
  --output PLOT/output/pca/example.mp4
```

## 训练 loss 曲线

`plot_train.py` 会自动识别两种训练日志：

- 原有 V-JEPA 多 rank CSV：按 `(epoch, iteration)` 聚合 loss，绘制原始值和移动平均；
- VIDEO-LeJEPA `history.csv`：绘制 loss、SIGReg、表征统计、学习率和梯度范数四面板曲线。

默认写入 `PLOT/output/train/<日志目录名>/train_loss.png`：

```bash
python PLOT/plot_train.py \
  --logdir OUTPUT/pretrain_v/vitl16/FaVoR-112px-48f/

# VIDEO-LeJEPA：自动读取目录根部的 history.csv 和参数快照中的 SIGReg 权重
python PLOT/plot_train.py \
  --logdir OUTPUT/pretrain_video_lejepa/vitl16/FaVoR-224px-16f/

# 调整平滑窗口或指定输出文件
python PLOT/plot_train.py --logdir path/to/logs --window 50 \
  --out PLOT/output/train/custom.png

# 可选：覆盖参数快照中的 SIGReg 权重
python PLOT/plot_train.py --logdir path/to/lejepa/output \
  --sigreg-weight 0.02
```

## 输入与输出约定

- `OUTPUT/` 保存训练 checkpoint、日志、逐样本预测和其他实验原始产物；绘图脚本只读取它。
- `PLOT/output/` 保存统计汇总、论文图表、LaTeX 表格和可视化视频；脚本会按需创建目录。
- `--out`、`--output` 或 `--csv` 可覆盖默认输出位置。
- `ana_bench.py` 和 `latex_table.py` 的默认实验通配符基于
  `OUTPUT/finetune_v/vitl16/`；若实验命名变化，请显式传入 `--favor` 和 `--vjepaori`。
