# FaVoR 表征指标

所有代码、结果、缓存和历史实验都在 `METRIC/` 内；不修改项目训练代码。

## 目录

- `main.py`、`config.yaml`：通用评估入口及配置；`metrics.py`、`extra_metrics.py`：指标实现。
- `scripts/`：当前 20% 音视频实验的采样、特征、统计和画图脚本。
- `results/sample20/`：当前结果、原始数据和图，具体见其中的 README。
- `results/finetune-time-estimate.csv`：53 个任务的微调耗时粗估。
- `cache/sample20/`：固定抽样清单、特征、分任务中间结果和运行状态。
- `cache/early-stop/`：目前汇总使用的微调结果队列和音视频配置。
- `logs/`：当前实验日志及用于核对三任务微调的日志。
- `archive/`：历史代码、试验结果和缓存；`archive/moves.csv` 记录移动前后路径。

## 当前实验口径

8 个音视频数据集：从训练集与验证集合并后抽取 20%，排除已明确标记的官方测试记录；抽中 25,775 条，17 个权重共同有效 25,752 条。特征按单句、音视频 6 秒提取，冻结 Encoder，供指标复用。53 个任务中有 38 个临床任务，因此任务等权偏重临床数据；结果同时提供数据集等权列。

RankMe 对同一轨迹 16 个权重按任务等权汇总，选择 **step-60000**；Top-1 Singular Energy 越低越好，实际值归一化后任务等权选择 **step-35000**，数据集等权选择 **step-25000**。对应权重未参加下面三任务的五权重微调对照，不应将指标选点当成微调结论。

相关性只使用已有 DSM、RAVDESS、IEMOCAP 的五权重真实微调验证结果。Spearman 基于排名；Pearson 基于实际值，分别提供单任务和八数据集综合值口径。由于抽样包含微调验证集、仅有五个对照权重，相关性是事后观察，不是独立验证。Probe 使用五折折外预测，临床数据按受试者分折，其余按样本分折；采样清单里的旧 `_probe_split` 列不用于本次计算。

## 常用命令

在项目根目录运行，使用现有 Python 环境和项目依赖：

```bash
python -B METRIC/main.py --help                    # 通用入口与参数
python -B METRIC/scripts/summary20.py              # 用缓存重建五权重汇总
python -B METRIC/scripts/correlation_values.py     # 重建 Pearson 相关性表
python -B METRIC/scripts/energy_all.py             # 用缓存重算全部权重 Top-1 Energy
MPLCONFIGDIR=METRIC/cache/matplotlib python -B METRIC/scripts/plot_rankme_trajectory.py
```

采样和 GPU 特征提取脚本分别为 `scripts/sample20.py` 和 `scripts/features20.py`。重新采样会改变样本清单并使已有特征缓存不再匹配；需要新实验时应单独保存原清单和缓存。历史实验中的原始路径可能指向归档前的位置，参见 `archive/README.md`。
