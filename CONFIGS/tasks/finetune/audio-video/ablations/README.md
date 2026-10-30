# FaVoR 内部对照

内部消融使用根项目的 `app.finetune_va`、原生数据管线和任务头，不进入 `BASELINES/` 的外部模型清单。

| 配置 | 视频 / 音频初始化 | 跳过的研究预训练 |
|---|---|---|
| `no-joint-pretraining.yaml` | FaVoR-V / FaVoR-A 已分别预训练的编码器，与当前联合预训练对应项一致 | 仅跳过联合音视频预训练 |
| `upstream-backbones.yaml` | `CKPT/vjepa2/vitl.pt` / `CKPT/emotion2vec_plus_large/model.pt` | 同时跳过 FaVoR 单模态和联合预训练 |

两份配置都设置 `finetune.pretrained_checkpoint: null`，两骨干、融合层和任务头均参与监督训练。融合层和任务头随机初始化。原生加载器把上游视频时间卷积从 tubelet=2 转为 1，并新增 CLS；这两项适配会在加载日志中报告。

配置以 RAVDESS 为最小入口，沿用根项目当前训练和选点流程。它们仅区分初始化条件；根项目当前 `finetune_va` 仍将 CSV split=1 用作验证选点，不能直接把该 split 的结果当作独立测试结果。用于正式外部比较前，需要统一根项目的验证/测试选点协议；不在这里重新划分 CSV。

```bash
torchrun --standalone --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/finetune/audio-video/ablations/no-joint-pretraining.yaml
torchrun --standalone --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/finetune/audio-video/ablations/upstream-backbones.yaml
```

mental 等长媒体任务使用根项目的 8 句实例采样与注意力任务头。切换任务时沿用对应任务的数据、采样和实例头配置，保持以上初始化条件与独立输出目录。

FaVoR-A/V 晚期概率融合位于根项目 `utils/late_fusion.py`，读取同一 CSV 划分下两个模态完整且匹配的预测。存在 split=2 时只用该验证集选择融合权重；没有验证集时固定音频权重（默认 0.5），测试集不选权重。

```bash
python -B -m utils.late_fusion --data-config CONFIGS/data/finetune/audio-video/mental/DSM-Depression.yaml --audio-test /音频预测.csv --video-test /视频预测.csv --test
```

去掉 `--test` 后默认输出到 `OUTPUT/ablations/favor-late-fusion/<experiment>/`，可用 `--output` 指定目录；仅保存配置和指标，`--save-predictions` 才导出融合预测。当前已用历史 mental 预测验证融合流程；历史源模型的选点协议不满足独立测试要求，其结果不作为正式对照。
