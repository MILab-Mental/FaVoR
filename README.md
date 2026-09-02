# FAVOR

多模态语音/视频情感与心理数据集 —— 视频 V-JEPA 预训练 + 下游微调。

## 目录结构

- `app/`
    - `main.py` — 统一启动器（按任务入口 YAML 的 `app` 字段分发到对应 train 模块）
    - `pretrain_v/` — 视频 V-JEPA 预训练
    - `finetune_v/` — 视频下游微调（分类 / 回归）
    - `preprocess/` — 数据预处理脚本
- `CONFIGS/`
    - `datas/` — 数据配置（`vpretrain-data*.yaml`、`vfinetune-*.yaml`）
    - `opt/` — 优化器/调度配置（`vpretrain-opt*.yaml`、`vfinetune-opt*.yaml`）
    - `models/` — 模型配置（`vpretrain-vit-*.yaml`、`vfinetune-vit-*.yaml`）
    - `mask_loss/` — 掩码与损失配置（`vjepa-pretrain.yaml`）
    - `tasks/` — 任务入口配置（`pretrain_v_*.yaml`、`finetune_v_*.yaml`）
    - `bk/` — 历史 / 参考配置备份
- `datasets/` — 数据集、dataloader、掩码、视频变换
- `models/` — 模型定义（vision_transformer / predictor / finetune_v_model …）
- `optimization/` — optimizer / schedulers
- `utils/` — 分布式、日志、指标、绘图等工具

> 规划中：音频预训练/微调（`finetune_a`）、音视频（VA）联合训练，以及 `multi_label_classification` 任务类型尚未实现。

## 调试

```bash
python -m app.main --fname CONFIGS/test.yaml --devices cuda:0 --debugmode True
python -m app.main --fname CONFIGS/test-finetune.yaml --devices cuda:0 --debugmode True
```

## 预训练

```bash
python -m app.main --fname CONFIGS/tasks/pretrain_v_FaVoR-112px-48f.yaml --devices cuda:0 cuda:1
```

## 退火 (cooldown / anneal)

```bash
python -m app.main --fname CONFIGS/tasks/pretrain_v_FaVoR-cooldown.yaml --devices cuda:0 cuda:1
```

## 微调

```bash
# 例：RAVDESS emotion 8 类
python -m app.main --fname CONFIGS/tasks/finetune_v_RAVDESS-emotion.yaml --devices cuda:0 cuda:1
```

已覆盖 7 个视频数据集：CREMA-D / EmotionTalk / IEMOCAP / MER2023 / MER242526 / RAVDESS / AVEC2014；各任务入口见 `CONFIGS/tasks/finetune_v_*.yaml`。

## 画 loss 曲线

```bash
python utils/plot_train.py --logdir OUTPUT/pretrain_v/vitl16/FaVoR-112px-48f/
```
