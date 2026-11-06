#!/usr/bin/env bash
# 取消需要的命令前面的 #，修改显卡、配置或参数后：bash train_pretrain.sh
# 通常一次只启用一条；启用多条会依次运行，遇到失败停止。
# 在命令末尾追加 --set data.batch_size=16 folder=OUTPUT/实验目录 等覆盖参数。
# 初始化权重和续训规则沿用对应 YAML。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 1. 视频 JEPA
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/pretrain/video/jepa/favor-112px-48frames-4fps.yaml

# 2. 视频 LeJEPA
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/pretrain/video/lejepa/favor-112px-48frames-8fps.yaml

# 3. 音频 JEPA
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/pretrain/audio/jepa/favor.yaml

# 4. 音频 LeJEPA
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/pretrain/audio/lejepa/default.yaml

# 5. 音视频 LeJEPA
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/pretrain/audio-video/lejepa/default.yaml
