#!/usr/bin/env bash
# bash train_audio.sh：先官方权重71个任务，再自预训练权重71个任务；连续10轮未提升早停。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
OUTPUT_ROOT=/home/data/sdd/mental/OUTPUTS/finetune_a_compare/baseline-patience10-20261027

TASKS=(
    CONFIGS/tasks/finetune/audio/mental/*.yaml
    CONFIGS/tasks/finetune/audio/classification/*.yaml
    CONFIGS/tasks/finetune/audio/regression/*.yaml
    CONFIGS/tasks/finetune/audio/multilabel/*.yaml
)

train_weight() {
    local checkpoint=$1 name=$2 config group task
    for config in "${TASKS[@]}"; do
        group=$(basename "$(dirname "$config")")
        task=$(basename "$config" .yaml)
        [[ $task == MER242526-emotion-lejepa ]] && continue  # 排除同一任务的实验变体
        echo "训练：$name / $group / $task"
        torchrun --standalone --nproc_per_node=2 -m app.main --fname "$config" --set \
            "meta.read_checkpoint=$checkpoint" \
            "folder=$OUTPUT_ROOT/$name/$group/$task" \
            meta.load_checkpoint=true meta.reset_epoch=true meta.frozen_encoder=false meta.seed=239 \
            model.features.mode=last data.batch_size=16 \
            optimization.grokfast=false optimization.early_stopping_patience=10 \
            || echo "训练失败：$name / $task" >&2
    done
}

train_weight "$PWD/CKPT/emotion2vec_plus_large/model.pt" official
train_weight /home/data/sdd/mental/OUTPUTS/pretrain_a/emotion2vec-plus-large/FaVoR-2to6s/latest.pt jepa
