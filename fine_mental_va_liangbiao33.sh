#!/usr/bin/env bash

checkpoint_dir=/home/lx/mental/CKPTS/pretrain_va_lejepa/JEPA-e5-latest-C1
sync_preflight_file=/home/lx/mental/FaVoR/DATASET/va_sync_preflight.json

for config in CONFIGS/tasks/finetune/audio-video/mental-auto/*-auto-total-best.yaml; do
  torchrun --standalone --nproc_per_node=4 -m app.main --fname "$config" \
    --set "finetune.pretrained_checkpoint=$checkpoint_dir/step-5000.pt" \
          "data.sync_preflight_file=$sync_preflight_file" || exit 1
done

for config in CONFIGS/tasks/finetune/audio-video/mental-auto/*-auto-anxiety-best.yaml; do
  torchrun --standalone --nproc_per_node=4 -m app.main --fname "$config" \
    --set "finetune.pretrained_checkpoint=$checkpoint_dir/step-5000.pt" \
          "data.sync_preflight_file=$sync_preflight_file" || exit 1
done

for config in CONFIGS/tasks/finetune/audio-video/mental-auto/*-auto-depression-best.yaml; do
  torchrun --standalone --nproc_per_node=4 -m app.main --fname "$config" \
    --set "finetune.pretrained_checkpoint=$checkpoint_dir/step-75000.pt" \
          "data.sync_preflight_file=$sync_preflight_file" || exit 1
done
