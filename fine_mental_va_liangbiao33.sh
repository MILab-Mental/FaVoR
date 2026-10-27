for config in CONFIGS/tasks/finetune/audio-video/mental-auto/*-auto-total-best.yaml; do torchrun --standalone --nproc_per_node=2 -m app.main --fname "$config" || exit 1; done
for config in CONFIGS/tasks/finetune/audio-video/mental-auto/*-auto-anxiety-best.yaml; do torchrun --standalone --nproc_per_node=2 -m app.main --fname "$config" || exit 1; done
for config in CONFIGS/tasks/finetune/audio-video/mental-auto/*-auto-depression-best.yaml; do torchrun --standalone --nproc_per_node=2 -m app.main --fname "$config" || exit 1; done
