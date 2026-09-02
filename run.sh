#!/usr/bin/env bash
python -m app.main --fname CONFIGS/tasks/finetune_v_AVEC2014-PHQ.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_CREMA-D-emotion.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_CREMA-D-intensity.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_EmotionTalk-emotion.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_IEMOCAP-activation.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_IEMOCAP-dominance.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_IEMOCAP-emotion.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_IEMOCAP-valence.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_MER2023-emotion.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_MER2023-pos_intensity.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_MER242526-26openset.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_MER242526-emotion.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_MER242526-pos_intensity.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_RAVDESS-emotion.yaml --devices cuda:0 cuda:1
python -m app.main --fname CONFIGS/tasks/finetune_v_RAVDESS-intensity.yaml --devices cuda:0 cuda:1
