#!/usr/bin/env bash
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/AVEC2014-PHQ/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/CREMA-D-emotion/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/CREMA-D-intensity/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/EmotionTalk-emotion/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/IEMOCAP-activation/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/IEMOCAP-dominance/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/IEMOCAP-emotion/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/IEMOCAP-valence/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/MER2023-emotion/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/MER2023-pos_intensity/finetune_v.yaml --devices cuda:0 cuda:1

#多标签
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/MER242526-26openset/finetune_v.yaml --devices cuda:0 cuda:1
###############
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/MER242526-emotion/finetune_v-favor-e11.yaml --devices cuda:0 cuda:1
###############
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/MER242526-pos_intensity/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/RAVDESS-emotion/finetune_v.yaml --devices cuda:0 cuda:1
torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/RAVDESS-intensity/finetune_v.yaml --devices cuda:0 cuda:1

favor-e11-4layer
  opt: 原始
  best F1:   0.44886 @ epoch 12
  best loss: 1.40019 @ epoch 4

favor-e11-23out
  opt: 原始
  best acc:  0.5219 @ epoch 15
  best F1:   0.4441 @ epoch 15
  best loss: 1.3780 @ epoch 4

favor-e11-frozen-23out
  opt: 原始
  best acc:  0.4781 @ epoch 10
  best F1:   0.4004 @ epoch 17
  best loss: 1.4886 @ epoch 7

favor-e11-23out-newopt
  opt: newopt
  best acc:  0.5219 @ epoch 4
  best F1:   0.4564 @ epoch 4
  best loss: 1.4331 @ epoch 3

favor-e11-23out-newnewopt
  opt: newnewopt
  best acc:  0.5219 @ epoch 13
  best F1:   0.4548 @ epoch 13
  best loss: 1.3976 @ epoch 4

favor-e11-23out-newnewopt-sqrtweight
  opt: newnewopt
  loss: sqrtweight
  best acc:  0.5271 @ epoch 9
  best F1:   0.4533 @ epoch 27
  best loss: 1.4344 @ epoch 4

favor-e11-4layer-newopt
  opt: newopt
  best acc:  0.5079 @ epoch 21
  best F1:   0.4438 @ epoch 22
  best loss: 1.4306 @ epoch 4

favor-e5-4layer-newopt
  opt: newopt
  best acc:  0.5482  @ epoch 11
  best F1:   0.4694  @ epoch 7
  best loss: 1.3378  @ epoch 7

favor-e5-4layer-data48-8-newopt
  opt: newopt
  data: 48-8
  best acc:  0.5289  @ epoch 57
  best F1:   0.4525  @ epoch 40
  best loss: 1.3936  @ epoch 4

favor-e5-4layer-data48-8-newnewopt
  opt: newnewopt
  data: 48-8
  best acc:  0.5692 @ epoch 10
  best F1:   0.4907 @ epoch 10
  best loss: 1.3803 @ epoch 9

favor-e5-4layer-data64-16-newopt
  opt: newopt
  data: 64-16
  best acc:  0.5394 @ epoch 49
  best F1:   0.4566 @ epoch 49
  best loss: 1.4047 @ epoch 4

favor-e5-4layer-data64-16-newnewopt
  opt: newnewopt
  data: 64-16
  best acc:  0.5569 @ epoch 13
  best F1:   0.4515 @ epoch 36
  best loss: 1.2716 @ epoch 7

favor-e5-23out-newnewopt
  opt: newnewopt
  data: 48-4
  best acc:  0.5499 @ epoch 25
  best F1:   0.4791 @ epoch 25
  best loss: 1.3814 @ epoch 6

favor-e5-23out-data48-8-newnewopt
  opt: newnewopt
  data: 48-8
  best acc:  0.5499 @ epoch 23
  best F1:   0.4722 @ epoch 23
  best loss: 1.4165 @ epoch 6 
