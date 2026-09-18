#!/usr/bin/env bash
# 分类
torchrun --master_port 12345 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/CREMA-D-emotion.yaml 
torchrun --master_port 12347 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/EmotionTalk-emotion.yaml 
torchrun --master_port 12350 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/IEMOCAP-emotion.yaml 
torchrun --master_port 12352 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER2023-emotion.yaml 
torchrun --master_port 12355 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml 
torchrun --master_port 12357 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-emotion.yaml 
torchrun --master_port 12358 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-intensity.yaml 

# 回归
torchrun --master_port 12344 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/AVEC2014-PHQ.yaml 
torchrun --master_port 12346 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/CREMA-D-intensity.yaml 
torchrun --master_port 12348 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-activation.yaml 
torchrun --master_port 12349 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-dominance.yaml 
torchrun --master_port 12351 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-valence.yaml 
torchrun --master_port 12353 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER2023-pos_intensity.yaml 
torchrun --master_port 12356 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER242526-pos_intensity.yaml 

#多标签分类
torchrun --master_port 12354 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/mlcls/MER242526-26openset.yaml 


# ---- vjepaori 基线：不加载 FaVoR 自监督预训练，直接用官方 V-JEPA 2.1 原始权重 ----
# 与上面 15 行一一对应（端口 / --fname 完全相同），只多一层 --set 覆盖：
#   meta.read_checkpoint -> CKPT/vjepa2/vitl.pt
#   folder               -> 任务同名目录下的 FaVoR-112px-48f-8fps-vjepaori/
# 写法说明见 CONFIGS/README.md 10.1；单任务先例见
# CONFIGS/tasks/vfinetune/cls/RAVDESS-emotion/vjepa.yaml
torchrun --master_port 12345 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/CREMA-D-emotion.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/CREMA-D-emotion/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12347 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/EmotionTalk-emotion.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/EmotionTalk-emotion/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12350 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/IEMOCAP-emotion.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/IEMOCAP-emotion/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12352 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER2023-emotion.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/MER2023-emotion/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12355 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/MER242526-emotion/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12357 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-emotion.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/RAVDESS-emotion/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12358 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-intensity.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/RAVDESS-intensity/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12344 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/AVEC2014-PHQ.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/AVEC2014-PHQ/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12346 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/CREMA-D-intensity.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/CREMA-D-intensity/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12348 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-activation.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/IEMOCAP-activation/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12349 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-dominance.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/IEMOCAP-dominance/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12351 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-valence.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/IEMOCAP-valence/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12353 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER2023-pos_intensity.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/MER2023-pos_intensity/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12356 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER242526-pos_intensity.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/MER242526-pos_intensity/FaVoR-112px-48f-8fps-vjepaori
torchrun --master_port 12354 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/mlcls/MER242526-26openset.yaml --set meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt folder=/home/data/sdc/FAVOR/OUTPUT/finetune_v/vitl16/MER242526-26openset/FaVoR-112px-48f-8fps-vjepaori

python PLOT/pca.py --ckpt OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt


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


torchrun --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/afinetune/cls/IEMOCAP-emotion.yaml --set  meta.read_checkpoint=/home/data/sdc/FAVOR/CKPT/emotion2vec_plus_large/model.pt    folder=/home/data/sdc/FAVOR/OUTPUT/finetune_a/emotion2vec-plus-large/IEMOCAP-emotion




torchrun --master_port 12358 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/latest.pt folder=OUTPUT/finetune_v/vitl16/MER242526-emotion/FaVoR-112px-48f-8fps-0-32-e32


##############################
torchrun --master_port 12359 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme-32resume/latest.pt folder=OUTPUT/finetune_v/vitl16/MER242526-emotion/FaVoR-112px-48f-8fps-32-38-latest
torchrun --master_port 12360 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme-32resume/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/MER242526-emotion/FaVoR-112px-48f-8fps-32-38-bestrankme
torchrun --master_port 12362 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme-32resume/best_loss.pt folder=OUTPUT/finetune_v/vitl16/MER242526-emotion/FaVoR-112px-48f-8fps-32-38-bestloss
torchrun --master_port 12361 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme-32resume/best_trace.pt folder=OUTPUT/finetune_v/vitl16/MER242526-emotion/FaVoR-112px-48f-8fps-32-38-besttrace

