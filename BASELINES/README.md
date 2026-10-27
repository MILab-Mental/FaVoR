# FaVoR baselines

从项目根目录统一使用 `python -m BASELINES.run`（也支持 `python BASELINES/run.py`）。模型范围遵循 [核心表](../核心对比模型精简表.md)；固定的 Hub revision、公开权重 URL 和本地路径在各模型 YAML 中，官方源码提交在 [sources.yaml](/home/data/sdd/mental/OUTPUTS/对比实验初始化权重/sources.yaml)。没有用 Base 替代 Large。

## 环境

已运行环境：Python 3.13、PyTorch/torchaudio 2.11.0、torchvision 0.26.0、transformers 4.57.6、timm 0.9.2，RTX PRO 6000 Blackwell，BF16。复用根目录依赖；LeAVJEPA/WavJEPA 另需 `lightning`、`torchmetrics`。没有在线日志服务。

```bash
pip install -r requirements.txt
pip install timm==0.9.2 transformers==4.57.6 lightning==2.6.1 torchmetrics==1.9.0
python -m BASELINES.common.official --fetch AudioMAE VideoMAE cav-mae LeAVJEPA WavJEPA vjepa2 dinov2 AVSiam
```

未验证的旧栈另需官方对应环境：EAT/AV-HuBERT 使用 Fairseq（AV-HuBERT 还需 `python_speech_features`）；MaskFeat 使用 SlowFast、固定源码版 PyTorchVideo、`simplejson`、detectron2；LanguageBind 使用其官方旧版 transformers/peft 组合。不能把当前 Python 3.13 环境当作这些旧栈已经兼容的证据。

官方源码位于 `/home/data/sdd/mental/OUTPUTS/对比实验初始化权重/sources/`，按提交校验。AudioMAE 使用官方 Swin 解码器；兼容层仅处理旧 timm 的默认 `qk_scale`、旧 NumPy/PyTorch 名称及隔离同名包，未修改系统 timm。AVSiam Base+ 仅删除经语法树确认未使用的调试/ToMe 导入，未改骨干计算。WavJEPA 绕过官方 `inference_mode` 特征接口，直接训练相同模块；继续预训练沿用发布权重的 GELU、原生 mask/教师目标并恢复 EMA。

## 模型与验证

表中既有验证记录来自此前的小规模接入检查；当前 CSV 原划分与 merged 数据协议的复查状态见验证记录，历史自定义划分结果不作当前协议结果。“完整 smoke”指指定权重、真实数据、编码器/任务头非零有限梯度及参数更新、保存、完整恢复后再更新、验证/测试、预测数量检查；预训练另检查解码器/投影器和编码器迁移。共 14 个直接微调模型、5 条预训练→微调路线通过；另在 CNSCED 上检查单独的多标签/回归任务。每次仅 2 个真实样本、至多 2 次更新，均为功能验证，不能当作正式性能结果。逐项参数量和数据清单 SHA256 见 [verification.json](/home/data/sdd/mental/OUTPUTS/对比实验初始化权重/verification.json)；下表最终状态也以该记录为准。

| 模型 / 配置 | 固定版本及保留部分 | 路线 / 状态 |
|---|---|---|
| [wavlm-large](finetune/configs/wavlm-large.yaml) | WavLM Large，24 层，纯 PT | 直接微调；完整 smoke |
| [whisper-large-v3](finetune/configs/whisper-large-v3.yaml) | 原版多语种 Whisper large-v3，仅 audio encoder | 直接微调；完整 smoke |
| [audiomae-base](finetune/configs/audiomae-base.yaml) | AudioMAE ViT-B，AS-2M 纯 PT | 直接微调；完整 smoke |
| [data2vec-audio-large](finetune/configs/data2vec-audio-large.yaml) | data2vec **1.0** audio Large | 直接微调；完整 smoke |
| [hubert-large-ll60k](finetune/configs/hubert-large-ll60k.yaml) | HuBERT Large LL60k，非 `-ft` | 直接微调；完整 smoke |
| [wav2vec2-large-lv60](finetune/configs/wav2vec2-large-lv60.yaml) | wav2vec 2.0 Large LV60，非 ASR FT | 直接微调；完整 smoke |
| [eat-large-20ep](finetune/configs/eat-large-20ep.yaml) | EAT Large 20ep，官方 Fairseq PT | 接入代码；当前缺 Fairseq/指定 Fairseq 权重，未验证；HF 推理版不替代正式训练 |
| [beats-iter3](finetune/configs/beats-iter3.yaml) | BEATs Iter3，非 Iter3+ | 接入代码；指定 OneDrive 权重本次 TLS/SSL 失败，未验证 |
| [wavjepa-base](finetune/configs/wavjepa-base.yaml) | 官方 WavJEPA Base，仅编码路径约 89M | 直接微调；完整 smoke |
| [dinov2-vit-large-patch14](finetune/configs/dinov2-vit-large-patch14.yaml) | DINOv2 ViT-L/14，无 registers | 逐帧 + 特征时间均值；完整 smoke |
| [videomae-vit-large-patch16](finetune/configs/videomae-vit-large-patch16.yaml) | 原版 VideoMAE ViT-L K400 1600e PT | 原生 16 帧视频；完整 smoke |
| [mae-dfer-lgi-former](finetune/configs/mae-dfer-lgi-former.yaml) | MAE-DFER/LGI-Former，dim512、depth16、160px | 原生视频；指定 Google Drive 权重本次 TLS/SSL 失败，未验证 |
| [dinov3-vit-large-patch16](finetune/configs/dinov3-vit-large-patch16.yaml) | DINOv3 ViT-L/16 LVD-1689M | 逐帧均值；代码接入，指定仓库需要访问授权，未验证 |
| [vjepa2-1-vit-large-patch16-384](finetune/configs/vjepa2-1-vit-large-patch16-384.yaml) | V-JEPA **2.1** ViT-L/16，384px；EMA 编码器 | 原生 64 帧、末层 token 均值；完整 smoke |
| [clip-vit-large-patch14](finetune/configs/clip-vit-large-patch14.yaml) | CLIP ViT-L/14 224px，仅视觉分支含投影 | 逐帧均值；完整 smoke |
| [hiera-large-16x224](finetune/configs/hiera-large-16x224.yaml) | Hiera-L `mae_k400`，16×224，非 FT | 原生视频；完整 smoke |
| [maskfeat-mvit-large](finetune/configs/maskfeat-mvit-large.yaml) | MaskFeat MViT-L K400 800ep | 代码接入；当前缺 detectron2 等 SlowFast 环境和指定权重，未验证 |
| [social-mae](finetune/configs/social-mae.yaml) | Social-MAE VoxCeleb2 纯 PT，8 帧/tubelet2 | 代码接入；指定下载链接本次返回 404，未验证 |
| [av-hubert-large](finetune/configs/av-hubert-large.yaml) | AV-HuBERT Large clean PT，仅 AV 编码器 | 代码接入；缺 Fairseq、指定权重及真实唇部 ROI 清单，未验证 |
| [cav-mae-scale-plus-plus](finetune/configs/cav-mae-scale-plus-plus.yaml) | CAV-MAE Scale++ 纯 PT，**单图像帧**主线 | 联合 AV 微调；完整 smoke |
| [avsiam-base-plus](finetune/configs/avsiam-base-plus.yaml) | AVSiam **Base+**，共享骨干 + 原生融合块 | 联合 AV 微调；完整 smoke；上游有 ImageNet-21k 监督初始化 |
| [languagebind-audio-large-video-large-v1-5](finetune/configs/languagebind-audio-large-video-large-v1-5.yaml) | LanguageBind Audio-Large + Video-Large V1.5，完整 FT | 独立编码器及投影 + 拼接任务头；代码接入，当前缺 peft，官方 transformers 4.30.2 接口亦需隔离环境验证 |

| 额外路线 | 原生目标 / 状态 |
|---|---|
| [audiomae-large](pretrain/configs/audiomae-large.yaml) | **Large 随机初始化**，官方局部 Swin 解码器及归一化掩码频谱重建；真实预训练→微调 |
| [videomae-vit-large-patch16](pretrain/configs/videomae-vit-large-patch16.yaml) | VideoMAE Large 公开初始化，保留指定权重的 **12 层解码器**，tube mask + 归一化视频像素重建；真实预训练→微调 |
| [leavjepa-large](pretrain/configs/leavjepa-large.yaml) | LeAVJEPA Large 随机初始化，共享 Echo、G2/K2、单模态 token drop、不变性 + SIGReg；真实预训练→微调 |
| [wavjepa-base](pretrain/configs/wavjepa-base.yaml) | WavJEPA Base 初始化，原生 context/target mask、12 层教师表示及 EMA；真实继续预训练→微调 |
| [cav-mae-scale-plus-plus](pretrain/configs/cav-mae-scale-plus-plus.yaml) | CAV-MAE Scale++ 初始化，重建 + 对比，保留原解码器；完整真实预训练→微调 smoke |
| [eat-large-20ep](pretrain/configs/eat-large-20ep.yaml) | 原生 Fairseq 帧级/句级目标、克隆 mask 与 EMA；同 EAT Large 环境/权重限制，未验证 |
| [maskfeat-mvit-large](pretrain/configs/maskfeat-mvit-large.yaml) | 原生 3D mask + HOG 目标；同 MaskFeat 环境/权重限制，未验证 |
| avsiam-large | **未实现、未注册**：发布仓库缺 Large 架构/训练配置，仅有 Base 入口和部分 `.pyc`；没有拿 Base+ 充当 Large |

Hiera 输入归一化采用固定提交的官方 `examples/inference.ipynb`：mean=`[.45,.45,.45]`、std=`[.225,.225,.255]`，其中第三通道不同于 SlowFast 通用默认值，已在适配器和运行配置明确记录。

FaVoR 无联合预训练对照使用 [favor-av-no-joint-pretrain.yaml](finetune/configs/favor-av-no-joint-pretrain.yaml)，复用原 `build_va_lejepa`、`VAFinetuner`、视图解码和句子聚合，只跳过联合 SSL。`model.favor_config` 指定原生下游配置，`model.favor_pretrain_config` 指定联合 SSL 对照配置并校验骨干/融合结构，从后者提取 A/V 初始化以保证对照一致。当前 i3 对照所需 `OUTPUT/pretrain_v_lejepa/vitl16/FaVoR-112px-48f8/e25.pt` 和 `OUTPUT/pretrain_a_lejepa/emotion2vec-plus-large/FaVoR-4s-local2s/latest.pt` 均不存在，尚未真实验证；不回退到其他权重。

FaVoR 晚期概率融合对照使用 `finetune.fusion`：CSV 有验证集时从验证预测选概率权重；没有验证集时用固定概率权重（默认 0.5），仅在测试集评估；严格核对样本、标签和划分。当前缺这样的两套预测，尚未完成真实 FaVoR 融合验证。

## 数据与权重

下游只读取项目 `DATASET/splits-0901/csvs/` 中对应任务的原始 CSV，使用 `<label_column>_split`：`0=train`、`1=test`、存在时 `2=validation`，`-1`/空值不参与该任务。不再按说话人、参与者或会话重新划分，也不要求额外身份字段。只有 CSV 提供验证集时才按验证指标选点；没有验证集时固定训练轮数，`best.pt` 保存最终 epoch，关闭验证早停，测试集只用于显式评估。

预训练按模态直接读取以下三个真实清单，不再限制为 CNCVS1 或划出验证对象：

| 模态 | 清单 | 真实格式 |
|---|---|---|
| 音频 | `DATASET/merged/pretrain_audios_0901.csv` | 无表头，每行“路径 + 数字标签” |
| 视频 | `DATASET/merged/pretrain_videos_0901.csv` | 无表头，每行“路径 + 数字标签” |
| 音视频 | `DATASET/merged/pretrain_va_canonical.csv` | `pair_id,source_id,video_path,audio_path,label` |

三个清单中的数字标签不作为划分，全部清单行用于预训练，不额外划验证集。复用 FaVoR 的 mmap 行索引按需读取，避免一次展开数百万行；音视频保留 canonical 身份和同步检查。`data.limit` 只用于小规模检查，不改变 CSV 的正式划分。

Mental 长媒体任务（`*_LA.csv` / `*_LV.csv`）按 FaVoR 的说话人/原始访谈实例处理：复用 `sentence_identity` 去掉 `-sXXXX` 后缀并保留目录，把同一原始长媒体的切片合为一个样本。训练默认采样 8 句，先用官方基线编码器提取句子特征，再复用 FaVoR 的 `InstancePredictionHead`（可学习注意力聚合、LayerNorm、分类/回归头）产生一次实例预测和监督损失。测试指标、预测 CSV 和样本数量均按实例统计。采样配置也复用 FaVoR：音频遍历句子窗口，视频/音视频默认随机采样 8 句，评估随机状态固定。CSV 的训练/测试归属保持原样，这里的聚合不会重新划分数据。

微调默认 `data.cohort=paired` 使用共同有 A/V 路径的样本；纯音频任务设 `data.cohort=modality`。标签、损失、指标和媒体路径注册复用 FaVoR。保留各模型原生输入与长样本窗口聚合，训练标签权重只从 CSV 训练行计算。AV-HuBERT 需要真实唇部输入。Whisper 原生固定 30 秒输入，池化排除 padding；WavJEPA SSL 按官方方式补齐短录音并使用固定长度原生目标，微调读出保留有效长度。

Hub 模型用固定 revision；本地快照放 `/home/data/sdd/mental/OUTPUTS/对比实验初始化权重/` 后配置 `model.weights`。例如：

```bash
hf download microsoft/wavlm-large --revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c \
  --include config.json preprocessor_config.json pytorch_model.bin \
  --local-dir /home/data/sdd/mental/OUTPUTS/对比实验初始化权重/wavlm-large
```

Google Drive/OneDrive/Dropbox 文件按模型 YAML 中 `weight_source` 获取，保持指定版本；下载到 HTML 或缺失文件不能当权重。加载只允许显式上游组件丢弃和新的读出归一化，其他缺失/形状不匹配直接报错。MaskFeat 仅使用官方相对位置插值。

## 最小运行命令

`--models` 使用上表对应的配置名，不带 `.yaml`；可给多个名称顺序运行。`--config` 可直接指定包含模型、任务和数据的完整 YAML。`--list` 只列配置，不构建模型；表中未验证条目仍需补齐所列资源。

```bash
BASELINE_OUTPUTS=/home/data/sdd/mental/OUTPUTS/对比实验输出

# 查看阶段配置
python -m BASELINES.run --stage finetune --list
python -m BASELINES.run --stage pretrain --list

# 直接微调；多个 --models 名称按顺序运行
python -m BASELINES.run --stage finetune --models wavlm-large --device cuda:1

# 从头预训练 / 继续预训练
python -m BASELINES.run --stage pretrain --models audiomae-large
python -m BASELINES.run --stage pretrain --models videomae-vit-large-patch16

# 加载研究预训练导出的编码器；默认路径已写在对应微调配置中
python -m BASELINES.run --stage finetune --models audiomae-large
python -m BASELINES.run --stage finetune --models videomae-vit-large-patch16-continued

# 自定义任务配置；--set 仍可覆盖数据、优化器或权重路径
python -m BASELINES.run --stage finetune \
  --models wavjepa-base --set data.datasets=[DATASET/splits-0901/csvs/2_CNSCED_SA.csv] \
  data.task=multi_label_classification data.label_column=emotion data.num_class=7 data.cohort=modality

# 测试 / 完整续训
python -m BASELINES.run --stage finetune --models wavlm-large --evaluate test \
  --checkpoint "$BASELINE_OUTPUTS/finetune/wavlm-large/ravdess-emotion-direct/best.pt" \
  --set save_predictions=true
python -m BASELINES.run --stage finetune --models wavlm-large \
  --resume "$BASELINE_OUTPUTS/finetune/wavlm-large/ravdess-emotion-direct/last.pt"

# 小规模真实检查；预训练模式还会验证编码器迁移到微调
python -m BASELINES.run --stage finetune --models wavlm-large --smoke --device cuda:1
python -m BASELINES.run --stage pretrain --models wavjepa-base --smoke --device cuda:1
python -m pytest tests/test_baselines.py -q

# FaVoR 概率融合；源预测必须覆盖同一组划分的共同验证/测试样本
python -m BASELINES.run --stage fusion --data-config audited-task.yaml \
  --audio-test audio-test.csv --video-test video-test.csv --audio-weight 0.5
```

用 `--device cpu` 或 `--device cuda:0` 选设备。`--smoke` 固定两样本、两次更新；默认检查完成后删除临时权重，`--keep-checkpoints` 可保留。预训练 smoke 的下游任务可用 `--finetune-config` 指定。正式训练的 epoch、批量和路径由 YAML / `--set` 决定；恢复拒绝改变模型、数据、优化器、seed 和原定预算，恢复边界是 epoch，尚未实现中途 batch 精确恢复或 DDP。

## 路径与输出

模型 ID 和配置文件统一使用小写、连字符和真实网络名，例如 `wavlm-large`、`videomae-vit-large-patch16`。继续预训练后的微调配置使用 `-continued.yaml` 后缀；预训练配置在 `pretrain/configs/` 选择同名模型。

固定存储根路径集中定义于 `common/runtime.py`：

- 输出：`/home/data/sdd/mental/OUTPUTS/对比实验输出/`。
- 初始化权重：`/home/data/sdd/mental/OUTPUTS/对比实验初始化权重/`，同时保存 `sources/`、`sources.yaml` 和 `verification.json`。

正式输出为 `<输出根目录>/<pretrain|finetune>/<model>/<experiment>/`，例如 `finetune/wavlm-large/ravdess-emotion-direct/`、`pretrain/videomae-vit-large-patch16/merged-continue/`。Smoke 位于对应阶段的 `smoke/<model>/<experiment>/`，检查结论直接写入各实验的 `metrics.json`。下游和预训练清单均使用上述原始数据路径。`--set output=...` 可覆盖单次输出路径，`model.weights` 和 `model.encoder_checkpoint` 使用配置中的显式路径。

正式训练仅保存最终 `config.yaml`、`train.log`、`metrics.json`、轻量 `best.pt` 和完整恢复的 `last.pt`；需要时用 `save_predictions=true` 生成 `predictions.csv`。恢复包含优化器、调度器、scaler、Python/NumPy/PyTorch/CUDA 随机状态及训练/验证 DataLoader RNG。预训练权重另含可直接转移的纯 `encoder` 映射；参数量区分保留编码器、新头及教师/解码器/投影器，EAT 的树外 EMA 单列。

历史两步 smoke 权重已清理，保留配置、日志、指标、预测和验证记录，已清理的 smoke 不能直接续训。新 smoke 也默认采用相同保留策略；正式训练保留检查点。

Mental 长视频分类小规模检查示例：

```bash
python -m BASELINES.run --stage finetune --models videomae-vit-large-patch16 \
  --smoke --device cuda:1 --experiment mental-dsm-depression \
  --set 'data.datasets=[DATASET/splits-0901/csvs/3_私有临床数据_LV.csv]' \
  data.task=classification data.label_column=DSM-Depression data.num_class=2
```
