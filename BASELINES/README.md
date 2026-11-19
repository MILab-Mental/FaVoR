# FaVoR 对比实验

在项目根目录运行。模型范围见 [核心表](核心对比模型精简表.md)，版本、Hub revision、权重来源和 SHA256 见各模型 YAML；官方源码固定提交见初始化目录的 `sources.yaml`。没有使用 Base 权重替代 Large。

## 环境与路径

主环境：Python 3.13、PyTorch/torchaudio 2.11.0、torchvision 0.26.0、transformers 4.57.6、timm 0.9.2；LeAVJEPA/WavJEPA 使用 lightning 2.6.1、torchmetrics 1.9.0。使用两张 RTX PRO 6000 96GB、BF16。

启动时在终端运行 `conda activate favor`，然后使用 `python -m BASELINES.run` 或下文的脚本。启动器会为需要兼容环境的模型选择对应解释器。

EAT、LanguageBind 使用项目内 `.venv/baseline-legacy/bin/python`，AV-HuBERT 使用 `.venv/baseline-avhubert/bin/python`；两套 Python 3.10 环境从初始化目录复用依赖文件，并分别加载各自的 Fairseq。初始化目录必须保持挂载。主环境不变。AV-HuBERT 使用预处理的完整人脸视频，不依赖 dlib 或嘴部检测。MaskFeat 使用固定 SlowFast/PyTorchVideo 源码；仅对其未使用的检测头导入提供 torchvision RoIAlign，分类和 HOG 目标使用官方实现。

权重、输出和兼容 Python 的位置统一在 [paths.yaml](paths.yaml) 配置。默认输出目录是 `OUTPUT/baselines`，相对路径以项目根目录为基准。换服务器时修改这份配置，也可以用同名环境变量覆盖，例如 `BASELINES_WEIGHTS_ROOT=/新服务器/对比实验初始化权重`。

权重目录需要包含模型权重、`sources/`、`sources.yaml` 和所需兼容依赖。可通过 `python -m BASELINES.common.official --fetch <源码名称>` 按锁定提交获取源码。

输出保存在输出根目录的 `{pretrain,finetune}/<model>/<experiment>/`。迁移微调默认从同一输出根目录读取对应预训练的 `latest.pt`。个别运行可用 `--set model.weights=/权重文件` 或 `--set model.encoder_checkpoint=/预训练检查点` 指定文件，联合模型也可按分支覆盖。

`python PLOT/增加radar.py --baselines` 在同一输出根目录读取 `对比模型进度表.md` 和微调指标，默认也将雷达图写到该目录。

数据读自 `CONFIGS/datasets.csv` 和相应的清单 CSV。更换机器时，数据注册表的媒体根目录和清单中的绝对路径需要在目标机器上可访问。

## 模型与验证状态

“全任务检查”是本轮真实 CSV、两卡小批量训练检查：指定初始化、有限 loss/梯度、编码器和任务头更新、两卡参数同步、内存序列化及恢复、验证/测试指标与预测数量核对。每任务通常一次更新，首个适用任务两次更新并恢复；不是正式收敛结果。音频有 71 个适用任务；视频/音视频有 53 个，其余 18 个只有音频。预训练另检查原生目标组件及编码器迁移。

下表配置名均位于 `finetune/configs/<配置名>.yaml`。当前清单中的初始化权重均已就绪。30 条微调路线、1,806 个适用的模型/任务组合通过真实数据小批量检查。9 条预训练路线均通过原生目标、恢复及编码器迁移检查，包含新增的 Social-MAE 从头预训练。WavJEPA、CLIP、CAV-MAE 另通过有效全局 batch 16、多进程加载检查；AV-HuBERT 通过每卡 batch 2、累积 4 次、每卡 8 个加载进程的真实普通/mental 检查，测试预测数量均为 16。 以下 AV-HuBERT 检查属于先前嘴部输入协议；当前全脸输入方案需要单独验证。

| 配置名 / 网络版本 | 路线与本轮状态 |
|---|---|
| `wavlm-large`：24 层 Large 纯 PT | 直接微调；71 任务通过 |
| `whisper-large-v3`：原版多语种，仅音频编码器 | 直接微调；71 任务通过 |
| `audiomae-base`：ViT-B，AS-2M 纯 PT | 直接微调；71 任务通过 |
| `data2vec-audio-large`：data2vec **1.0** Large | 直接微调；71 任务通过 |
| `hubert-large-ll60k`：Large LL60k，非 ASR FT | 直接微调；71 任务通过 |
| `wav2vec2-large-lv60`：Large LV60，非 ASR FT | 直接微调；71 任务通过 |
| `eat-large-20ep`：官方 Fairseq Large 20ep | 直接 / 继续预训练；各 71 任务通过，原生预训练、恢复及迁移通过 |
| `beats-iter3`：Iter3，非 Iter3+ | 直接微调；71 任务通过；官方链接不可用时下载指定版本镜像，来源、提交和 SHA256 明确登记，未声称与无法获取的原文件完成哈希比对 |
| `wavjepa-base`：官方 Base | 直接 / 继续预训练；各 71 任务通过 |
| `dinov2-vit-large-patch14`：ViT-L/14，无 registers | 逐帧微调；53 任务通过 |
| `dinov3-vit-large-patch16`：ViT-L/16 LVD-1689M | 逐帧微调；53 任务通过 |
| `clip-vit-large-patch14`：ViT-L/14，视觉分支含投影 | 逐帧微调；53 任务通过 |
| `videomae-vit-large-patch16`：原版 Large K400 1600ep PT | 直接 / 继续预训练；各 53 任务通过，保留原生 12 层预训练解码器 |
| `mae-dfer-lgi-former`：LGI-Former，512 维、16 层、160px | 直接微调；53 任务通过 |
| `vjepa2-1-vit-large-patch16-384`：V-JEPA **2.1** Large EMA（384 权重） | 直接微调输入使用 112×112、48 帧、目标 8 fps |
| `hiera-large-16x224`：Hiera-L MAE K400，非 FT | 直接微调；53 任务通过 |
| `maskfeat-mvit-large`：MViT-L K400 800ep | 直接 53 任务通过；继续预训练及迁移通过，迁移后全部 53 个任务通过 |
| `cav-mae-scale-plus-plus`：Scale++ 纯 PT，单图像帧 | 直接 / 继续预训练；各 53 任务通过 |
| `avsiam-base-plus`：官方 Base+，共享骨干及融合块 | 直接微调；53 任务通过；上游包含 ImageNet-21k 监督初始化 |
| `avsiam-large`：随机共享 ViT-L/16，24 层、1024 维 | 论文构建候选；2 层融合 + 6 层解码器，0–50% 多比例掩码；真实数据两卡预训练、完整恢复、编码器迁移及全部 53 个适用微调任务通过 |
| `languagebind-audio-large-video-large-v1-5`：Audio-Large + Video-Large V1.5 完整 FT 发布版 | 独立编码器及投影 + 拼接任务头；两个分支严格加载、真实训练及全部 53 个适用任务通过 |
| `av-hubert-large`：Large LRS3+VoxCeleb2 clean PT | 当前使用预处理全脸输入；历史嘴部输入的全部 53 个适用任务检查不代表当前全脸方案已完成训练验证 |
| `social-mae`：原生 768 维，8 帧 / tubelet 2，随机初始化 | 从头预训练 → 编码器迁移 → 全参数微调；两卡原生目标、恢复、严格迁移及全部 53 个适用微调任务通过；另通过有效全局 batch 16、每卡 4 个加载进程的普通/mental 检查 |

另外两条从头预训练路线：`audiomae-large` 使用随机 Large 和官方局部 Swin 解码器；`leavjepa-large` 使用随机 Large Echo、原生不变性 + SIGReg。两者均通过真实预训练、恢复、迁移及全部适用微调任务。加上 Social-MAE 和上表 AVSiam Large，预训练配置共 9 份，其中 5 条为上表 EAT、WavJEPA、VideoMAE、MaskFeat、CAV-MAE 的继续预训练，微调配置后缀为 `-continued`。

Social-MAE 使用固定官方代码的随机初始化、可训练位置编码、8 层重建解码器、75% 非结构化掩码及 0.01 对比损失；两卡汇集对比负例。输入为 4 秒 / 2fps / 8 帧，沿用其 OpenCV BGR 通道顺序、224px 双三次缩放和频谱归一化，变短视频重复真实帧并按有效时间池化。该路线在本研究配对 CSV 上从头训练，不能作为 VoxCeleb2 公开权重的直接微调或继续预训练结果。

AVSiam Large 按[论文 Sec.3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02220.pdf)补齐，实测下游编码器 329,496,576 参数、预训练模型 349,850,624 参数。复用发布代码的 blocks、重建解码和对比损失，按论文共享重建路径骨干，使用六层解码器及 0–50%、5% 步长的随机分组掩码。按核心表要求**随机初始化**，区别于论文的 ImageNet-21k 初始化，不能当作发表结果的等条件复现。

`verification.json` 是历史接入记录，本轮按“不保存测试结果”要求没有更新；本轮状态以上述说明及控制台检查为准。

## 数据与协议

微调使用 `DATASET/splits-0901/csvs/` 原始 CSV 和 FaVoR 原有 33 个普通、38 个 Mental 任务定义。`<标签列>_split` 的 `0=train`、`1=test`、`2=validation`（如有）。普通任务没有验证集时固定轮数并使用最后一轮。Mental 每轮在测试集评估，按测试指标选点，仍跑满配置的轮数。Mental 的最终结果来自参与选点的测试集。

预训练直接读取 `DATASET/merged/pretrain_audios_0901.csv`、`pretrain_videos_0901.csv`、`pretrain_va_canonical.csv`，数字标签不用于监督或划分，不额外划验证集。复用 FaVoR mmap 清单索引、媒体路径注册、标签解析、损失和指标。音视频比较使用共同有真实 A/V 路径的样本，复用 FaVoR 通用解码器的同步校验、媒体时间起点和共同有效时间区间。`data.sync` 可显式覆盖解码器容差，生效来源记入 `data_audit`；不会把联合预训练模板的零偏移约束强加到下游 CSV（实查 1,121 对媒体，其中 543 对不满足该更严约束，主要包含 MP3 编码起点等差异）。

九条预训练路线默认双卡，每卡 batch size 16，梯度累积 1。每轮从该轮重新打乱的全量清单取 10,000 个 batch，20 轮合计 200,000 次更新、640 万条样本次，跨轮可重复采样。若需用每卡 batch size 8 和梯度累积 2 保持同一更新预算，每轮需设置 `max_batches=20000`。

预训练启动时从采样列表排除 `OUTPUT/baselines/pretrain/error_media_a.csv`、`error_media_v.csv` 和 `error_media_va.csv` 中对应模态的条目。各模型和进程共用这三份 `name,reason` 表。音频和视频使用清单里的文件路径，音视频使用 `pair_id`。运行中遇到空文件、缺失文件或解码失败时，登记原因后从同一清单确定性选取替代样本。修复媒体后可删除表内对应记录。连续读取失败会报错退出。

长媒体按 FaVoR `sentence_identity` 聚合同一说话人/原始录制实例的切片，使用原生 8 句采样与注意力实例头；不会把同一说话人的不同录制全局合并。音频评估遍历句子窗口，视频/音视频复用其固定随机状态的原生采样。FaVoR 无联合预训练对照保留 CSV 中不足 100ms 的非空真实切片，使用原生 zero-pad 和有效长度 mask，不增加样本排除规则。

模型各自保留采样率、频谱、帧数、分辨率和读出。AV-HuBERT 直接将已有全脸视频转换为灰度并缩放到 88×88，不做检测或嘴部裁剪；这与官方嘴部输入协议不同，比较结果应说明此差异。Hiera 使用固定官方 notebook 的 mean `[.45,.45,.45]` / std `[.225,.225,.255]`；MaskFeat 按实际发布权重恢复 split Q/K/V、无 CLS、Gaussian HOG window 16，并仅使用官方相对位置插值。树外 EMA、预训练解码器和下游保留编码器分别统计。

## 最小命令

```bash
# 正式对比训练。两卡依次运行各模型，先预训练再跑微调。
bash BASELINES/pretrain.sh
bash BASELINES/finetune.sh

# 本轮使用的真实数据小批量检查：不保存训练结果或测试报告
bash BASELINES/pretrain.sh --test
bash BASELINES/finetune.sh --test

# 选择路线；配置名不带 .yaml
bash BASELINES/finetune.sh --models wavlm-large
bash BASELINES/pretrain.sh --models audiomae-large       # 从头
bash BASELINES/pretrain.sh --models social-mae          # 从头 AV 重建 + 对比
bash BASELINES/finetune.sh --models social-mae          # 加载上述 latest.pt
bash BASELINES/pretrain.sh --models videomae-vit-large-patch16  # 继续
bash BASELINES/finetune.sh --models videomae-vit-large-patch16-continued

# 只选择某个任务；可用 --set 覆盖权重路径等配置
bash BASELINES/finetune.sh --models wavlm-large --tasks mental/DSM-Depression

# 查看配置；单实验评估已保存的最佳模型
python -m BASELINES.run --stage finetune --list
python -m BASELINES.run --stage finetune --models wavlm-large --evaluate test \
  --checkpoint /完整路径/best.pt --set save_predictions=true
```

两个脚本都只有一条运行命令。`common/defaults.yaml` 定义共享的优化器、学习率、随机种子和精度，预训练的 batch 和步数由 `pretrain/configs/defaults.yaml` 定义。句子微调每卡 batch 8，长媒体实例微调默认每卡 batch 2、累积 4 次梯度。各方法使用激活检查点，默认每卡 4 个数据加载进程，AV-HuBERT 默认 8 个。各方法保留原生目标及相应输入配置。

正式模式会在开始前检查全部所选模型资源；缺权重、研究预训练产物或唇部数据时明确退出，不静默省略。测试模式打印阻塞项目，继续检查可用项，最后返回非零状态。Social-MAE 改为随机初始化后在本研究配对数据上从头预训练，不再依赖失效的 VoxCeleb2 公开权重；只支持该预训练产物的微调，不提供直接微调路线。迁移路线的正式微调仍需先完成对应预训练。`--test` 的少步数结果也不能代替空卡时正式 batch 的吞吐量验证。

## 输出与恢复

临床 Mental 微调首次为整份临床清单生成 `OUTPUT/baselines/mental-precheck.json`。38 个任务、双卡进程和后续测试共用这份媒体检查结果，各任务仍使用自己的标签和划分。可提前运行 `python -m BASELINES.common.mental_precheck`。CSV 或媒体根目录改变后会自动重查；原地替换媒体后可加 `--refresh` 重建。

正式运行生成 `config.yaml`、`train.log`、`metrics.json` 和下列检查点。Mental 的 `infer.csv` 每轮记录测试集逐录制预测，通过 `epoch` 列区分轮次。需要额外导出最终评估预测时才生成 `predictions.csv`。目录名如 `finetune/wavlm-large/direct/mental/DSM-Depression/`，迁移路线使用 `scratch-pretrained` 或 `continued-pretrained`。

- **预训练每轮保留 `epoch-0001.pt`、`epoch-0002.pt` 等权重**。每份包含该轮模型和可迁移编码器。`latest.pt` 每轮更新，另保留优化器、调度器和随机状态等续训信息。
- **预训练 `step_loss.csv` 每次优化器更新一行**。字段为 `epoch,step,loss`，loss 按各 rank 和各 micro-batch 的样本数加权。每 100 步和每轮结束时写入，续训时保留截至已恢复步数的记录。
- **普通微调保留 `best.pt`**。有 CSV 验证集时仅改善才覆盖，没有验证集时保存最后一轮。完成训练和测试后移除续训状态。
- **Mental 微调训练中保留 `best.pt` 和 `latest.pt`**。分类和多标签默认以测试集 macro F1 越高越好，回归默认以 MAE 越低越好。`best.pt` 只在指标严格改善时覆盖，`latest.pt` 保存最近完成轮次的续训状态。任务队列完成测试后保留 `best.pt`。

双卡微调遇到非有限的评估输出时，先对异常批次重算 FP32。仍失败则重启该模型队列一次，从最近保存的检查点恢复。末轮已保存的任务直接重跑测试。再次失败会保留权重和错误日志。

重新执行相同脚本即可恢复未完成的实验，完成训练和测试的微调任务自动跳过。预训练和 Mental 优先从 `latest.pt` 恢复，普通微调从未精简的 `best.pt` 恢复。旧版 Mental 可从未完成的 `best.pt` 续训，并补评该权重作为第一个候选。单实验也可通过 `--resume` 指定含续训状态的检查点。恢复时校验模型、数据、优化器、seed、轮数及卡数，恢复点为轮次边界。

正式运行应为每个预训练模型预留 20 份权重和一份续训检查点的空间。每轮的预训练权重可通过 `model.encoder_checkpoint` 指定给下游微调。
