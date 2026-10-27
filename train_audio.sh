#!/usr/bin/env bash
# 音频两种初始化的顺序对照：38 个 mental + 33 个其他任务，共 142 次。
# 复用现有 YAML；Python 部分负责读取 tasks.json、校验配置与顺序运行队列。
#
# 预览：bash train_audio.sh --dry-run
# 训练：CUDA_VISIBLE_DEVICES=0,1 bash train_audio.sh --run
# 早停：bash train_audio.sh --run --patience 10 --run-name baseline-patience10-20261027
# 分组：bash train_audio.sh --run --group mental  # 或 --group other
# 权重：bash train_audio.sh --run --weights official  # 或 --weights jepa
# 输出：bash train_audio.sh --run --output-root /有足够空间的目录/finetune_a_compare
#
# 默认：2 卡、每卡 batch=16、100 轮、patience=0（不早停）、关闭 Grokfast。
# 每项任务先 official 再 jepa；各组内部按 tasks.json 顺序。
# 长音频保留 8 句 attention 聚合，短音频保留 4 clips。
# 保留全部 best/latest 预计约需 550GB；请安排输出空间。
# 两个初始化 checkpoint 在整个对照期间应保持不变。
# 默认只预览；不会启动训练或创建输出目录。
# 可用 PYTHON_BIN=/path/to/python 选择训练环境。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${PYTHON_BIN:-python}" -u - "$@" <<'AUDIO_COMPARE_PY'
"""按 tasks.json 顺序运行音频两种初始化对照；默认只校验并打印命令。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

from app.main import apply_overrides, load_config


ROOT = Path.cwd()
TASKS = ROOT / "DATASET/splits-0901/info/tasks.json"
MENTAL = "3_私有临床数据_LV"
CHECKPOINTS = {
    "official": ROOT / "CKPT/emotion2vec_plus_large/model.pt",
    "jepa": Path("/home/data/sdd/mental/OUTPUTS/pretrain_a/emotion2vec-plus-large/FaVoR-2to6s/latest.pt"),
}

# 与历史 IEMOCAP 原始基线对齐；保留每项任务的标签、类型和长/短音频采样。
BASELINE = [
    "meta.load_checkpoint=true", "meta.reset_epoch=true", "meta.frozen_encoder=false",
    "meta.seed=239", "meta.min_load_ratio=0.95", "meta.dtype=bfloat16",
    "model.backbone.mode=emotion2vec", "model.features.mode=last",
    "model.backbone.prenet_dropout=0.0", "model.extractor.dropout=0.0",
    "model.encoder.dropout=0.0", "model.encoder.attention_dropout=0.0",
    "model.encoder.activation_dropout=0.0", "model.encoder.post_mlp_dropout=0.0",
    "model.encoder.dropout_input=0.0", "model.head.dropout=0.1",
    "model.head.attention_dropout=0.0", "optimization.grokfast=false",
    "optimization.epochs=100", "optimization.backbone_lr=2.0e-5",
    "optimization.head_lr=1.0e-3", "optimization.final_lr=1.0e-6",
    "optimization.weight_decay=1.0e-4", "optimization.betas=[0.9,0.999]",
    "optimization.loss.classification.class_weighting=none",
    "optimization.loss.classification.label_smoothing=0.0",
    "optimization.loss.multi_label_classification.positive_weighting=none",
    "data.sample_rate=16000", "data.process_seconds=4.0",
    "data.normalize_waveform=true",
]


def task_entries(group="all"):
    registry = json.loads(TASKS.read_text(encoding="utf-8"))
    # mental 在前，其他数据集在后；各分组内部保留 tasks.json 原始顺序。
    result = []
    for is_mental in (True, False):
        if group != "all" and is_mental != (group == "mental"):
            continue
        for dataset, tasks in registry.items():
            if (dataset == MENTAL) != is_mental:
                continue
            for label, spec in tasks.items():
                dataset_name = re.sub(r"^\d+_|_(?:SA|SV|LA|LV)$", "", dataset)
                name = label if is_mental else f"{dataset_name}-{label}".replace("_", "-")
                section = "mental" if is_mental else {
                    "classification": "classification", "regression": "regression",
                    "multi_label_classification": "multilabel",
                }[spec["type"]]
                config = ROOT / "CONFIGS/tasks/finetune/audio" / section / f"{name}.yaml"
                merged = load_config(str(config))
                assert merged["app"] == "finetune_a", config
                assert merged["data"]["label_column"] == label, config
                assert merged["data"]["task"] == spec["type"], config
                if "class_count" in spec:
                    assert merged["data"]["num_class"] == spec["class_count"], config
                for path in merged["data"]["datasets"]:
                    if not (ROOT / path).is_file():
                        raise FileNotFoundError(f"数据清单不存在：{path}")
                result.append((section, name, config, merged))
    expected = {"all": 71, "mental": 38, "other": 33}[group]
    assert len(result) == expected, (group, len(result))
    return result


def build_jobs(args):
    weights = list(CHECKPOINTS) if args.weights == "both" else [args.weights]
    for weight in weights:
        if not CHECKPOINTS[weight].is_file():
            raise FileNotFoundError(CHECKPOINTS[weight])
    jobs = []
    for section, name, config, merged in task_entries(args.group):
        for weight in weights:
            folder = args.output_root.resolve() / args.run_name / weight / section / name
            overrides = BASELINE + [
                f"meta.read_checkpoint={CHECKPOINTS[weight]}", f"folder={folder}",
                f"data.batch_size={args.batch_size}",
                f"optimization.early_stopping_patience={args.patience}",
            ]
            # 与当前长音频任务保持相同聚合头；两种权重使用完全相同的设置。
            effective = apply_overrides(merged, overrides)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={args.nproc_per_node}", "-m", "app.main",
                       "--fname", str(config.relative_to(ROOT)), "--set", *overrides]
            identity = {"command": command, "config": effective}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            jobs.append({"task": name, "weight": weight, "folder": folder,
                         "command": command, "digest": digest})
    assert len({str(j["folder"]) for j in jobs}) == len(jobs)
    return jobs


def main():
    parser = argparse.ArgumentParser(prog="train_audio.sh", description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="实际顺序训练；默认仅预览")
    mode.add_argument("--dry-run", action="store_true", help="只校验并打印命令")
    parser.add_argument("--weights", choices=["both", "official", "jepa"], default="both")
    parser.add_argument("--group", choices=["all", "mental", "other"], default="all")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16, help="每个 GPU 的 batch size")
    parser.add_argument("--patience", type=int, default=0, help="0=历史基线的完整100轮；10=启用早停")
    parser.add_argument("--run-name", default="baseline-20261027")
    parser.add_argument("--output-root", type=Path,
                        default=Path("/home/data/sdd/mental/OUTPUTS/finetune_a_compare"))
    args = parser.parse_args()
    if min(args.nproc_per_node, args.batch_size) < 1 or args.patience < 0:
        parser.error("卡数和 batch size 必须为正数，patience 必须非负")
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("run-name 必须是单个目录名")
    os.chdir(ROOT)
    jobs = build_jobs(args)
    print(f"共 {len(jobs)} 次训练；严格串行；每个任务先 official 后 jepa（如选 both）。", flush=True)
    print(f"每卡 batch={args.batch_size}；epochs=100；patience={args.patience}；输出={args.output_root / args.run_name}", flush=True)
    if not args.run:
        for i, job in enumerate(jobs, 1):
            print(f"[{i:03}/{len(jobs)}] {job['weight']} {job['task']}\n{shlex.join(job['command'])}")
        return
    # 已成功的同配置任务可跳过；已有但未完成的目录显式拒绝，避免覆盖历史。
    pending = []
    for job in jobs:
        folder = job["folder"]
        marker = folder / "queue_success.json"
        if marker.exists() and json.loads(marker.read_text()).get("digest") == job["digest"]:
            print(f"跳过已完成：{job['weight']} {job['task']}", flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"输出目录已有内容：{folder}；请使用新的 --run-name，避免覆盖。")
        pending.append(job)
    failures = []
    for i, job in enumerate(pending, 1):
        folder = job["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / "launcher.log"
        print(f"[{i}/{len(pending)}] 开始 {job['weight']} {job['task']}；日志 {log_path}", flush=True)
        with log_path.open("w") as log:
            log.write(shlex.join(job["command"]) + "\n")
            log.flush()
            process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                rc = process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if rc:
            failures.append(f"{job['weight']} {job['task']} (exit={rc})")
            print(f"失败：{failures[-1]}，继续下一项；详见 {log_path}", flush=True)
        else:
            (folder / "queue_success.json").write_text(json.dumps({"digest": job["digest"],
                "command": job["command"]}, ensure_ascii=False, indent=2) + "\n")
            print(f"完成：{job['weight']} {job['task']}", flush=True)
    if failures:
        print("失败任务：\n" + "\n".join(failures), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

AUDIO_COMPARE_PY
