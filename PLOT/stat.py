#!/usr/bin/env python3
"""汇总匹配通配符的实验目录的最佳指标。

用法:
    python PLOT/stat.py 'OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-e5'
    python PLOT/stat.py 'OUTPUT/finetune_v/vitl16/RAVDESS-emotion/*' --sort name
    python PLOT/stat.py 'OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-*' \
        --config --csv PLOT/output/stat/summary.csv

可传多个通配符，重复命中的目录只统计一次。目录里没有 logs/history.csv 时仍会输出
一行（标成「无指标」），以免漏看没跑起来的实验。

数据来源是每个实验的 ``{folder}/logs/history.csv``（逐 epoch 的 val_* 指标），
而不是 ``logs/best.csv`` —— 后者按任务类型有三套不同版式（分类是「名称行 + 数值行」
的紧凑转置版式，回归是一行表头 + 一行数值），而 history.csv 三种任务都是同一套扁平列。

best 的判定与 ``app/finetune_v/train.py`` 保持一致：

| task | 主指标 | 方向 | 同 epoch 一并显示的次指标 |
|---|---|---|---|
| `classification` | `val_f1_macro` | 越大越好 | `val_accuracy`、`val_loss` |
| `multi_label_classification` | `val_f1_macro` | 越大越好 | `val_hamming_loss`、`val_loss` |
| `regression` | `val_rmse` | 越小越好 | `val_mae`、`val_r2` |

同一 epoch 在 history.csv 里出现多次（实验重跑过、历史被追加）时取**最后一行**，
并在输出里用 ``dup`` 列标出重复行数。

只依赖标准库 + PyYAML，不导入 torch，训练进行中也能随时查。
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PLOT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PLOT_ROOT / "output" / "stat"

# task -> (主指标, 是否越大越好, 一并显示的次指标及其短标签)
TASK_RULES = {
    "classification": ("val_f1_macro", True, (("val_accuracy", "acc"), ("val_loss", "loss"))),
    "multi_label_classification": (
        "val_f1_macro", True, (("val_hamming_loss", "hamming"), ("val_loss", "loss")),
    ),
    "regression": ("val_rmse", False, (("val_mae", "mae"), ("val_r2", "r2"))),
}
TASK_SHORT = {
    "classification": "cls",
    "multi_label_classification": "mlcls",
    "regression": "reg",
}
# 判断方向用：列名里含这些词的指标越小越好，其余按越大越好
LOWER_IS_BETTER = ("rmse", "mae", "mse", "loss", "hamming")
# 次指标的显示顺序（同时决定 CSV 里 sec_* 列的集合）
SECONDARY_LABELS = ("acc", "hamming", "loss", "mae", "r2")


@dataclass
class Record:
    """一个实验目录的统计结果。"""

    name: str
    folder: Path
    task: str
    task_is_guessed: bool = False
    epochs_seen: int = 0
    duplicates: int = 0
    target_epochs: int | None = None
    best_epoch: int | None = None
    main_metric: str | None = None
    main_value: float | None = None
    secondary: dict[str, float] = field(default_factory=dict)
    config: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def higher_is_better(self) -> bool:
        return not any(token in (self.main_metric or "") for token in LOWER_IS_BETTER)


def compile_pattern(pattern: str) -> tuple[re.Pattern, int]:
    """把 glob 编译成正则，每个通配符变成一个命名组，用于给结果列命名。"""
    parts: list[str] = []
    index = 0
    count = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**", index):
            count += 1
            parts.append(f"(?P<g{count}>.*)")
            index += 2
        elif char == "*":
            count += 1
            parts.append(f"(?P<g{count}>[^/]*)")
            index += 1
        elif char == "?":
            count += 1
            parts.append(f"(?P<g{count}>[^/])")
            index += 1
        elif char == "[":
            end = pattern.find("]", index)
            if end == -1:
                parts.append(re.escape(char))
                index += 1
            else:
                count += 1
                parts.append(f"(?P<g{count}>{pattern[index:end + 1]})")
                index = end + 1
        else:
            parts.append(re.escape(char))
            index += 1
    return re.compile("^" + "".join(parts) + "$"), count


def matched_name(pattern: str, folder: Path) -> str:
    """用通配符捕获到的部分作为实验名；没有通配符时退回目录名。

    例 ``OUTPUT/finetune_v/vitl16/*/FaVoR-...-e5`` 命中
    ``OUTPUT/.../RAVDESS-emotion/FaVoR-...-e5`` 时，实验名就是 ``RAVDESS-emotion``。
    """
    regex, count = compile_pattern(pattern)
    match = regex.match(str(folder))
    if not match or count == 0:
        return folder.name
    captured = [match.group(f"g{i}") for i in range(1, count + 1)]
    return " / ".join(part or "." for part in captured)


def read_history(path: Path) -> tuple[dict[int, dict[str, float]], int, list[str]]:
    """读 ``logs/history.csv``，返回 ({epoch: {val_* 指标}}, 重复行数, 表头列名)。

    同一 epoch 出现多次时后一行覆盖前一行（重跑后的结果更可信）；解析不了的行直接
    跳过，因此训练正在写文件时读到半行也不会崩。
    """
    rows: dict[int, dict[str, float]] = {}
    duplicates = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for raw in reader:
            try:
                epoch = int(float(raw.get("epoch") or ""))
            except (TypeError, ValueError):
                continue
            metrics: dict[str, float] = {}
            for key, value in raw.items():
                if not key or not key.startswith("val_"):
                    continue
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if not metrics:
                continue
            if epoch in rows:
                duplicates += 1
            rows[epoch] = metrics
    return rows, duplicates, fieldnames


def read_params(folder: Path) -> dict:
    """读 ``{folder}/params-*.yaml``（启动时自动写出的合并后参数快照）。"""
    for path in sorted(folder.glob("params-*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def config_summary(params: dict) -> dict[str, str]:
    """从参数快照里挑出区分实验最常用的几列。"""
    data = params.get("data") or {}
    meta = params.get("meta") or {}
    opt = params.get("optimization") or {}

    frames = ""
    fpcs = data.get("dataset_fpcs")
    fps = data.get("fps")
    if isinstance(fpcs, (list, tuple)) and fpcs:
        frames = "/".join(str(item) for item in fpcs) + "f"
    if fps:
        frames = f"{frames}@{fps}fps" if frames else f"@{fps}fps"

    ckpt = meta.get("read_checkpoint")
    learning_rate = opt.get("lr")
    return {
        "ckpt": Path(str(ckpt)).name if ckpt else "-",
        "seed": str(meta.get("seed", "-")),
        "frames": frames or "-",
        "epochs": str(opt.get("epochs", "-")),
        "lr": f"{learning_rate:g}" if isinstance(learning_rate, (int, float)) else "-",
    }


def as_epoch_count(value) -> int | None:
    """参数快照里的 epochs 可能是 int 或 "80" 这样的字符串。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def pick_best(rows: dict[int, dict[str, float]], metric: str, higher: bool):
    """返回 (最优值, 对应 epoch)；该指标在所有 epoch 上都不存在时返回 (None, None)。"""
    candidates = [(entry[metric], epoch) for epoch, entry in rows.items() if metric in entry]
    if not candidates:
        return None, None
    return (max if higher else min)(candidates)


def collect(pattern: str, folder: Path, metric_override: str | None) -> Record:
    """统计单个实验目录。"""
    params = read_params(folder)
    data = params.get("data") or {}
    opt = params.get("optimization") or {}

    task = str(data.get("task") or "").strip()
    task_is_guessed = not task

    rows: dict[int, dict[str, float]] = {}
    duplicates = 0
    fieldnames: list[str] = []
    history = folder / "logs" / "history.csv"
    if history.is_file():
        rows, duplicates, fieldnames = read_history(history)

    if not task:
        # 参数快照缺失时才靠指标列反推（cls 有 val_balanced_accuracy，mlcls 没有）
        if not fieldnames:
            task = "unknown"
        elif "val_rmse" in fieldnames:
            task = "regression"
        elif "val_hamming_loss" in fieldnames and "val_balanced_accuracy" not in fieldnames:
            task = "multi_label_classification"
        else:
            task = "classification"

    record = Record(
        name=matched_name(pattern, folder),
        folder=folder,
        task=task,
        task_is_guessed=task_is_guessed,
        epochs_seen=len(rows),
        duplicates=duplicates,
        target_epochs=as_epoch_count(opt.get("epochs")),
        config=config_summary(params),
    )

    if not fieldnames:
        record.note = "无 logs/history.csv"
        return record
    if not rows:
        record.note = "history.csv 无可解析的指标行"
        return record

    default_metric, _default_higher, secondary_spec = TASK_RULES.get(task, ("val_f1_macro", True, ()))
    metric = metric_override or default_metric
    record.main_metric = metric
    value, epoch = pick_best(rows, metric, not any(t in metric for t in LOWER_IS_BETTER))
    if value is None:
        available = sorted({key for entry in rows.values() for key in entry})
        record.note = f"无 {metric} 列（可用: {', '.join(available[:5])} …）" if available else f"无 {metric} 列"
        return record

    record.main_value, record.best_epoch = value, epoch
    for source, label in secondary_spec:
        if source in rows[epoch]:
            record.secondary[label] = rows[epoch][source]
    return record


def expand(patterns: list[str]) -> list[tuple[str, Path]]:
    """展开所有通配符并去重，返回 [(展开后的通配符, 命中的目录)]。"""
    found: dict[Path, tuple[str, Path]] = {}
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser()).rstrip("/") or "/"
        for hit in glob.glob(expanded, recursive=True):
            path = Path(hit)
            if path.is_dir():
                found.setdefault(path.resolve(), (expanded, path))
    return [pair for _, pair in sorted(found.items(), key=lambda item: str(item[0]))]


def rank_key(record: Record) -> tuple[int, float]:
    """按主指标名次排序的键；没有主指标（缺 history.csv 等）的一律排在最后。"""
    if record.main_value is None:
        return (1, 0.0)
    return (0, -record.main_value if record.higher_is_better else record.main_value)


def sort_records(records: list[Record], mode: str) -> list[Record]:
    """metric（默认）：先任务类型、再主指标名次（回归由小到大，其余由大到小）。"""
    if mode == "name":
        return sorted(records, key=lambda record: (record.task, record.name))
    groups: dict[str, list[Record]] = {}
    for record in records:
        groups.setdefault(record.task, []).append(record)
    ordered: list[Record] = []
    for task in sorted(groups):
        ordered.extend(sorted(groups[task], key=rank_key))
    return ordered


def record_row(record: Record, show_config: bool) -> list[str]:
    """把一条记录摊平成表格行。"""
    secondary = ", ".join(
        f"{label} {record.secondary[label]:.4f}"
        for label in SECONDARY_LABELS if label in record.secondary
    )
    progress = str(record.epochs_seen)
    if record.target_epochs:
        progress = f"{record.epochs_seen}/{record.target_epochs}"
        if record.epochs_seen < record.target_epochs:
            progress += " (未跑完)"
    row = [
        record.name,
        TASK_SHORT.get(record.task, record.task) + ("?" if record.task_is_guessed else ""),
        str(record.best_epoch) if record.best_epoch is not None else "-",
        (record.main_metric or "-").replace("val_", ""),
        f"{record.main_value:.4f}" if record.main_value is not None else "-",
        secondary or "-",
    ]
    if show_config:
        row += [record.config.get(key, "-") for key in ("ckpt", "seed", "frames", "epochs", "lr")]
    row += [progress, str(record.duplicates) if record.duplicates else "", record.note]
    return row


def render(records: list[Record], show_config: bool, sort_mode: str) -> None:
    """打印对齐的文本表（表头全 ASCII，避免中英混排的宽度错位）。"""
    header = ["experiment", "task", "best_ep", "main_metric", "main_value", "secondary"]
    if show_config:
        header += ["ckpt", "seed", "frames", "epochs", "lr"]
    header += ["progress", "dup", "note"]

    table = [record_row(record, show_config) for record in records]
    widths = [
        max([len(header[index])] + [len(row[index]) for row in table])
        for index in range(len(header))
    ]
    print("  ".join(name.ljust(widths[index]) for index, name in enumerate(header)).rstrip())
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip())

    counts: dict[str, int] = {}
    for record in records:
        counts[record.task] = counts.get(record.task, 0) + 1
    print(f"\n共 {len(records)} 个实验：" +
          " / ".join(f"{task} {count}" for task, count in sorted(counts.items())))
    print("主指标: f1_macro 越大越好 / rmse 越小越好（--metric 可换成任意 val_* 列）")
    unfinished = [r for r in records if r.target_epochs and r.epochs_seen < r.target_epochs]
    empty = [r for r in records if r.main_value is None]
    restarted = [r for r in records if r.duplicates]
    if unfinished:
        print("未跑完: " + ", ".join(f"{r.name} ({r.epochs_seen}/{r.target_epochs})" for r in unfinished))
    if empty:
        print("无主指标: " + ", ".join(f"{r.name}（{r.note}）" for r in empty))
    if restarted:
        print("有重复 epoch 行（实验重跑过，已取同一 epoch 的最后一行）: "
              + ", ".join(f"{r.name} (+{r.duplicates})" for r in restarted))
    if sort_mode == "metric":
        print("排序: 先任务类型、再主指标名次（--sort name 改为按名字）")


def write_csv(records: list[Record], path: Path, show_config: bool) -> None:
    """把同一张表写成 CSV，列比终端多出 folder 与 epoch 明细。"""
    fieldnames = ["experiment", "task", "best_epoch", "main_metric", "main_value"]
    fieldnames += [f"sec_{label}" for label in SECONDARY_LABELS]
    if show_config:
        fieldnames += ["ckpt", "seed", "frames", "epochs", "lr"]
    fieldnames += ["epochs_seen", "target_epochs", "duplicate_epochs", "note", "folder"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {
                "experiment": record.name,
                "task": record.task,
                "best_epoch": record.best_epoch,
                "main_metric": record.main_metric,
                "main_value": record.main_value,
                "epochs_seen": record.epochs_seen,
                "target_epochs": record.target_epochs,
                "duplicate_epochs": record.duplicates,
                "note": record.note,
                "folder": str(record.folder),
            }
            if show_config:
                row.update(record.config)
            for label in SECONDARY_LABELS:
                row[f"sec_{label}"] = record.secondary.get(label)
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="汇总匹配通配符的实验目录的最佳指标（数据源: 各实验的 logs/history.csv）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python PLOT/stat.py 'OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-e5'\n"
            "  python PLOT/stat.py 'OUTPUT/finetune_v/vitl16/RAVDESS-emotion/*' --sort name\n"
            "  python PLOT/stat.py 'OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-*' "
            "--config --csv PLOT/output/stat/summary.csv\n"
        ),
    )
    parser.add_argument("patterns", nargs="+", metavar="GLOB",
                        help="实验目录通配符，可给多个，如 'OUTPUT/.../vitl16/*/FaVoR-112px-48f-8fps-e5'")
    parser.add_argument("--sort", choices=("metric", "name"), default="metric",
                        help="metric（默认）= 先任务类型再主指标名次；name = 按任务与名字排序")
    parser.add_argument("--metric", default=None, metavar="COLUMN",
                        help="覆盖主指标列名，如 val_accuracy；列名含 rmse/mae/loss 的按越小越好")
    parser.add_argument("--config", action="store_true",
                        help="额外显示参数快照里的 ckpt/seed/frames/epochs/lr 列")
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT / "summary.csv", metavar="PATH",
                        help=f"把结果同时写成 CSV（默认 {DEFAULT_OUTPUT / 'summary.csv'}）")
    args = parser.parse_args()

    matched = expand(args.patterns)
    if not matched:
        print("没有目录匹配这些通配符: " + ", ".join(args.patterns), file=sys.stderr)
        print("提示: 通配符从当前目录算起；先确认目录存在，或改用绝对路径。", file=sys.stderr)
        return 2

    records = [collect(pattern, folder, args.metric) for pattern, folder in matched]
    ordered = sort_records(records, args.sort)
    render(ordered, args.config, args.sort)
    if args.csv:
        write_csv(ordered, args.csv, args.config)
        print(f"CSV 已写入: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
