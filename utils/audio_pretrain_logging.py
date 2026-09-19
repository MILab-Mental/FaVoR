"""Persistent metrics and plots for audio JEPA pre-training."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path


HISTORY_FIELDS = (
    "step", "epoch", "loss", "context_ratio", "target_ratio", "pad_ratio",
    "target_std", "lr_pretrained", "lr_new", "ema_decay", "grad_norm",
    "step_time", "samples_per_second",
)


def load_audio_pretrain_history(path, max_step=None):
    path = Path(path)
    if not path.is_file():
        return []
    rows_by_step = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                row = {
                    field: int(raw[field]) if field in {"step", "epoch"} else float(raw[field])
                    for field in HISTORY_FIELDS
                }
            except (KeyError, TypeError, ValueError):
                continue
            if max_step is None or row["step"] <= int(max_step):
                rows_by_step[row["step"]] = row
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def write_audio_pretrain_history(path, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row[field] for field in HISTORY_FIELDS})
    os.replace(temporary, path)


def save_audio_pretrain_curves(history, path):
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in history]
    figure, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)

    def line(axis, fields, title, *, log_y=False):
        for field in fields:
            values = [row[field] for row in history]
            axis.plot(steps, values, label=field)
        axis.set_title(title)
        axis.set_xlabel("step")
        axis.grid(alpha=0.3)
        if len(fields) > 1:
            axis.legend()
        if log_y and any(
            math.isfinite(row[field]) and row[field] > 0
            for row in history for field in fields
        ):
            axis.set_yscale("log")

    line(axes[0, 0], ("loss",), "JEPA loss")
    line(axes[0, 1], ("context_ratio", "target_ratio", "pad_ratio"), "Mask and padding ratios")
    line(axes[1, 0], ("target_std",), "Target standard deviation")
    line(axes[1, 1], ("lr_pretrained", "lr_new"), "Learning rates", log_y=True)
    line(axes[2, 0], ("ema_decay",), "EMA decay")
    line(axes[2, 1], ("grad_norm",), "Gradient norm", log_y=True)
    line(axes[3, 0], ("step_time",), "Seconds per optimizer step")
    line(axes[3, 1], ("samples_per_second",), "Global audio crops per second")
    figure.savefig(path, dpi=160)
    plt.close(figure)
