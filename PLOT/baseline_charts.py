#!/usr/bin/env python3
"""Generate three comparison charts from 对比模型结果.csv for 13 Mental Core + 15 Ordinary tasks:
  1. 1_聚合柱状图.png: 3x3 cross-task mean bar chart (Mental Core Cls, Ordinary Cls, Ordinary Reg) x 3 metrics
  2. 2_任务模型热力图.png: 35 models x 28 tasks task-normalized heatmap with raw primary metric annotations
  3. 3_逐任务精选柱状图.png: 4x7 per-task horizontal bar chart (FaVoR V/A/VA + Top-6 baselines per task)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.finetune_catalog import mental_core_tasks, read_tasks

DEFAULT_CSV = ROOT / "OUTPUT/figs/结果图表/对比模型结果.csv"
DEFAULT_OUT_DIR = ROOT / "OUTPUT/figs/对比图"

# Unified color.md palette
COLOR_FAVOR = {"VA": "#ED66A4", "A": "#8690C2", "V": "#F59E42"}
COLOR_FAVOR_EDGE = {"VA": "#C94F8B", "A": "#6673A8", "V": "#D17E32"}
COLOR_BASE = {"VA": "#8CC7A1", "A": "#9FA8D1", "V": "#C4B080"}
COLOR_BASE_EDGE = {"VA": "#6FA987", "A": "#6673A8", "V": "#A98F5C"}

MODALITY_BY_MODEL = {
    "AudioMAE": "A",
    "AudioMAE Large (Domain Adaption from Scratch)": "A",
    "wav2vec 2.0": "A",
    "WavLM": "A",
    "HuBERT": "A",
    "WavJEPA": "A",
    "WavJEPA (Domain Adaption)": "A",
    "Whisper": "A",
    "data2vec": "A",
    "BEATs": "A",
    "EAT": "A",
    "EAT (Domain Adaption)": "A",
    "Emotion2Vec": "A",
    "FaVoR (A)": "A",
    "CLIP": "V",
    "DINOv2": "V",
    "DINOv3": "V",
    "VideoMAE": "V",
    "VideoMAE (Domain Adaption)": "V",
    "Hiera": "V",
    "MAE-DFER": "V",
    "V-JEPA 2.1": "V",
    "V-JEPA 2": "V",
    "MaskFeat": "V",
    "MaskFeat (Domain Adaption)": "V",
    "FaVoR (V)": "V",
    "CAV-MAE Scale++": "VA",
    "CAV-MAE Scale++ (Domain Adaption)": "VA",
    "AVSiam": "VA",
    "AVSiam Large (Domain Adaption from Scratch)": "VA",
    "LanguageBind": "VA",
    "AV-HuBERT": "VA",
    "LeAVJEPA (Domain Adaption from Scratch)": "VA",
    "Social-MAE (Domain Adaption from Scratch)": "VA",
    "FaVoR (VA)": "VA",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#B8BEC8",
        "axes.linewidth": 0.8,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "text.color": "#1F2937",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def short_model_name(name: str) -> str:
    return (
        name.replace(" (Domain Adaption from Scratch)", " (DA-Scratch)")
        .replace(" (Domain Adaption)", " (DA)")
    )


def short_task_name(task: str) -> str:
    return (
        task.replace("DSMxSCL-", "DxS-")
        .replace("DSM-", "DSM-")
        .replace("RESULT-", "RES-")
        .replace("PersonalityInterpersonal", "PersInterp")
        .replace("ObsessiveCompulsive", "OCD")
        .replace("SomaticSymptoms", "Somatic")
        .replace("SuicidalIdeation", "Suicidal")
        .replace("SleepProblems", "Sleep")
        .replace("AngerHostility", "Anger")
        .replace("MER242526-", "MER24-")
        .replace("MER2023-", "MER23-")
        .replace("EmotionTalk-", "EmoTalk-")
        .replace("pos-intensity", "pos-int")
        .replace("intensity", "int")
        .replace("emotion", "emo")
        .replace("activation", "act")
        .replace("dominance", "dom")
        .replace("valence", "val")
        .replace("26openset", "26open")
    )


def parse_cell(cell: str) -> tuple[float, float, float] | None:
    cleaned = (cell or "").strip().strip("*")
    if not cleaned or cleaned in ("-/-/-", "～/～/～"):
        return None
    parts = cleaned.split("/")
    if len(parts) != 3:
        return None
    try:
        vals = tuple(float(p) for p in parts)
        if any(not np.isfinite(v) for v in vals):
            return None
        return vals  # type: ignore[return-value]
    except ValueError:
        return None


def load_data(csv_path: Path):
    task_rows = read_tasks(ROOT)
    task_meta = {r["task_id"]: r for r in task_rows}
    core_list = mental_core_tasks(ROOT)

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = [c for c in (reader.fieldnames or []) if c != "model"]
        rows = list(reader)

    ordinary_cls = [
        c for c in columns
        if c in task_meta and task_meta[c]["scope"] != "mental" and task_meta[c]["task_type"] != "regression"
    ]
    ordinary_reg = [
        c for c in columns
        if c in task_meta and task_meta[c]["scope"] != "mental" and task_meta[c]["task_type"] == "regression"
    ]
    mental_core = [c for c in core_list if c in columns]

    models: list[str] = []
    values: dict[tuple[str, str], tuple[float, float, float]] = {}
    for r in rows:
        model = r["model"].strip()
        models.append(model)
        for col in columns:
            parsed = parse_cell(r.get(col, ""))
            if parsed is not None:
                values[(model, col)] = parsed

    return models, values, task_meta, mental_core, ordinary_cls, ordinary_reg


def model_style(model: str) -> tuple[str, str, bool]:
    mod = MODALITY_BY_MODEL.get(model, "V")
    is_favor = model.startswith("FaVoR")
    if is_favor:
        return COLOR_FAVOR[mod], "black", True
    return COLOR_BASE[mod], COLOR_BASE_EDGE[mod], False


def plot_1_aggregate_bars(
    models: list[str],
    values: dict[tuple[str, str], tuple[float, float, float]],
    mental_core: list[str],
    ordinary_cls: list[str],
    ordinary_reg: list[str],
    out_path: Path,
) -> None:
    groups = [
        ("Mental Core · Classification", mental_core, [("Acc (%)", 0, False), ("Macro-F1 (%)", 1, False), ("AUC", 2, False)], 1, False),
        ("Ordinary · Classification", ordinary_cls, [("Acc (%)", 0, False), ("Macro-F1 (%)", 1, False), ("AUC", 2, False)], 1, False),
        ("Ordinary · Regression", ordinary_reg, [("MAE (↓)", 0, True), ("RMSE (↓)", 1, True), ("R² (↑)", 2, False)], 0, True),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(33, 19), gridspec_kw={"wspace": 0.14, "hspace": 0.52})

    for row_idx, (group_title, tasks, metrics_spec, sort_idx, sort_asc) in enumerate(groups):
        n_tasks = len(tasks)
        records = []
        for m in models:
            triples = [values[(m, t)] for t in tasks if (m, t) in values]
            if not triples:
                continue
            arr = np.array(triples, dtype=float)
            means = arr.mean(axis=0)
            sems = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(3)
            records.append((m, len(arr), means, sems))

        # Best on the left -> right
        records.sort(key=lambda r: (r[2][sort_idx] if sort_asc else -r[2][sort_idx]))
        x = np.arange(len(records))

        for col_idx, (metric_label, m_idx, lower_better) in enumerate(metrics_spec):
            ax = axes[row_idx, col_idx]
            means = np.array([r[2][m_idx] for r in records])
            sems = np.array([r[3][m_idx] for r in records])
            colors = [model_style(r[0])[0] for r in records]
            edges = [model_style(r[0])[1] for r in records]
            lws = [0.7 if model_style(r[0])[2] else 0.6 for r in records]

            bars = ax.bar(x, means, width=0.72, color=colors, edgecolor=edges, linewidth=lws, alpha=0.92, zorder=3)
            ax.errorbar(
                x, means, yerr=sems, fmt="none", ecolor="#6B7280", elinewidth=0.75, capsize=1.8, alpha=0.75, zorder=4
            )

            ax.set_xticks(x)
            labels = [f"{short_model_name(r[0])} ({r[1]}/{n_tasks})" for r in records]
            ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=7.6)
            for tick, r in zip(ax.get_xticklabels(), records):
                if r[0].startswith("FaVoR"):
                    tick.set_fontweight("bold")
                    tick.set_color("#111827")

            ymin = float((means - sems).min())
            ymax = float((means + sems).max())
            span = max(ymax - ymin, 1e-4)
            margin = max(span * 0.18, 0.04)
            if ymin > 0 and row_idx < 2 and m_idx < 2:
                ax.set_ylim(max(0.0, ymin - margin * 0.6), ymax + margin * 1.4)
            elif ymin > 0:
                ax.set_ylim(max(0.0, ymin - margin * 0.4), ymax + margin * 1.45)
            else:
                ax.set_ylim(ymin - margin * 1.35, ymax + margin * 1.45)
                ax.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=2)

            pad = span * 0.03
            for bar, mean, sem, r in zip(bars, means, sems, records):
                txt = f"{mean:.2f}" if (m_idx == 2 or row_idx == 2) else f"{mean:.1f}"
                ypos = mean + sem + pad if mean >= 0 else mean - sem - pad
                va = "bottom" if mean >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, ypos, txt,
                    ha="center", va=va, fontsize=6.6, rotation=90 if len(records) > 18 else 0,
                    fontweight="bold" if r[0].startswith("FaVoR") else "normal",
                    color="#111827" if r[0].startswith("FaVoR") else "#4B5563",
                )

            ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.6, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(f"{group_title} ({n_tasks}) — {metric_label}", fontsize=11.5, fontweight="bold", pad=8)

    legend_handles = [
        Patch(facecolor=COLOR_FAVOR["VA"], edgecolor="black", linewidth=0.7, label="FaVoR (VA)"),
        Patch(facecolor=COLOR_FAVOR["V"], edgecolor="black", linewidth=0.7, label="FaVoR (V)"),
        Patch(facecolor=COLOR_FAVOR["A"], edgecolor="black", linewidth=0.7, label="FaVoR (A)"),
        Patch(facecolor=COLOR_BASE["VA"], edgecolor=COLOR_BASE_EDGE["VA"], linewidth=0.8, label="Baseline (VA)"),
        Patch(facecolor=COLOR_BASE["V"], edgecolor=COLOR_BASE_EDGE["V"], linewidth=0.8, label="Baseline (V)"),
        Patch(facecolor=COLOR_BASE["A"], edgecolor=COLOR_BASE_EDGE["A"], linewidth=0.8, label="Baseline (A)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=6, frameon=False, fontsize=11.5)
    fig.suptitle("Cross-Task Aggregated Comparison (Mean ± SEM across completed tasks)", fontsize=16, fontweight="bold", y=0.972)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_2_heatmap(
    models: list[str],
    values: dict[tuple[str, str], tuple[float, float, float]],
    task_meta: dict[str, dict],
    mental_core: list[str],
    ordinary_cls: list[str],
    ordinary_reg: list[str],
    out_path: Path,
) -> None:
    tasks = mental_core + ordinary_cls + ordinary_reg
    # Order models by modality group (VA, V, A), putting FaVoR first within each group, and dropping all-empty DA rows only? No, keep all models with >=1 result, plus note empty ones
    active_models = [m for m in models if any((m, t) in values for t in tasks)]
    mod_rank = {"VA": 0, "V": 1, "A": 2}
    active_models.sort(key=lambda m: (mod_rank.get(MODALITY_BY_MODEL.get(m, "V"), 3), 0 if m.startswith("FaVoR") else 1, models.index(m)))

    n_m, n_t = len(active_models), len(tasks)
    raw = np.full((n_m, n_t), np.nan)
    norm = np.full((n_m, n_t), np.nan)
    is_reg = [task_meta[t]["task_type"] == "regression" for t in tasks]

    for j, t in enumerate(tasks):
        idx = 0 if is_reg[j] else 1  # MAE for regression, Macro-F1 for classification
        for i, m in enumerate(active_models):
            if (m, t) in values:
                raw[i, j] = values[(m, t)][idx]
        col = raw[:, j]
        valid = np.isfinite(col)
        if valid.any():
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
            if hi > lo:
                frac = (col - lo) / (hi - lo)
                if is_reg[j]:
                    frac = 1.0 - frac
                norm[:, j] = frac
            else:
                norm[valid, j] = 1.0

    # Soft academic colormap from color.md (#F4F5F7 -> #C8D9E6 -> #8CC7A1 -> #ED66A4? Better: sequential soft slate-mint-rose or soft neutral -> #8690C2 -> #ED66A4)
    cmap = LinearSegmentedColormap.from_list("favor_hm", ["#F3F4F6", "#D6DCEB", "#8690C2", "#ED66A4"])
    cmap.set_bad(color="#FAFAFA")

    fig, ax = plt.subplots(figsize=(28, 14.5))
    im = ax.imshow(norm, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    # Cell annotations
    for j, t in enumerate(tasks):
        col = raw[:, j]
        valid = np.isfinite(col)
        best_val = (np.nanmin(col) if is_reg[j] else np.nanmax(col)) if valid.any() else np.nan
        for i in range(n_m):
            v = raw[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6.5, color="#D1D5DB")
                continue
            txt = f"{v:.2f}" if is_reg[j] else f"{v:.1f}"
            is_best = np.isclose(v, best_val, rtol=1e-4, atol=1e-4)
            score = norm[i, j]
            fg = "white" if score > 0.72 else "#1F2937"
            ax.text(
                j, i, txt, ha="center", va="center",
                fontsize=6.8, fontweight="bold" if is_best else "normal", color=fg,
            )

    # Ticks and group dividers
    ax.set_xticks(np.arange(n_t))
    ax.set_xticklabels([short_task_name(t) for t in tasks], rotation=42, ha="right", fontsize=8.5)
    for tick, reg in zip(ax.get_xticklabels(), is_reg):
        tick.set_color("#16723C" if reg else "#1F2937")

    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels([f"[{MODALITY_BY_MODEL.get(m, '?')}] {short_model_name(m)}" for m in active_models], fontsize=8.8)
    for tick, m in zip(ax.get_yticklabels(), active_models):
        if m.startswith("FaVoR"):
            tick.set_fontweight("bold")
            tick.set_color(COLOR_FAVOR_EDGE[MODALITY_BY_MODEL[m]])

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_t, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_m, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Vertical section dividers
    ax.axvline(len(mental_core) - 0.5, color="#374151", linewidth=2.0)
    ax.axvline(len(mental_core) + len(ordinary_cls) - 0.5, color="#374151", linewidth=1.5, linestyle="--")

    # Horizontal modality dividers
    prev_mod = MODALITY_BY_MODEL.get(active_models[0])
    for i, m in enumerate(active_models[1:], start=1):
        cur_mod = MODALITY_BY_MODEL.get(m)
        if cur_mod != prev_mod:
            ax.axhline(i - 0.5, color="#374151", linewidth=1.6)
            prev_mod = cur_mod

    # Section top headers
    sec_centers = [
        (len(mental_core) / 2 - 0.5, f"Mental Core ({len(mental_core)} tasks · Macro-F1 ↑)"),
        (len(mental_core) + len(ordinary_cls) / 2 - 0.5, f"Ordinary Cls ({len(ordinary_cls)} tasks · Macro-F1 ↑)"),
        (len(mental_core) + len(ordinary_cls) + len(ordinary_reg) / 2 - 0.5, f"Ordinary Reg ({len(ordinary_reg)} tasks · MAE ↓)"),
    ]
    for cx, title in sec_centers:
        ax.text(cx, -0.85, title, ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#111827")

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.012)
    cbar.set_label("Task-Normalized Score (0 = worst, 1 = best)", fontsize=10)
    ax.set_title("Model × Task Performance Matrix (Bold = Task Best; Green labels = Regression MAE ↓)", fontsize=14, fontweight="bold", pad=46)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_3_per_task_topk(
    models: list[str],
    values: dict[tuple[str, str], tuple[float, float, float]],
    task_meta: dict[str, dict],
    mental_core: list[str],
    ordinary_cls: list[str],
    ordinary_reg: list[str],
    out_path: Path,
    top_k: int = 6,
) -> None:
    tasks = mental_core + ordinary_cls + ordinary_reg
    ncols = 7
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(35, 5.1 * nrows), gridspec_kw={"wspace": 0.24, "hspace": 0.62})
    axes_flat = axes.ravel()

    favor_names = ["FaVoR (VA)", "FaVoR (V)", "FaVoR (A)"]

    for idx_t, task in enumerate(tasks):
        ax = axes_flat[idx_t]
        is_reg = task_meta[task]["task_type"] == "regression"
        m_idx = 0 if is_reg else 1
        metric_name = "MAE ↓" if is_reg else "F1 ↑"

        avail = [(m, values[(m, task)][m_idx]) for m in models if (m, task) in values]
        baselines = [item for item in avail if not item[0].startswith("FaVoR")]
        baselines.sort(key=lambda x: x[1] if is_reg else -x[1])
        chosen = baselines[:top_k]
        for fm in favor_names:
            if (fm, task) in values:
                chosen.append((fm, values[(fm, task)][m_idx]))

        # Best on the left -> right
        chosen.sort(key=lambda x: x[1] if is_reg else -x[1])
        x = np.arange(len(chosen))
        vals = np.array([v for _, v in chosen])
        colors = [model_style(m)[0] for m, _ in chosen]
        edges = [model_style(m)[1] for m, _ in chosen]
        lws = [0.7 if model_style(m)[2] else 0.6 for m, _ in chosen]

        bars = ax.bar(x, vals, width=0.68, color=colors, edgecolor=edges, linewidth=lws, alpha=0.93, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([short_model_name(m) for m, _ in chosen], rotation=40, ha="right", fontsize=7.4)
        for tick, (m, _) in zip(ax.get_xticklabels(), chosen):
            if m.startswith("FaVoR"):
                tick.set_fontweight("bold")
                tick.set_color("#111827")

        vmin, vmax = float(vals.min()), float(vals.max())
        span = max(vmax - vmin, 1e-4)
        lo = max(0.0, vmin - span * 0.25) if vmin >= 0 else vmin - span * 0.25
        hi = vmax + span * 0.34
        ax.set_ylim(lo, hi)

        for bar, (m, v) in zip(bars, chosen):
            txt = f"{v:.2f}" if is_reg else f"{v:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + span * 0.03, txt,
                ha="center", va="bottom", fontsize=7.0,
                fontweight="bold" if m.startswith("FaVoR") else "normal",
                color="#111827" if m.startswith("FaVoR") else "#4B5563",
            )

        scope_tag = "Core" if task in mental_core else ("Reg" if is_reg else "Cls")
        title_color = "#16723C" if is_reg else ("#C94F8B" if scope_tag == "Core" else "#1F2937")
        ax.set_title(f"{short_task_name(task)} ({metric_name})", fontsize=9.5, fontweight="bold", color=title_color, pad=6)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.2)

    for extra_ax in axes_flat[len(tasks):]:
        extra_ax.set_visible(False)

    legend_handles = [
        Patch(facecolor=COLOR_FAVOR["VA"], edgecolor="black", linewidth=0.7, label="FaVoR (VA)"),
        Patch(facecolor=COLOR_FAVOR["V"], edgecolor="black", linewidth=0.7, label="FaVoR (V)"),
        Patch(facecolor=COLOR_FAVOR["A"], edgecolor="black", linewidth=0.7, label="FaVoR (A)"),
        Patch(facecolor=COLOR_BASE["VA"], edgecolor=COLOR_BASE_EDGE["VA"], linewidth=0.8, label="Baseline (VA)"),
        Patch(facecolor=COLOR_BASE["V"], edgecolor=COLOR_BASE_EDGE["V"], linewidth=0.8, label="Baseline (V)"),
        Patch(facecolor=COLOR_BASE["A"], edgecolor=COLOR_BASE_EDGE["A"], linewidth=0.8, label="Baseline (A)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=6, frameon=False, fontsize=12)
    fig.suptitle(
        f"Per-Task Comparison: FaVoR (VA/V/A) vs Top-{top_k} Baselines — 13 Mental Core + 15 Ordinary Tasks",
        fontsize=17, fontweight="bold", y=0.972,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成对比模型三种汇总图（聚合柱状图、任务模型热力图、逐任务Top-K柱状图）")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    setup_style()
    models, values, task_meta, mental_core, ordinary_cls, ordinary_reg = load_data(args.csv)
    p1 = args.out_dir / "1_聚合柱状图.png"
    p2 = args.out_dir / "2_任务模型热力图.png"
    p3 = args.out_dir / "3_逐任务精选柱状图.png"
    plot_1_aggregate_bars(models, values, mental_core, ordinary_cls, ordinary_reg, p1)
    print("wrote", p1)
    plot_2_heatmap(models, values, task_meta, mental_core, ordinary_cls, ordinary_reg, p2)
    print("wrote", p2)
    plot_3_per_task_topk(models, values, task_meta, mental_core, ordinary_cls, ordinary_reg, p3, top_k=args.top_k)
    print("wrote", p3)


if __name__ == "__main__":
    main()
