#!/usr/bin/env python3
"""绘制跨任务、多配置的论文风格平滑雷达图（SVG）。

每个任务目录占一根轴，按任务类型选择主指标：
  * classification / multi_label_classification: val_f1_macro（越大越好）
  * regression: val_rmse（越小越好）

各轴在给定配置中单独归一化，因此该任务的最优值一定位于 100% 外圈。
归一化只用于几何位置，图上标注的仍是原始指标值。

对比模型八面板（分类 Macro-F1、回归 MAE；直线连接相邻有效节点）：
    python PLOT/增加radar.py --baselines

示例：
    python PLOT/增加radar.py \
      --task OUTPUT/finetune_v/vitl16 \
      --configs FaVoR-112px-48f-8fps-e5 FaVoR-112px-48f-8fps-vjepaori

可以限定任务及缩短图例名：
    python PLOT/增加radar.py --task OUTPUT/finetune_v/vitl16 \
      --configs FaVoR-112px-48f-8fps-e5 FaVoR-112px-48f-8fps-vjepaori \
      --labels FaVoR "V-JEPA 2.1" \
      --tasks RAVDESS-emotion MER2023-emotion MER242526-26openset \
              MER2023-pos_intensity \
      --out OUTPUT/figs/雷达图/main.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO33_RADAR_NAME = "audio33雷达图.png"
RESULT_CSV_NAME = "对比模型结果.csv"
FAVOR_CSV_NAME = "FAVOR结果.csv"

MODEL_DISPLAY_NAMES = {
    "audiomae-base": "AudioMAE",
    "audiomae-large": "AudioMAE Large",
    "wav2vec2-large-lv60": "wav2vec 2.0",
    "wavlm-large": "WavLM",
    "hubert-large-ll60k": "HuBERT",
    "wavjepa-base": "WavJEPA",
    "whisper-large-v3": "Whisper",
    "data2vec-audio-large": "data2vec",
    "beats-iter3": "BEATs",
    "eat-large-20ep": "EAT",
    "clip-vit-large-patch14": "CLIP",
    "dinov2-vit-large-patch14": "DINOv2",
    "dinov3-vit-large-patch16": "DINOv3",
    "videomae-vit-large-patch16": "VideoMAE",
    "hiera-large-16x224": "Hiera",
    "mae-dfer-lgi-former": "MAE-DFER",
    "vjepa2-1-vit-large-patch16-384": "V-JEPA 2.1",
    "maskfeat-mvit-large": "MaskFeat",
    "cav-mae-scale-plus-plus": "CAV-MAE Scale++",
    "avsiam-base-plus": "AVSiam",
    "avsiam-large": "AVSiam Large",
    "languagebind-audio-large-video-large-v1-5": "LanguageBind",
    "av-hubert-large": "AV-HuBERT",
    "leavjepa-large": "LeAVJEPA",
    "social-mae": "Social-MAE",
    "emotion2vec": "Emotion2Vec",
}

FAVOR_VA_ROUTES = ("anxiety-best", "depression-best", "total-best")
BASELINE_KINDS = {
    "A": {"audiomae", "beats", "eat", "waveform", "wavjepa", "whisper"},
    "V": {"clip", "dinov2", "dinov3", "hiera", "maedfer", "maskfeat", "videomae", "vjepa21"},
    "VA": {"avhubert", "avsiam", "avsiam_large", "cavmae", "languagebind", "leavjepa", "socialmae"},
}
BASELINE_UNSTARTED, BASELINE_INCOMPLETE = "-/-/-", "～/～/～"

# PLOT/color.md：给定配置按列表顺序使用 Primary palette。
PRIMARY_COLORS = (
    "#ED66A4", "#8690C2", "#8CC7A1", "#F59E42",
    "#959BA5", "#C4B080", "#9E7BBC", "#F1F1F1",
)
EDGE_COLORS = (
    "#C94F8B", "#6673A8", "#6FA987", "#D17E32",
    "#707780", "#A98F5C", "#805F9F", "#D5D5D5",
)
BASELINE_FAMILY_COLORS = {"V": "#F08800", "A": "#009BF0", "VA": "#5EF000", "FaVoR": "#E800F0"}
FAVOR_MODALITY_COLORS = {"V": "#F054F5", "A": "#99009E", "VA": BASELINE_FAMILY_COLORS["FaVoR"]}


@dataclass(frozen=True)
class MetricRule:
    key: str
    label: str
    higher_is_better: bool


DEFAULT_RULES = {
    "classification": MetricRule("val_f1_macro", "F1-macro", True),
    "multi_label_classification": MetricRule("val_f1_macro", "F1-macro", True),
    "regression": MetricRule("val_rmse", "RMSE", False),
}
TASK_TYPE_ORDER = {
    "classification": 0,
    "multi_label_classification": 1,
    "regression": 2,
}
TASK_TYPE_SHORT = {
    "classification": "CLS",
    "multi_label_classification": "MLC",
    "regression": "REG",
}
LOWER_IS_BETTER = ("rmse", "mae", "mse", "loss", "error", "hamming")


@dataclass
class TaskResult:
    name: str
    task_type: str
    rule: MetricRule
    values: list[float]


def _metric_key(value: str) -> str:
    value = value.strip()
    return value if value.startswith("val_") else f"val_{value}"


def _custom_rule(value: str, default_label: str) -> MetricRule:
    key = _metric_key(value)
    label = key.removeprefix("val_").replace("_", " ").title()
    higher = not any(token in key.lower() for token in LOWER_IS_BETTER)
    return MetricRule(key, label or default_label, higher)


def _read_task_type(experiment: Path) -> str:
    """优先读参数快照，缺失时再从 history.csv 列名稳健推断。"""
    for path in sorted(experiment.glob("params-*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        task_type = str((payload.get("data") or {}).get("task") or "").strip()
        if task_type:
            return task_type

    history = experiment / "logs" / "history.csv"
    if history.is_file():
        with history.open(newline="", encoding="utf-8-sig") as handle:
            fields = set(csv.DictReader(handle).fieldnames or ())
        if "val_rmse" in fields:
            return "regression"
        if "val_hamming_loss" in fields and "val_balanced_accuracy" not in fields:
            return "multi_label_classification"
        if "val_f1_macro" in fields:
            return "classification"
    return "unknown"


def _best_value(experiment: Path, rule: MetricRule) -> float:
    """读 history.csv，同 epoch 重复时保留最后一行，再按主指标取最优。"""
    history = experiment / "logs" / "history.csv"
    if not history.is_file():
        raise ValueError("缺少 logs/history.csv")

    by_epoch: dict[int, float] = {}
    with history.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                epoch = int(float(row.get("epoch") or ""))
                value = float(row.get(rule.key) or "")
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                by_epoch[epoch] = value
    if not by_epoch:
        raise ValueError(f"没有可解析的 {rule.key}")
    return (max if rule.higher_is_better else min)(by_epoch.values())


def collect_tasks(
    root: Path,
    configs: list[str],
    selected_tasks: list[str] | None,
    rules: dict[str, MetricRule],
) -> tuple[list[TaskResult], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"任务根目录不存在: {root}")

    if selected_tasks:
        folders = [root / name for name in selected_tasks]
        absent = [path.name for path in folders if not path.is_dir()]
        if absent:
            raise FileNotFoundError("指定的任务目录不存在: " + ", ".join(absent))
    else:
        folders = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name.lower())

    results: list[TaskResult] = []
    warnings: list[str] = []
    for folder in folders:
        experiments = [folder / config for config in configs]
        if not any(path.is_dir() for path in experiments):
            continue
        missing = [path.name for path in experiments if not path.is_dir()]
        if missing:
            warnings.append(f"{folder.name}: 缺少配置 {', '.join(missing)}，已跳过")
            continue

        task_types = [_read_task_type(path) for path in experiments]
        known_types = {value for value in task_types if value != "unknown"}
        if len(known_types) != 1:
            warnings.append(f"{folder.name}: 配置的任务类型不一致 {task_types}，已跳过")
            continue
        task_type = known_types.pop()
        if task_type not in rules:
            warnings.append(f"{folder.name}: 不支持任务类型 {task_type!r}，已跳过")
            continue
        rule = rules[task_type]
        try:
            values = [_best_value(path, rule) for path in experiments]
        except ValueError as error:
            warnings.append(f"{folder.name}: {error}，已跳过")
            continue
        results.append(TaskResult(folder.name, task_type, rule, values))

    if not selected_tasks:
        results.sort(key=lambda item: (TASK_TYPE_ORDER.get(item.task_type, 99), item.name.lower()))
    return results, warnings


def normalize(tasks: list[TaskResult]) -> np.ndarray:
    """返回 [config, task] 的相对性能，每个 task 的最优值为 1。"""
    raw = np.asarray([task.values for task in tasks], dtype=float).T
    normalized = np.empty_like(raw)
    for column, task in enumerate(tasks):
        values = raw[:, column]
        if task.rule.higher_is_better:
            best = np.max(values)
            # 主指标理论上应为正；若用户自定义了可为负的指标，用极差归一化保证方向正确。
            if best > 0:
                normalized[:, column] = values / best
            else:
                worst = np.min(values)
                span = best - worst
                normalized[:, column] = 1.0 if span == 0 else 0.8 + 0.2 * (values - worst) / span
        else:
            best = np.min(values)
            if best > 0 and np.all(values > 0):
                normalized[:, column] = best / values
            else:
                worst = np.max(values)
                span = worst - best
                normalized[:, column] = 1.0 if span == 0 else 0.8 + 0.2 * (worst - values) / span
    return normalized


def _smooth_periodic(theta: np.ndarray, radii: np.ndarray, samples: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """在极坐标的半径维度上做周期三次样条，曲线严格穿过所有数据点。"""
    period = 2.0 * np.pi
    closed_theta = np.r_[theta, period]
    closed_radii = np.r_[radii, radii[0]]
    dense_theta = np.linspace(0.0, period, samples, endpoint=True)
    spline = CubicSpline(closed_theta, closed_radii, bc_type="periodic")
    # 样条可能在点间轻微超调，不允许它超过表示最优值的外圈。
    dense_radii = np.clip(spline(dense_theta), 0.0, 1.0)
    return dense_theta, dense_radii


def _format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _auto_radial_min(values: np.ndarray) -> float:
    smallest = float(np.nanmin(values))
    # 保留至少 20% 的可见径向宽度，且向下取整到 5%，不夸大微小差异。
    return max(0.0, min(0.8, math.floor((smallest - 0.025) / 0.05) * 0.05))


def _register_arial() -> None:
    """注册系统 Arial，避免 Matplotlib 安装前生成的旧字体缓存造成假回退。"""
    font_roots = (
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        Path("/usr/local/share/fonts"),
    )
    for root in font_roots:
        if root.is_dir():
            for path in root.glob("Arial*.ttf"):
                font_manager.fontManager.addfont(path)
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError(
            "未找到 Arial 字体；Ubuntu/Debian 可安装 ttf-mscorefonts-installer"
        ) from error


def draw_radar(
    tasks: list[TaskResult],
    configs: list[str],
    labels: list[str],
    output: Path,
    radial_min: float | None,
) -> None:
    normalized = normalize(tasks)
    lower = _auto_radial_min(normalized) if radial_min is None else radial_min
    if not 0.0 <= lower < 1.0:
        raise ValueError("--radial-min 必须在 [0, 1) 之间")

    count = len(tasks)
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    size = max(8.0, min(12.0, 7.0 + count * 0.28))

    _register_arial()
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 14.0,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",  # 保留可编辑文字，便于论文排版。
    })
    fig = plt.figure(figsize=(size, size), facecolor="white")
    ax = fig.add_subplot(111, projection="polar")
    # 这里只控制主图的相对布局；导出时再按所有可见元素的实际包围盒裁切。
    fig.subplots_adjust(left=0.095, right=0.905, top=0.905, bottom=0.095)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    # 100% 网格圆就是雷达图边界，不再在它外面绘制额外的 polar spine。
    ax.set_ylim(lower, 1.0)
    ax.set_facecolor("#FCFCFC")

    ring_values = np.linspace(lower, 1.0, 5)
    ax.set_yticks(ring_values)
    ax.set_yticklabels([f"{value:.0%}" for value in ring_values], color="#707780", fontsize=12.0)
    ax.set_rlabel_position(8)
    ax.yaxis.grid(True, color="#D5D5D5", linewidth=0.75, alpha=0.95)
    ax.xaxis.grid(True, color="#E5E5E5", linewidth=0.7, alpha=0.9)
    ax.spines["polar"].set_visible(False)

    axis_labels = [
        f"{task.name}\n{TASK_TYPE_SHORT[task.task_type]} · {task.rule.label}"
        for task in tasks
    ]
    ax.set_xticks(theta)
    task_texts = ax.set_xticklabels(axis_labels, fontsize=13.0, color="#30343B")
    # 所有任务标签都向圆外展开：水平/垂直对齐由所在象限决定，
    # 使文字包围盒不会向内侵入曲线区域。
    for angle, text in zip(theta, task_texts):
        horizontal = math.sin(angle)
        vertical = math.cos(angle)
        text.set_horizontalalignment(
            "left" if horizontal > 0.15 else "right" if horizontal < -0.15 else "center"
        )
        text.set_verticalalignment(
            "bottom" if vertical > 0.15 else "top" if vertical < -0.15 else "center"
        )
    ax.tick_params(axis="x", pad=15)

    linestyles = ("-", "--", "-.", ":")
    markers = ("o", "s", "D", "^", "v", "P", "X", "h")
    for index, (config, label) in enumerate(zip(configs, labels)):
        primary = PRIMARY_COLORS[index % len(PRIMARY_COLORS)]
        edge = EDGE_COLORS[index % len(EDGE_COLORS)]
        radii = normalized[index]
        smooth_theta, smooth_radii = _smooth_periodic(theta, radii)
        ax.fill(smooth_theta, smooth_radii, color=primary, alpha=0.11, zorder=2 + index)
        ax.plot(
            smooth_theta, smooth_radii,
            color=edge, linewidth=2.5, linestyle=linestyles[index % len(linestyles)],
            solid_capstyle="round", dash_capstyle="round", dash_joinstyle="round",
            clip_on=False, label=label, zorder=4 + index,
        )
        ax.scatter(
            theta, radii, s=31, marker=markers[index % len(markers)],
            facecolor=primary, edgecolor="white", linewidth=0.85,
            clip_on=False, zorder=8 + index,
        )

        for task_index, (angle, radius) in enumerate(zip(theta, radii)):
            raw_value = tasks[task_index].values[index]
            # 数值统一放在 marker 的径向内侧，避免 100% 点与圆外任务标签重叠。
            # 不同配置使用不同内缩深度及微小切向偏移，保持数值可分辨。
            depth = (24 if radius >= 0.985 else 8) + index * 8
            tangent = -7.0 if index % 2 == 0 else 7.0
            offset_x = -math.sin(angle) * depth + math.cos(angle) * tangent
            offset_y = -math.cos(angle) * depth - math.sin(angle) * tangent
            ax.annotate(
                _format_value(raw_value), xy=(angle, radius), xytext=(offset_x, offset_y),
                textcoords="offset points", ha="center", va="center",
                fontsize=11.0, fontweight="semibold", color=edge,
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                annotation_clip=False, zorder=12 + index,
            )

    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.120, 0.976), bbox_transform=fig.transFigure,
        ncol=1, frameon=True, fancybox=False, framealpha=0.90,
        facecolor="white", edgecolor="#D5D5D5", borderpad=0.45,
        fontsize=12.5, handlelength=2.8, labelspacing=0.45,
    )
    for text in legend.get_texts():
        text.set_color("#30343B")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        # 不强制正方形；用实际内容边界裁切，四周不额外留白。
        output, format="svg", bbox_inches="tight", pad_inches=0.0,
        facecolor="white", metadata={"Creator": "PLOT/增加radar.py"},
    )
    plt.close(fig)


def remove_bold(cell):
    return cell[2:-2] if cell.startswith('**') and cell.endswith('**') else cell


def baseline_metric(cell, task_type):
    cell = remove_bold(cell)
    if cell in ('', BASELINE_UNSTARTED, BASELINE_INCOMPLETE):
        return None
    parts = cell.split('/')
    if len(parts) != 3:
        raise ValueError(f'Invalid baseline metric triplet: {cell}')
    values = [None if part == '-' else float(part) for part in parts]
    if any(value is not None and not math.isfinite(value) for value in values):
        raise ValueError(f'Nonfinite baseline metric triplet: {cell}')
    return values[0 if task_type == 'regression' else 1]


def select_favor_va_routes(scores, task_catalog):
    """Select each task by its radar metric, breaking ties in route order."""
    winners = {}
    best = {}
    for route in FAVOR_VA_ROUTES:
        for task, value in scores.get(route, {}).items():
            score = -value if task_catalog[task]['task_type'] == 'regression' else value
            if task not in best or score > best[task]:
                best[task] = score
                winners[task] = route
    return winners


def baseline_series_label(model, route, modality):
    name = MODEL_DISPLAY_NAMES.get(model, model)
    if model == 'FaVoR':
        return f'{name} ({modality})'
    suffixes = {
        'scratch-pretrained': ' (Domain Adaption from Scratch)',
        'continued-pretrained': ' (Domain Adaption)',
    }
    return name + suffixes.get(route, '')


def baseline_models():
    configs = PROJECT_ROOT / 'BASELINES/finetune/configs'
    modalities = {kind: modality for modality, kinds in BASELINE_KINDS.items() for kind in kinds}
    models = {}
    for path in sorted(configs.glob('*.yaml')):
        if path.stem == 'defaults':
            continue
        model = yaml.safe_load(path.read_text(encoding='utf-8'))['model']
        name = model['id']
        route = 'direct'
        if model.get('encoder_checkpoint'):
            pretrain = PROJECT_ROOT / 'BASELINES/pretrain/configs' / f'{name}.yaml'
            training_route = yaml.safe_load(pretrain.read_text(encoding='utf-8'))['route']
            route = {'scratch_pretrain': 'scratch-pretrained', 'continue_pretrain': 'continued-pretrained'}[training_route]
        models[name, route] = modalities[model['kind']]
    model_index = {model: i for i, model in enumerate(MODEL_DISPLAY_NAMES)}
    return dict(sorted(models.items(), key=lambda item: (model_index.get(item[0][0], len(model_index)), item[0][1] != 'direct')))


def read_favor_csv(path, task_catalog):
    """Read validation triplets and select FaVoR VA's best clinical route per task."""
    models = {}
    cells = {}
    candidates = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        tasks = header[2:]
        if header[:2] != ['模型（模态）', '微调路线 / 实验名'] or len(tasks) != len(task_catalog) or set(tasks) != set(task_catalog):
            raise ValueError(f'Unexpected task columns in {path}')
        for row in reader:
            match = re.fullmatch(r'(.+)（(A|V|VA)）', row[0]) if row else None
            if not match or len(row) != len(header):
                raise ValueError(f'Invalid validation row at {path}:{reader.line_num}')
            model, modality = match.groups()
            route = row[1].strip()
            key = (model, modality if model == 'FaVoR' else 'validation')
            models[key] = modality
            values = {task: remove_bold(cell.strip()) for task, cell in zip(tasks, row[2:]) if cell.strip()}
            if key == ('FaVoR', 'VA') and route in FAVOR_VA_ROUTES:
                if route in candidates:
                    raise ValueError(f'Duplicate FaVoR route {route} in {path}')
                candidates[route] = {task: cell for task, cell in values.items() if task_catalog[task]['scope'] == 'mental'}
                if route != 'total-best':
                    continue
                values = {task: cell for task, cell in values.items() if task_catalog[task]['scope'] != 'mental'}
            cells.setdefault(key, {}).update(values)

    if candidates:
        scores = {route: {task: score for task, cell in values.items()
                          if (score := baseline_metric(cell, task_catalog[task]['task_type'])) is not None}
                  for route, values in candidates.items()}
        winners = select_favor_va_routes(scores, task_catalog)
        combined = cells.setdefault(('FaVoR', 'VA'), {})
        for values in candidates.values():
            for task, cell in values.items():
                if cell == BASELINE_INCOMPLETE:
                    combined.setdefault(task, BASELINE_INCOMPLETE)
        combined.update({task: candidates[route][task] for task, route in winners.items()})

    ordered = {key: modality for key, modality in models.items() if key[0] != 'FaVoR'}
    for modality in ('V', 'A', 'VA'):
        key = ('FaVoR', modality)
        if key in models:
            ordered[key] = modality
    return ordered, cells


def baseline_csv_tasks(task_catalog):
    return [task for task, row in task_catalog.items()
            if row['scope'] == 'mental' or {'a', 'v', 'va'}.issubset(row['modalities'].split('|'))]


def baseline_run_cells(root, task_catalog, models, tasks):
    """Collect test triplets and incomplete runs from the training outputs."""
    cells = {}
    finetune = root / 'finetune'
    for folder in finetune.glob('*/*/*/*'):
        model, route, scope, task = folder.relative_to(finetune).parts
        key = (model, route)
        if key not in models or task not in tasks:
            continue
        info = task_catalog[task]
        expected_scope = 'mental' if info['scope'] == 'mental' else 'ordinary'
        if scope != expected_scope or models[key].lower() not in info['modalities'].split('|'):
            continue
        path = folder / 'metrics.json'
        if not any((folder / name).is_file() for name in ('metrics.json', 'config.yaml', 'train.log')):
            continue
        if model == 'vjepa2-1-vit-large-patch16-384':
            config = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))['model']
            if tuple(config.get(k) for k in ('input_size', 'frames', 'fps')) != (112, 48, 8):
                continue
        result = json.loads(path.read_bytes()) if path.is_file() else {}
        evaluation = result.get('evaluations', {}).get('test', {})
        metrics = evaluation.get('metrics', {}) if evaluation.get('scope', 'full') == 'full' else {}
        regression = info['task_type'] == 'regression'
        fields = ('mae', 'rmse', 'r2') if regression else ('accuracy', 'f1_macro', 'auc_ovr_macro')
        parts = []
        for i, field in enumerate(fields):
            value = metrics.get(field)
            if type(value) not in (float, int) or not math.isfinite(value):
                parts.append('-')
            elif not regression and i < 2:
                parts.append(f'{100 * value:.2f}')
            else:
                parts.append(f'{value:.4f}')
        cell = '/'.join(parts)
        cells.setdefault(key, {})[task] = BASELINE_INCOMPLETE if cell == BASELINE_UNSTARTED else cell
    return cells


def write_baseline_csv(path, tasks, values, task_catalog):
    for column, task in enumerate(tasks, start=1):
        task_type = task_catalog[task]['task_type']
        for row in values:
            row[column] = remove_bold(row[column])
        scores = [(row, score) for row in values if (score := baseline_metric(row[column], task_type)) is not None]
        if scores:
            select = min if task_type == 'regression' else max
            best = select(score for _, score in scores)
            for row, score in scores:
                if score == best:
                    row[column] = f'**{row[column]}**'

    with NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent, prefix='.result-', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.writer(handle)
            writer.writerow(['model', *tasks])
            writer.writerows(values)
            handle.flush()
            temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o664)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def export_baseline_csv(root, task_catalog, models, cells):
    tasks = baseline_csv_tasks(task_catalog)
    values = [[baseline_series_label(*key, modality),
               *(cells.get(key, {}).get(task, BASELINE_UNSTARTED) for task in tasks)]
              for key, modality in models.items()]
    write_baseline_csv(root / RESULT_CSV_NAME, tasks, values, task_catalog)


def read_baseline_csv(path, task_catalog, models, *, favor_cells=None):
    expected = set(baseline_csv_tasks(task_catalog))
    labels = {baseline_series_label(*key, modality): key for key, modality in models.items()}
    if len(labels) != len(models):
        raise ValueError('Baseline display names must be unique')
    metrics = {}
    values = []
    changed = False
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if header[:1] != ['model'] or len(header) != len(expected) + 1 or set(header[1:]) != expected:
            raise ValueError(f'Unexpected task columns in {path}')
        for row in reader:
            if len(row) != len(header) or row[0] not in labels or any(not cell for cell in row[1:]):
                raise ValueError(f'Invalid model row at {path}:{reader.line_num}')
            key = labels.pop(row[0])
            if favor_cells is not None and key in favor_cells:
                updated = [favor_cells[key].get(task, BASELINE_UNSTARTED) for task in header[1:]]
                if updated != [remove_bold(cell) for cell in row[1:]]:
                    row[1:] = updated
                    changed = True
            values.append(row)
            for task, cell in zip(header[1:], row[1:]):
                info = task_catalog[task]
                score = baseline_metric(cell, info['task_type'])
                if score is not None:
                    metrics[(*key, info['scope'], task)] = score
    if labels:
        raise ValueError(f'Missing models in {path}: {", ".join(labels)}')
    if changed:
        write_baseline_csv(path, header[1:], values, task_catalog)
    return metrics


def baseline_task_label(row: dict[str, str]) -> str:
    task = row["task_id"]
    dataset = row["dataset"].split("_", 1)[-1].rsplit("_", 1)[0]
    prefix = f"{dataset}-"
    if task.startswith(prefix):
        return f"{dataset}\n{task.removeprefix(prefix)}"
    return task.replace("-", "\n", 1)


def baseline_family_shades(family: str) -> list[str]:
    """Alternate light and dark shades of the comparison chart's base color."""
    primary = np.array(to_rgb(BASELINE_FAMILY_COLORS[family]))
    secondary = .78 * primary
    difference = primary - secondary
    return [to_hex(shade) for shade in (
        secondary, .60 * secondary, primary, .77 * secondary,
        primary + .14 * (1 - primary), .49 * secondary,
        secondary + .65 * difference, .69 * secondary,
        secondary + .34 * difference, .85 * secondary, .93 * secondary,
    )]


def draw_baselines(root: Path, output: Path, *, results_root: Path | None = None, audio33_output: Path | None = None,
                   refresh_results: bool = False) -> None:
    """绘制心理/普通 × V/A/VA 八面板，使用 Macro-F1 和 MAE。

    普通音频面板使用三模态共有任务，可另外导出完整的普通音频雷达图。
    每轴使用原始指标刻度，将该任务的最佳和最差结果线性映射到外圈和近圆心。
    相邻节点均有结果时以直线相连，不跨越缺失任务。
    """
    if audio33_output is not None and audio33_output.resolve() == output.resolve():
        raise ValueError("八面板雷达图与完整音频雷达图需要不同的输出路径")
    results_root = root if results_root is None else results_root
    with (PROJECT_ROOT / 'CONFIGS/tasks/finetune/tasks.csv').open(encoding='utf-8') as stream:
        rows = [row for row in csv.DictReader(stream) if row['enabled'] == '1']
    task_catalog = {row['task_id']: row for row in rows}
    task_types = {task: row['task_type'] for task, row in task_catalog.items()}
    baselines = baseline_models()
    validation_models, favor_cells = read_favor_csv(results_root / FAVOR_CSV_NAME, task_catalog)
    models = baselines | validation_models
    csv_tasks = set(baseline_csv_tasks(task_catalog))
    result_csv = results_root / RESULT_CSV_NAME
    if refresh_results or not result_csv.exists():
        test_cells = baseline_run_cells(root, task_catalog, baselines, csv_tasks)
        export_baseline_csv(results_root, task_catalog, models, test_cells | favor_cells)
    metrics = read_baseline_csv(result_csv, task_catalog, models, favor_cells=favor_cells)
    if audio33_output is not None:
        extra_tasks = task_catalog.keys() - csv_tasks
        audio_baselines = {key: modality for key, modality in baselines.items() if modality == 'A'}
        audio_cells = baseline_run_cells(root, task_catalog, audio_baselines, extra_tasks)
        audio_cells.update(favor_cells)
        for key, modality in models.items():
            if modality != 'A':
                continue
            for task, cell in audio_cells.get(key, {}).items():
                if task in extra_tasks and (score := baseline_metric(cell, task_types[task])) is not None:
                    metrics[(*key, 'nonmental', task)] = score

    favor_va_key = ('FaVoR', 'VA')
    # Rank against the same series that set the combined panel's axis scales.
    mental_peers = [key for key in models if key != favor_va_key]

    def task_order(row):
        regression = row['task_type'] == 'regression'
        if row['scope'] != 'mental':
            return regression, 0
        task = row['task_id']
        score = metrics.get((*favor_va_key, 'mental', task))
        peers = [value for key in mental_peers if (value := metrics.get((*key, 'mental', task))) is not None]
        if score is None or not peers:
            return regression, math.inf
        rank = 1 + sum(value < score if regression else value > score for value in peers)
        return regression, rank

    rows.sort(key=task_order)
    model_colors = {}
    for modality in ('V', 'A', 'VA'):
        shades = baseline_family_shades(modality)
        names = dict.fromkeys(name for (name, _), model_mod in models.items()
                              if model_mod == modality and name != 'FaVoR')
        for i, name in enumerate(names):
            model_colors[name] = shades[i % len(shades)]
    markers={'V':'o', 'A':'s', 'VA':'D'}
    opacities={'V':.85, 'A':.85, 'VA':1.0}
    line_width = 1.1
    legend_marker_size = 2.2
    fig, axes=plt.subplots(3,4,figsize=(44,36),subplot_kw={'projection':'polar'})
    fig.subplots_adjust(left=.025, right=.975, top=.96, bottom=.065, wspace=.14, hspace=.42)
    figures = [(fig, output)]
    panels = [(ax, scope, mod, mod == 'ALL' or (scope, mod) == ('nonmental', 'A'))
              for scope, panel_row in zip(('mental', 'mental_core', 'nonmental'), axes)
              for mod, ax in zip(('V', 'A', 'VA', 'ALL'), panel_row)]
    if audio33_output is not None:
        audio_fig, audio_ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
        audio_fig.subplots_adjust(left=.15, right=.85, top=.90, bottom=.14)
        panels.append((audio_ax, 'nonmental', 'A', False))
        figures.append((audio_fig, audio33_output))

    for ax, scope, mod, shared_only in panels:
      required = {'v', 'a', 'va'} if shared_only else {mod.lower()}
      metric_scope = 'mental' if scope == 'mental_core' else scope
      tasks = [row['task_id'] for row in rows if row['scope'] == metric_scope
               and (scope != 'mental_core' or row.get('mental_core') == '1')
               and required.issubset(row['modalities'].split('|'))]
      n = len(tasks)
      if not n: continue
      theta=2*np.pi*np.arange(n)/n
      task_values = [[metrics[m,rt,metric_scope,t] for (m,rt),mo in models.items() if (mo==mod or mod=='ALL') and (m,rt,metric_scope,t) in metrics] for t in tasks]
      lower=np.array([min(values,default=np.nan) for values in task_values])
      upper=np.array([max(values,default=np.nan) for values in task_values])
      span = upper - lower
      regression = np.array([task_types[task] == 'regression' for task in tasks])
      ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1);ax.set_ylim(0,110);ax.set_xticks(theta);ax.set_xticklabels([]);ax.set_yticks(np.linspace(18,90,9));ax.set_yticklabels([]);ax.xaxis.grid(color='#AEB8C4',alpha=.25,lw=.55,ls='-');ax.yaxis.grid(color='#AEB8C4',alpha=.23,lw=.7,ls='-');ax.spines['polar'].set_visible(False)
      dense=np.linspace(0,2*np.pi,1000);ax.plot(dense,np.full_like(dense,90),color='#9FA5AD',lw=1,ls='-',alpha=.7)
      # Each spoke has its own original metric scale; the direction respects the task type.
      for i,t in enumerate(tasks):
       if not np.isfinite(lower[i]) or upper[i] <= lower[i]: continue
       ticks = np.linspace(upper[i], lower[i], 9) if task_types[t]=='regression' else np.linspace(lower[i], upper[i], 9)
       for radius,value in zip(np.linspace(18,90,9),ticks):
        label=f'{value:.3f}' if task_types[t]=='regression' else f'{value:.1f}'
        tangent_x, tangent_y = np.cos(theta[i]), -np.sin(theta[i])
        ha = 'left' if tangent_x > .3 else ('right' if tangent_x < -.3 else 'center')
        va = 'bottom' if tangent_y > .7 else ('top' if tangent_y < -.7 else 'center')
        ax.annotate(label, (theta[i], radius), xytext=(5*tangent_x, 5*tangent_y),
                    textcoords='offset points', color='black', fontsize=5.5, ha=ha, va=va,
                    zorder=10, bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':.1})
      handles_by_modality = {'V': [], 'A': [], 'VA': []}
      favor_handles = []
      for (model,route),mo in models.items():
       if mo != mod and mod != 'ALL':
        continue
       vals = np.array([metrics.get((model, route, metric_scope, task), np.nan) for task in tasks])
       valid = np.isfinite(vals)
       has_results = valid.any()
       is_favor = model == 'FaVoR'
       color = FAVOR_MODALITY_COLORS[mo] if is_favor else model_colors[model]
       alpha = 1.0 if is_favor else opacities[mo]
       if has_results:
        fraction = np.divide(vals - lower, span, out=np.ones_like(vals), where=span > 0)
        invert = regression & (span > 0)
        fraction[invert] = 1 - fraction[invert]
        radii = 18 + 72 * np.clip(fraction, 0, 1)
        if valid.all() and n >= 3:
         ax.fill(theta, radii, color=color, alpha=.035, edgecolor='none', zorder=1)
        for i in range(n):
         j = (i + 1) % n
         if valid[i] and valid[j]:
          ax.plot([theta[i], theta[i] + 2*np.pi/n], [radii[i], radii[j]], color=color,
                  lw=line_width, alpha=alpha, linestyle='-', zorder=4 if is_favor else 2)
        ax.scatter(theta[valid], radii[valid], s=10, marker=markers[mo],
                   color=[color], alpha=alpha, edgecolor='white', linewidth=.25,
                   zorder=6 if is_favor else 5)
       handles = favor_handles if is_favor and mod == 'ALL' else handles_by_modality[mo]
       handles.append(
           Line2D([0], [0], color=color, lw=line_width, alpha=alpha, linestyle='-',
                  marker=markers[mo], markersize=legend_marker_size,
                  label=baseline_series_label(model, route, mo),
                  markerfacecolor=color if has_results else 'none')
       )
      for i, (angle, task) in enumerate(zip(theta, tasks)):
       label = baseline_task_label(task_catalog[task])
       degrees = np.degrees(angle)
       align = 'center' if degrees < 8 or degrees > 352 or 172 < degrees < 188 else ('left' if degrees < 180 else 'right')
       ax.text(angle, 100, label, ha=align, va='center', multialignment='center',
               fontsize=7.2 if n == 38 else 8, linespacing=.8 if n == 38 else .95,
               color='#16723C' if regression[i] else '#A61B29', clip_on=False)
      scope_title = {'mental': 'Mental', 'mental_core': 'Mental Core', 'nonmental': 'Ordinary'}[scope]
      ax.set_title(scope_title+' · '+{'V':'Video','A':'Audio','VA':'Audio-video','ALL':'All modalities'}[mod]+f' ({n} tasks)',fontsize=18,pad=16,weight='bold')
      if not np.isfinite(lower).any():
       ax.text(0, 0, 'No completed results', ha='center', va='center', fontsize=16, color='#888888')
      if mod == 'ALL':
       # Matplotlib fills legends column-first. Pad to keep each modality in one column.
       column_height = max(map(len, handles_by_modality.values()))
       blank = Line2D([], [], linestyle='none', label=' ')
       handles = [handle for group in handles_by_modality.values()
                  for handle in group + [blank] * (column_height - len(group))]
      else:
       handles = handles_by_modality[mod]
      if favor_handles:
       favor_legend = ax.legend(handles=favor_handles, loc='upper center', bbox_to_anchor=(.5, -.02),
                                fontsize=8, frameon=False, ncol=len(favor_handles), handlelength=2)
       ax.add_artist(favor_legend)
      if handles:
       ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.057 if favor_handles else -.02), fontsize=8,
                 frameon=False, ncol=3 if mod == 'ALL' else 2, handlelength=2,
                 handletextpad=.5, columnspacing=1, labelspacing=.28 if mod == 'ALL' else .35)
    for figure, path in figures:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=190, bbox_inches='tight', pad_inches=.06)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从任务根目录读取多个配置的最优主指标，绘制平滑归一化雷达图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baselines", action="store_true", help="绘制对比模型十二面板雷达图和完整普通音频雷达图")
    parser.add_argument("--refresh-results", action="store_true", help=f"绘图前从测试产物和 {FAVOR_CSV_NAME} 重建 {RESULT_CSV_NAME}")
    parser.add_argument("--audio33-out", type=Path, help="对比模型模式下完整普通音频雷达图的输出路径")
    parser.add_argument("--task", type=Path, help="包含多个任务子目录的根目录")
    parser.add_argument("--configs", nargs="+", help="实验配置目录名，顺序同时决定配色")
    parser.add_argument("--labels", nargs="+", help="图例显示名，数量必须与 --configs 相同")
    parser.add_argument("--tasks", nargs="+", help="只画指定任务，并保留此处给出的轴顺序")
    parser.add_argument("--classification-metric", default="val_f1_macro", help="单标签分类主指标")
    parser.add_argument("--multilabel-metric", default="val_f1_macro", help="多标签分类主指标")
    parser.add_argument("--regression-metric", default="val_rmse", help="回归主指标")
    parser.add_argument("--radial-min", type=float, help="径向刻度下限，默认根据数据自动取 5%% 整刻度")
    parser.add_argument("--out", type=Path, help="输出路径；默认使用对应模式的输出路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_ROOT))
    from BASELINES.common.paths import OUTPUT_ROOT, RADAR_ROOT, RESULTS_ROOT

    if args.baselines:
        output = PROJECT_ROOT / args.out if args.out is not None else RADAR_ROOT / "对比模型雷达图.png"
        audio33_output = args.audio33_out
        if audio33_output is None and args.out is None:
            audio33_output = RADAR_ROOT / AUDIO33_RADAR_NAME
        if audio33_output is not None:
            audio33_output = PROJECT_ROOT / audio33_output
        draw_baselines(OUTPUT_ROOT, output, results_root=RESULTS_ROOT,
                       audio33_output=audio33_output, refresh_results=args.refresh_results)
        for written in (output, audio33_output):
            if written is not None:
                print(f"wrote {written.resolve()}")
        return 0
    if args.task is None or not args.configs:
        raise SystemExit("常规模式需要 --task 和 --configs；对比模型图使用 --baselines")
    args.out = args.out or RADAR_ROOT / "radar.svg"
    if len(args.configs) < 1:
        raise SystemExit("--configs 至少需要一项")
    if len(args.configs) > len(PRIMARY_COLORS):
        raise SystemExit(f"配置数不能超过 {len(PRIMARY_COLORS)}（color.md 只定义了 8 个主色）")
    labels = args.labels or args.configs
    if len(labels) != len(args.configs):
        raise SystemExit("--labels 的数量必须与 --configs 相同")
    if args.out.suffix.lower() != ".svg":
        raise SystemExit("--out 必须以 .svg 结尾")

    root = args.task if args.task.is_absolute() else PROJECT_ROOT / args.task
    output = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    rules = {
        "classification": _custom_rule(args.classification_metric, "F1-macro"),
        "multi_label_classification": _custom_rule(args.multilabel_metric, "F1-macro"),
        "regression": _custom_rule(args.regression_metric, "RMSE"),
    }
    # 保留默认指标的标准、紧凑显示名。
    for task_type, default in DEFAULT_RULES.items():
        if rules[task_type].key == default.key:
            rules[task_type] = default

    tasks, warnings = collect_tasks(root.resolve(), args.configs, args.tasks, rules)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if len(tasks) < 3:
        raise SystemExit(f"可用任务只有 {len(tasks)} 个；雷达图至少需要 3 个共同任务")

    draw_radar(tasks, args.configs, labels, output.resolve(), args.radial_min)
    print(f"wrote {output.resolve()}")
    print(f"tasks ({len(tasks)}): " + ", ".join(task.name for task in tasks))
    for task in tasks:
        direction = "max" if task.rule.higher_is_better else "min"
        values = ", ".join(f"{label}={_format_value(value)}" for label, value in zip(labels, task.values))
        print(f"  {task.name} [{task.rule.key}, {direction}]: {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
