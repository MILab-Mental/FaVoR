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
      --out PLOT/output/radar/main.svg
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline


PLOT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLOT_ROOT.parent
DEFAULT_OUTPUT = PLOT_ROOT / "output" / "radar" / "radar.svg"

# PLOT/color.md：给定配置按列表顺序使用 Primary palette。
PRIMARY_COLORS = (
    "#ED66A4", "#8690C2", "#8CC7A1", "#F59E42",
    "#959BA5", "#C4B080", "#9E7BBC", "#F1F1F1",
)
EDGE_COLORS = (
    "#C94F8B", "#6673A8", "#6FA987", "#D17E32",
    "#707780", "#A98F5C", "#805F9F", "#D5D5D5",
)


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


def read_baseline_validation(text, task_catalog):
    """Read the source-labelled validation tables attached to the report."""
    _, _, section = text.partition('\n## 历史验证集结果\n')
    scopes = {'### 普通任务': 'nonmental', '### 心理临床任务': 'mental'}
    scope = None
    tasks = []
    for line in section.splitlines():
        if line in scopes:
            scope = scopes[line]
            tasks = []
        if not line.startswith('| '):
            continue
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if cells[0] == '模型（模态）':
            tasks = cells[3:]
            continue
        match = re.fullmatch(r'(.+)（(A|V|VA)）', cells[0])
        if not match:
            continue
        if scope is None or not tasks or len(cells) != len(tasks) + 3:
            raise ValueError(f'Invalid historical validation table row: {cells[:3]}')
        model, modality = match.groups()
        route = cells[1].strip('`')
        values = {}
        for task, cell in zip(tasks, cells[3:]):
            if not cell or cell == '-/-/-':
                continue
            task_info = task_catalog[task]
            if task_info['scope'] != scope or modality.lower() not in task_info['modalities'].split('|'):
                raise ValueError(f'Unexpected validation metric for {model}, {route}: {task}')
            parts = cell.split('/')
            if len(parts) != 3:
                raise ValueError(f'Invalid validation metric for {model}, {route}, {task}: {cell}')
            value = parts[0 if task_info['task_type'] == 'regression' else 1]
            if value != '-' and math.isfinite(float(value)):
                values[task] = float(value)
        yield model, modality, route, scope, values


def draw_baselines(root: Path, output: Path) -> None:
    """绘制心理/普通 × V/A/VA 八面板，使用 Macro-F1 和 MAE。

    图例以 [val] 标记历史验证集；每轴使用相同的原始指标刻度，
    将每个任务的最佳和最差结果线性映射到外圈和近圆心。
    相邻节点均有结果时以直线相连，不跨越缺失任务。
    """
    report = root / "对比模型进度表.md"
    text = report.read_text(encoding="utf-8")
    with (PROJECT_ROOT / 'CONFIGS/tasks/finetune/tasks.csv').open(encoding='utf-8') as stream:
        rows = [row for row in csv.DictReader(stream) if row['enabled'] == '1']
    task_catalog = {row['task_id']: row for row in rows}
    task_types={r['task_id']:r['task_type'] for r in rows}
    models={}
    for model,mod,route in re.findall(r'\| `([^`]+)`（(A|V|VA)） \| ([^|]+) \|',text):
     models[(model,{'直接微调':'direct','继续预训练后微调':'continued-pretrained','从头预训练后微调':'scratch-pretrained'}[route.strip()])]=mod
    metrics={}
    test_selected = set()
    for p in (root/'finetune').glob('*/*/*/*/metrics.json'):
     model,route,scope,task,_=p.relative_to(root/'finetune').parts
     try:
      evaluation = json.loads(p.read_text()).get('evaluations', {}).get('test', {})
      m = evaluation.get('metrics', {})
      v = m.get('mae' if task_types.get(task) == 'regression' else 'f1_macro')
      if isinstance(v, (int, float)) and math.isfinite(v):
       panel_scope = 'nonmental' if scope == 'ordinary' else scope
       metrics[model, route, panel_scope, task] = v if task_types.get(task) == 'regression' else 100*v
       if evaluation.get('selection_split') == 'test':
        test_selected.add((model, route, panel_scope))
     except (ValueError,OSError,AttributeError): continue
    validation = set()
    for model, modality, route, scope, values in read_baseline_validation(text, task_catalog):
        key = (model, route)
        models[key] = modality
        validation.add(key)
        for task, value in values.items():
            metrics[model, route, scope, task] = value
    # Qualitative colors within each family retain modality cues while separating models.
    palettes={
     'A':['#123B8A','#0077BB','#56B4E9','#6650A4','#242F59','#3280A1','#47316E','#2867D0','#596B8A','#008DC8','#235AAB'],
     'V':['#B2182B','#E66101','#D4A000','#8C510A','#722530','#F28E2B','#A27700','#C84B54','#D66816'],
     'VA':['#005A42','#009E73','#18A999','#66A61E','#0A4039','#327848','#087E83','#4E6815'],
    }
    model_colors={}
    for modality, palette in palettes.items():
     names=list(dict.fromkeys(name for (name, _), model_mod in models.items() if model_mod==modality and name!='FaVoR'))
     for i,name in enumerate(names): model_colors[name]=palette[i % len(palette)]
    pink_palette = ('#DA2B82', '#AE206A', '#E755A1', '#8C1457', '#FF84B9')
    favor_colors = {key: pink_palette[i % len(pink_palette)]
                    for i, key in enumerate(key for key in models if key[0] == 'FaVoR')}
    markers={'V':'o', 'A':'s', 'VA':'D'}
    opacities={'V':.85, 'A':.85, 'VA':1.0}
    line_widths={'V':1.25, 'A':1.25, 'VA':2.0}
    fig, axes=plt.subplots(2,4,figsize=(48,26),subplot_kw={'projection':'polar'})
    for row,scope in enumerate(['mental','nonmental']):
     for col,mod in enumerate(['V','A','VA','ALL']):
      ax=axes[row,col]; tasks=[r['task_id'] for r in rows if r['scope']==scope and (all(m in r['modalities'].split('|') for m in ('v','a','va')) if mod=='ALL' else mod.lower() in r['modalities'].split('|'))];n=len(tasks)
      if not n: continue
      theta=2*np.pi*np.arange(n)/n
      task_values = [[metrics[m,rt,scope,t] for (m,rt),mo in models.items() if (mo==mod or mod=='ALL') and (m,rt,scope,t) in metrics] for t in tasks]
      lower=np.array([min(values,default=np.nan) for values in task_values])
      upper=np.array([max(values,default=np.nan) for values in task_values])
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
      handles=[]
      for (model,route),mo in models.items():
       if mo != mod and mod != 'ALL':
        continue
       vals=np.array([metrics.get((model,route,scope,t),np.nan) for t in tasks]);valid=np.isfinite(vals)
       if not np.any(valid): continue
       span=upper-lower
       fraction=np.divide(vals-lower,span,out=np.ones_like(vals),where=span>0)
       for i,t in enumerate(tasks):
        if task_types[t]=='regression' and span[i]>0: fraction[i]=1-fraction[i]
       rad=18+72*np.clip(fraction,0,1)
       key = (model, route)
       is_favor = key in favor_colors
       is_validation = key in validation
       color = favor_colors[key] if is_favor else model_colors[model]
       width = 2.25 if is_favor else line_widths[mo]
       alpha = 1.0 if is_favor else opacities[mo]
       if valid.all() and n >= 3:
        ax.fill(theta,rad,color=color,alpha=.035,edgecolor='none',zorder=1)
       for i in range(n):
        j=(i+1)%n
        if valid[i] and valid[j]:ax.plot([theta[i],theta[i]+2*np.pi/n],[rad[i],rad[j]],color=color,lw=width,alpha=alpha,linestyle='-',zorder=4 if is_favor else 2)
       ax.scatter(theta[valid],rad[valid],s=35 if is_favor else 28,marker=markers[mo],color=[color],alpha=alpha,edgecolor='white',linewidth=.3,zorder=6 if is_favor else 5)
       if is_validation:
        label = f'{model} ({mo}) · {route} [val]'
       else:
        label = model if route=='direct' else model+' ('+('continued' if route=='continued-pretrained' else 'scratch')+')'
        if (model, route, scope) in test_selected:
         label += ' [test-selected]'
       handles.append(Line2D([0],[0],color=color,lw=width,alpha=alpha,linestyle='-',marker=markers[mo],markersize=4,label=label))
      for i,t in enumerate(theta):
       label=tasks[i].replace('DSMxSCL-','D×S-').replace('RESULT-','RES-').replace('RepetitiveThoughtsBehaviors','Repetitive').replace('InterpersonalSensitivity','Interpersonal').replace('PersonalityInterpersonal','PersonalityInter').replace('PsychosisParanoidExpanded','PsychosisParanoid').replace('ObsessiveCompulsive','Obsessive').replace('PersonalityFunctioning','PersonalityFunc').replace('SomaticSymptoms','Somatic')
       deg=np.degrees(t);align='center' if deg<8 or deg>352 or 172<deg<188 else ('left' if deg<180 else 'right')
       ax.text(t,98+(i%2)*2,label,ha=align,va='center',fontsize=7.4 if n==38 else 8,color='#30343A',clip_on=False)
      ax.set_title(('Mental' if scope=='mental' else 'Ordinary')+' · '+{'V':'Video','A':'Audio','VA':'Audio-video','ALL':'All modalities'}[mod]+f' ({n} tasks)',fontsize=18,pad=32,weight='bold')
      if not handles: ax.text(0,0,'No completed results',ha='center',va='center',fontsize=16,color='#888888')
      if mod=='ALL':
       handles += [Line2D([0],[0], color=palettes[m][0], lw=line_widths[m], alpha=opacities[m], marker=markers[m], markersize=4, label={'V':'Video','A':'Audio','VA':'Audio-video'}[m]) for m in ('V','A','VA')]
      if handles: ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.06),fontsize=8,frameon=False,ncol=2,handlelength=2.6)
    if validation or test_selected:
        fig.suptitle('[val]: historical validation  |  [test-selected]: checkpoint selected on test  |  Other curves: test',
                     y=.988, fontsize=18)
    fig.subplots_adjust(left=.04,right=.96,top=.95,bottom=.12,wspace=.38,hspace=.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=190,bbox_inches='tight',pad_inches=.25);plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从任务根目录读取多个配置的最优主指标，绘制平滑归一化雷达图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baselines", action="store_true", help="绘制对比模型八面板雷达图")
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
    if args.baselines:
        sys.path.insert(0, str(PROJECT_ROOT))
        from BASELINES.common.paths import OUTPUT_ROOT

        output = args.out or OUTPUT_ROOT / "对比模型雷达图.png"
        output = output if output.is_absolute() else PROJECT_ROOT / output
        draw_baselines(OUTPUT_ROOT, output)
        print(f"wrote {output.resolve()}")
        return 0
    if args.task is None or not args.configs:
        raise SystemExit("常规模式需要 --task 和 --configs；对比模型图使用 --baselines")
    args.out = args.out or DEFAULT_OUTPUT
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
