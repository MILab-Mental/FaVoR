#!/usr/bin/env python3
"""FaVoR vs V-JEPA 2.1 的跨任务对比图：柱状图 / 箱线图 / 小提琴图 + ROC。

和 ``ana.py`` 的分工：``ana.py`` 是**单任务内部**多配置的对比（bootstrap 分布 + 配对
显著性 + ROC）；本脚本是**跨任务**对比，固定两个方法——主结果是 FaVoR，对比的是
直接加载官方 V-JEPA 2.1 权重的 ``vjepaori`` 基线。

用法:
    python PLOT/ana_bench.py                                   # 全部任务，默认 1000 次 bootstrap
    python PLOT/ana_bench.py --n-bootstrap 2000 --per-task      # 多跑一点，并额外出每任务分面图
    python PLOT/ana_bench.py --tasks RAVDESS-emotion MER2023-emotion
    python PLOT/ana_bench.py --redraw-only --per-task           # 只读缓存重画，不重算指标/显著性/ROC

产出（默认 ``PLOT/output/bench/``）::

    <task 类型>/
        bar/<metric>.png          # 横轴 = 该类型的各个任务，每个任务两根柱子
        box/<metric>.png          # 同上，箱线图（箱体来自 bootstrap 重采样分布）
        violin/<metric>.png       # 同上，小提琴图
        task/<task>.png           # --per-task 才有：单个任务，横轴 = 各指标的分面图
    roc/<task>.png                # 每个分类任务一张，FaVoR 与 V-JEPA 叠在同一张图上
    roc/overview.png              # 上面所有任务的缩略网格
    significance.csv              # 逐任务、逐指标的配对 bootstrap 显著性
    metrics_bootstrap.csv         # 逐任务、逐方法、逐指标的 bootstrap 均值/标准差/95% CI
    bootstrap_samples.npz         # 原始 bootstrap 分布，供 --redraw-only 快速精确重画
    README.txt                    # 口径说明

取数是**每个任务各自的评测集预测文件**（``logs/eval_best_predict.csv``），
不是 ``history.csv``——因为 bootstrap 需要逐样本的
预测，而且要保证两个方法比较的是同一批样本。所有指标都在同一条代码路径上重算，
两个方法口径完全一致，比拿训练时写进 CSV 的数字更可比。

"没有的指标就留空"：任务缺 vjepaori 那一侧、或某个指标在重采样里算不出来（比如某个
bootstrap 子样本里某个类一次都没出现，macro AUC 无定义）时，图上那个位置就是空的
（柱子位置标一个浅灰 "n/a"），CSV 里对应单元格留空，显著性列留空。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, hamming_loss, jaccard_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

PLOT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLOT_ROOT.parent
DEFAULT_OUTPUT = PLOT_ROOT / "output" / "bench"
BOOTSTRAP_CACHE = "bootstrap_samples.npz"
CACHE_SEPARATOR = "::"


# --------------------------------------------------------------------------------------
# 复用 ana.py 的配色 / 显著性口径 / 分类指标定义，避免两个脚本的口径漂移
# --------------------------------------------------------------------------------------
def _load_module(name: str, filename: str):
    """按文件路径加载同目录的模块（不用 ``import ana``，免得起名太泛被别的包顶掉）。"""
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # stat.py / ana.py 用到 @dataclass，dataclasses 解析字符串注解时要靠 sys.modules 找回命名空间
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ana = _load_module("favor_ana_helpers", "ana.py")
stat = _load_module("favor_stat_helpers", "stat.py")

# 主结果用调色板第一位（ana.py 的约定：main_result 占 #ED66A4），对比方法顺延一位。
METHODS = (("favor", "FaVoR"), ("vjepaori", "V-JEPA 2.1"))

# (指标键, 图上的轴标签, 越大越好还是越小越好)。键就是各 _*_metrics() 返回的字典键。
# 指标清单按任务类型分开——分类的 accuracy 和回归的 r2 没法画在同一根轴上。
METRICS = {
    "classification": [
        ("accuracy", "Accuracy", "up"),
        ("balanced_accuracy", "Balanced accuracy", "up"),
        ("precision_macro", "Precision (macro)", "up"),
        ("precision_weighted", "Precision (weighted)", "up"),
        ("precision_micro", "Precision (micro)", "up"),
        ("recall_macro", "Recall (macro)", "up"),
        ("recall_weighted", "Recall (weighted)", "up"),
        ("recall_micro", "Recall (micro)", "up"),
        ("f1_macro", "F1 (macro)", "up"),
        ("f1_weighted", "F1 (weighted)", "up"),
        ("f1_micro", "F1 (micro)", "up"),
        ("cohen_kappa", "Cohen's kappa", "up"),
        ("specificity_macro", "Specificity (macro)", "up"),
        ("specificity_weighted", "Specificity (weighted)", "up"),
        ("log_loss", "Log loss", "down"),
        ("auc_ovr_macro", "AUC (OvR macro)", "up"),
        ("auc_ovr_weighted", "AUC (OvR weighted)", "up"),
        ("average_precision_macro", "Avg. precision (macro)", "up"),
        ("average_precision_weighted", "Avg. precision (weighted)", "up"),
    ],
    "multi_label_classification": [
        ("hamming_loss", "Hamming loss", "down"),
        # 多标签下的 accuracy 是 subset/exact-match accuracy，23 个标签全对才算对，
        # 实测基本恒为 0——但既然是"所有指标都算"，还是列出来，只是把它排在最前面说明清楚。
        ("subset_accuracy", "Subset accuracy", "up"),
        ("precision_macro", "Precision (macro)", "up"),
        ("precision_weighted", "Precision (weighted)", "up"),
        ("precision_micro", "Precision (micro)", "up"),
        ("recall_macro", "Recall (macro)", "up"),
        ("recall_weighted", "Recall (weighted)", "up"),
        ("recall_micro", "Recall (micro)", "up"),
        ("f1_macro", "F1 (macro)", "up"),
        ("f1_weighted", "F1 (weighted)", "up"),
        ("f1_micro", "F1 (micro)", "up"),
        ("jaccard_macro", "Jaccard (macro)", "up"),
        ("jaccard_weighted", "Jaccard (weighted)", "up"),
        ("jaccard_micro", "Jaccard (micro)", "up"),
        ("log_loss", "Log loss", "down"),
        ("auc_ovr_macro", "AUC (macro)", "up"),
        ("auc_ovr_weighted", "AUC (weighted)", "up"),
        ("average_precision_macro", "Avg. precision (macro)", "up"),
        ("average_precision_weighted", "Avg. precision (weighted)", "up"),
    ],
    "regression": [
        ("mae", "MAE", "down"),
        ("mse", "MSE", "down"),
        ("rmse", "RMSE", "down"),
        ("r2", "R$^2$", "up"),
        ("adjusted_r2", "Adjusted R$^2$", "up"),
        ("pearson", "Pearson r", "up"),
        ("spearman", "Spearman rho", "up"),
    ],
}

TASK_TYPE_DIR = {
    "classification": "classification",
    "multi_label_classification": "multilabel",
    "regression": "regression",
}
# 所有任务统一使用最佳 epoch 的评测集预测。
PREDICTION_FILES = ("eval_best_predict.csv",)


# --------------------------------------------------------------------------------------
# 指标定义（分类复用 ana.py，多标签和回归在这里补齐）
# --------------------------------------------------------------------------------------
def _multilabel_metrics(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """多标签指标。``truth`` 是 0/1 指示矩阵，``scores`` 是每个标签的概率。

    口径逐条对齐 ``utils/classification_metrics.multilabel_metrics``（训练时用的那个），
    否则本脚本算出来的数会和 ``logs/best.csv`` 对不上：
      * ``subset_accuracy`` = 整行全对的比例（训练里就叫 ``accuracy``）；
      * ``log_loss`` 是**逐元素 BCE**——sklearn 的 ``log_loss`` 会把 2D 输入当多分类分布
        重新归一，对各自独立的 sigmoid 没有意义，所以这里直接按公式算；
      * ``auc_ovr_*`` / ``average_precision_*`` 是**先逐类算再聚合**，不是把整个矩阵丢给
        sklearn 的 ``average=`` 参数——后者只要有一个类在子样本里只剩一种取值就整体报错，
        而逐类算可以把那一类跳过（训练时也是这么做的）。
    """
    predictions = (scores >= 0.5).astype(int)
    labels = np.arange(scores.shape[1])
    answer = {
        "subset_accuracy": accuracy_score(truth, predictions),
        "hamming_loss": hamming_loss(truth, predictions),
    }
    for average in ("macro", "weighted", "micro"):
        answer[f"precision_{average}"] = precision_score(
            truth, predictions, labels=labels, average=average, zero_division=0)
        answer[f"recall_{average}"] = recall_score(
            truth, predictions, labels=labels, average=average, zero_division=0)
        answer[f"f1_{average}"] = f1_score(
            truth, predictions, labels=labels, average=average, zero_division=0)
        answer[f"jaccard_{average}"] = jaccard_score(
            truth, predictions, labels=labels, average=average, zero_division=0)

    clipped = np.clip(scores, 1e-15, 1.0 - 1e-15)
    answer["log_loss"] = float(
        -(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped)).mean())

    per_class_auc, per_class_ap, support = [], [], []
    for column in labels:
        target = truth[:, column]
        count = int(target.sum())
        per_class_auc.append(roc_auc_score(target, scores[:, column])
                             if target.min() != target.max() else np.nan)
        per_class_ap.append(average_precision_score(target, scores[:, column])
                            if count else np.nan)
        support.append(count)
    for name, values in (("auc_ovr", per_class_auc), ("average_precision", per_class_ap)):
        values = np.asarray(values, dtype=float)
        available = np.isfinite(values)
        if not available.any():
            continue
        answer[f"{name}_macro"] = float(np.nanmean(values))
        answer[f"{name}_weighted"] = float(np.average(
            values[available], weights=np.asarray(support, dtype=float)[available]))
    return answer


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """回归指标。``prediction`` 是 (n, 1) 的预测列，先摊平。"""
    truth = np.asarray(truth, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    residual = truth - prediction
    count = truth.size
    mse = float(np.mean(residual ** 2))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # 与 app/finetune_v/train.py:_regression_extra_metrics 同一公式（单自变量，p = 1）
    adjusted_r2 = 1.0 - (1.0 - r2) * (count - 1) / (count - 2) if count > 2 else float("nan")
    answer = {
        "mae": float(np.mean(np.abs(residual))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
        "adjusted_r2": adjusted_r2,
    }
    for key, function in (("pearson", _pearson), ("spearman", _spearman)):
        try:
            answer[key] = function(truth, prediction)
        except ValueError:
            pass
    return answer


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        raise ValueError("constant input")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda vector: np.argsort(np.argsort(vector)).astype(float)  # noqa: E731
    return _pearson(rank(a), rank(b))


METRIC_FUNCTIONS = {
    "classification": ana._classification_metrics,
    "multi_label_classification": _multilabel_metrics,
    "regression": _regression_metrics,
}


# --------------------------------------------------------------------------------------
# 读数据
# --------------------------------------------------------------------------------------
def _multilabel_indicator(values, num_labels: int) -> np.ndarray:
    """``'7|14|23'`` -> 0/1 指示矩阵。

    **标签 1-based**：训练侧 ``utils/classification_metrics._format_label_set`` 写的是
    ``str(index + 1)``，所以 ``'7|14|23'`` 指的是第 6、13、22 列（0-based）。这里要减 1，
    否则整个指示矩阵会整体错位一列，hamming / f1 全都不对。
    """
    matrix = np.zeros((len(values), num_labels), dtype=int)
    for row, text in enumerate(values):
        for token in str(text).split("|"):
            token = token.strip()
            if not token:
                continue
            index = int(float(token)) - 1
            if 0 <= index < num_labels:
                matrix[row, index] = 1
    return matrix


@dataclass
class Run:
    """一个方法在一个任务上的预测。"""

    method: str
    folder: Path | None = None
    task_type: str | None = None
    paths: np.ndarray | None = None
    truth: np.ndarray | None = None
    scores: np.ndarray | None = None
    note: str = ""

    @property
    def available(self) -> bool:
        return self.scores is not None


@dataclass
class Task:
    name: str
    task_type: str
    runs: dict[str, Run] = field(default_factory=dict)


def _read_predictions(folder: Path, task_type: str | None) -> Run | None:
    """读一个实验目录的评测预测，返回 ``Run``（读不到返回 None）。"""
    for filename in PREDICTION_FILES:
        path = folder / "logs" / filename
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        columns = set(frame.columns)
        probability_columns = ana._probability_columns(frame)
        if probability_columns and {"path", "true_label"} <= columns:
            paths = frame.path.astype(str).to_numpy()
            scores = frame[probability_columns].to_numpy(float)
            # 多标签的 true_label 是 '7|14|23' 这样的字符串，单标签是整数
            if task_type == "multi_label_classification" or frame.true_label.dtype == object:
                truth = _multilabel_indicator(frame.true_label, len(probability_columns))
                kind = "multi_label_classification"
            else:
                truth = frame.true_label.to_numpy(int)
                kind = "classification"
            return Run(method="", folder=folder, task_type=kind, paths=paths,
                       truth=truth, scores=scores, note=filename)
        if {"path", "target", "prediction"} <= columns:
            return Run(method="", folder=folder, task_type="regression",
                       paths=frame.path.astype(str).to_numpy(),
                       truth=frame.target.to_numpy(float),
                       scores=frame.prediction.to_numpy(float).reshape(-1, 1), note=filename)
    return None


def _index_by_task(pattern: str) -> dict[str, Path]:
    """展开通配符，按「任务目录名 -> 实验目录」建索引。"""
    import glob

    found: dict[str, Path] = {}
    for hit in glob.glob(str(Path(pattern).expanduser()), recursive=True):
        path = Path(hit)
        if path.is_dir():
            found.setdefault(path.parent.name, path)
    return found


def _task_type_of(folder: Path | None) -> str | None:
    if folder is None:
        return None
    task = stat.collect(str(folder), folder, None).task
    return task if task in METRICS else None


def collect_tasks(favor_pattern: str, vjepaori_pattern: str, only: list[str] | None) -> dict[str, Task]:
    """把两侧目录合成 ``{任务名: Task}``，缺失的一侧留空。"""
    favor_dirs = _index_by_task(favor_pattern)
    vjepaori_dirs = _index_by_task(vjepaori_pattern)
    names = sorted(set(favor_dirs) | set(vjepaori_dirs))
    if only:
        wanted = set(only)
        names = [name for name in names if name in wanted]
        for missing in sorted(wanted - set(names)):
            print(f"警告: --tasks 里的 {missing} 没有任何匹配目录，跳过", file=sys.stderr)

    tasks: dict[str, Task] = {}
    for name in names:
        task_type = _task_type_of(favor_dirs.get(name)) or _task_type_of(vjepaori_dirs.get(name))
        if task_type is None:
            print(f"跳过 {name}: 推不出任务类型（缺 history.csv / params 快照？）", file=sys.stderr)
            continue
        task = Task(name=name, task_type=task_type)
        for method, pattern_dirs in (("favor", favor_dirs), ("vjepaori", vjepaori_dirs)):
            folder = pattern_dirs.get(name)
            if folder is None:
                task.runs[method] = Run(method=method, note="目录不存在")
                continue
            run = _read_predictions(folder, task_type)
            if run is None:
                task.runs[method] = Run(method=method, folder=folder, note="没有评测预测 csv")
            else:
                run.method = method
                task.runs[method] = run
        tasks[name] = task
    return tasks


# --------------------------------------------------------------------------------------
# 配对对齐 + bootstrap
# --------------------------------------------------------------------------------------
def align(favor: Run, other: Run):
    """按 ``path`` 把两个方法的样本对齐，返回 (truth, favor_scores, other_scores)。

    两个方法必须评的是同一批样本，否则"配对" bootstrap 无从谈起。样本数相同且没有重复
    path 时按 path 对齐；有重复 path 就退回按行号对齐（两个 csv 都来自同一个评测集，
    顺序一致）；数量和内容都对不上则返回 None，那这个任务的显著性就留空。
    """
    if not (favor.available and other.available):
        return None
    if favor.truth.shape != other.truth.shape or favor.scores.shape != other.scores.shape:
        return None
    favor_paths, other_paths = favor.paths, other.paths
    if len(set(favor_paths)) == len(favor_paths) and len(set(other_paths)) == len(other_paths):
        position = {path: index for index, path in enumerate(other_paths)}
        rows_favor = [index for index, path in enumerate(favor_paths) if path in position]
        rows_other = [position[favor_paths[index]] for index in rows_favor]
        if len(rows_favor) < 2:
            return None
        rows_favor, rows_other = np.asarray(rows_favor), np.asarray(rows_other)
    elif len(favor_paths) == len(other_paths):
        rows_favor = rows_other = np.arange(len(favor_paths))
    else:
        return None
    favor_truth = favor.truth[rows_favor]
    other_truth = other.truth[rows_other]
    if not np.array_equal(favor_truth, other_truth):
        return None
    return favor_truth, favor.scores[rows_favor], other.scores[rows_other]


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def bootstrap(task_type: str, truth, scores, count: int, rng) -> dict[str, np.ndarray]:
    """非参数 bootstrap：每次有放回重采样整批样本，重算全部指标。

    某个指标在某个子样本上算不出来（``function`` 里被 ``except ValueError`` 吞掉）时，
    那一次留在 NaN —— 后面取均值/分位数都用 ``_finite`` 先过滤掉。
    """
    function = METRIC_FUNCTIONS[task_type]
    keys = [key for key, _label, _direction in METRICS[task_type]]
    samples = {key: np.full(count, np.nan) for key in keys}
    for step, rows in enumerate(rng.integers(0, len(truth), size=(count, len(truth)))):
        computed = function(truth[rows], scores[rows])
        for key in keys:
            value = computed.get(key)
            if value is not None:
                samples[key][step] = value
    return samples


def paired_bootstrap(task_type: str, truth, favor_scores, other_scores, count: int, rng):
    """配对 bootstrap：同一个重采样下标同时作用于两个方法，取差值。"""
    function = METRIC_FUNCTIONS[task_type]
    keys = [key for key, _label, _direction in METRICS[task_type]]
    differences = {key: np.full(count, np.nan) for key in keys}
    for step, rows in enumerate(rng.integers(0, len(truth), size=(count, len(truth)))):
        favor_values = function(truth[rows], favor_scores[rows])
        other_values = function(truth[rows], other_scores[rows])
        for key in keys:
            left, right = favor_values.get(key), other_values.get(key)
            if left is None or right is None:
                continue
            difference = left - right
            if np.isfinite(difference):
                differences[key][step] = difference
    return differences


def _mean_of(values) -> float | None:
    """bootstrap 分布的均值；没有可用样本（该方法没跑）时返回 None，CSV 里就是空格。"""
    finite = _finite(values) if values is not None else np.array([])
    return float(np.mean(finite)) if finite.size else None


def two_sided_p(differences: np.ndarray) -> float:
    """ana.py 同款的两侧 bootstrap p 值：小于等于 0 和大于等于 0 的比例取小者乘二。"""
    finite = differences[np.isfinite(differences)]
    if finite.size == 0:
        return float("nan")
    return float(2 * min(np.mean(finite <= 0), np.mean(finite >= 0)))


# --------------------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------------------
def _colors() -> dict[str, str]:
    return {method: ana._category_color(index) for index, (method, _label) in enumerate(METHODS)}


def _legend(ax):
    colors = _colors()
    handles = [Patch(facecolor=colors[method], edgecolor=ana.INK, label=label)
               for method, label in METHODS]
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)


def _draw(ax, kind: str, x: float, values: np.ndarray, color: str, dots: bool):
    """在 ``x`` 位置画一个柱子 / 一个箱子 / 一把小提琴，返回 (均值, 图形顶部)。

    "图形顶部"是画出来的东西的上沿：柱子是均值+标准差，箱/小提琴是数据最大值。
    显著性括号接在这个高度上，才不会浮在半空。
    """
    width = 0.34
    if kind == "bar":
        mean = float(np.mean(values))
        error = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        ax.bar(x, mean, width=width, yerr=error, capsize=2.5, color=color,
               edgecolor=ana.INK, linewidth=0.6, zorder=3)
        # 柱子本身已经用误差棒表达了离散度，再叠散点只会糊成一片（箱/小提琴才叠）
        return mean, mean + error
    if kind == "box":
        artists = ax.boxplot([values], positions=[x], widths=width, patch_artist=True,
                             showfliers=False, zorder=3)
        for box in artists["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.75)
            box.set_edgecolor(ana.INK)
        for group in ("whiskers", "caps", "medians"):
            for line in artists[group]:
                line.set_color(ana.INK)
    else:
        artists = ax.violinplot([values], positions=[x], widths=width,
                                showmeans=True, showextrema=False)
        for body in artists["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(ana.INK)
            body.set_alpha(0.65)
        if "cmeans" in artists:
            artists["cmeans"].set_color(ana.INK)
            artists["cmeans"].set_linewidth(1.0)
    if dots and values.size > 1:
        # 上千个 bootstrap 复制全画出来是糊的，等距抽稀到最多 120 个
        picked = values if values.size <= 120 else values[np.linspace(0, values.size - 1, 120).astype(int)]
        jitter = np.random.default_rng(ana.RNG_SEED).uniform(-width * 0.35, width * 0.35, picked.size)
        ax.scatter(x + jitter, picked, s=5, color=color, alpha=0.22, edgecolors="none", zorder=2)
    return float(np.mean(values)), float(np.max(values))


def _set_limits(ax, kind: str, series: list[np.ndarray], pad: float = 0.08):
    """收紧 y 轴范围。

    柱状图**不从 0 起**：这些任务的指标差异普遍在 0.01 量级（比如 F1 0.692 vs 0.685），
    从 0 起画就是两根一样高的柱子，看不出任何信息。这里按"均值±标准差"定标并在柱顶标数值，
    标题里也会写明 y 轴不是从 0 起的。箱/小提琴本来就不从 0 起，同理按整个分布定标。
    """
    if not series:
        return
    if kind == "bar":
        low = min(float(np.mean(values) - (np.std(values, ddof=1) if values.size > 1 else 0)) for values in series)
        high = max(float(np.mean(values) + (np.std(values, ddof=1) if values.size > 1 else 0)) for values in series)
    else:
        low = min(float(np.min(values)) for values in series)
        high = max(float(np.max(values)) for values in series)
    span = high - low
    margin = span * pad if span > 0 else (abs(high) * 0.05 or 0.05)
    ax.set_ylim(low - margin, high + margin)


def _bracket(ax, x1: float, x2: float, top1: float, top2: float, text: str, clearance: float = 0.08):
    """两个位置之间的显著性括号（和 ana.py 的画法一致）。

    ``clearance`` 是横杠相对"最高那个图形顶部"再抬多少（占当前纵向范围的比例）——柱状图
    顶部还有一排数值标注，抬太少括号会压在数字上。
    """
    lower, upper = ax.get_ylim()
    span = upper - lower
    if span <= 0:
        span = 1e-6
    step = span * 0.05
    bar = max(top1, top2) + span * clearance
    ax.plot([x1, x1, x2, x2], [top1, bar, bar, top2], color=ana.INK, linewidth=1.0, zorder=6)
    ax.text((x1 + x2) / 2, bar + step * 0.15, text, ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=ana.INK, zorder=7)
    # 星号文字会被顶到轴顶；多留 1.6 倍 step 的余量，分面图里才不会和面板标题贴在一起
    ax.set_ylim(lower, max(upper, bar + step * 1.6))


def _mark_missing(ax, x: float, text: str = "n/a"):
    """缺数据的槽位不放东西，只在底部标一个浅灰 n/a，跟"值为 0"区分开。"""
    lower, upper = ax.get_ylim()
    ax.text(x, lower + (upper - lower) * 0.03, text, ha="center", va="bottom", rotation=90,
            fontsize=6.5, color="#959BA5", zorder=1)


def _axis_label(label: str, direction: str) -> str:
    return f"{label} ({'higher' if direction == 'up' else 'lower'} is better)"


def plot_metric(kind: str, spec, task_names: list[str], distributions: dict,
                significance: dict, output: Path, dots: bool):
    """横轴 = 各任务、每个任务一根 FaVoR 一根 V-JEPA 的图（柱 / 箱 / 小提琴）。"""
    key, label, direction = spec
    colors = _colors()
    positions = np.arange(len(task_names), dtype=float)
    offsets = {"favor": -0.19, "vjepaori": 0.19}

    # 宽度随任务数走；多标签那组只有 1 个任务，固定 8 英寸会让一根柱子撑满整张图
    fig, ax = plt.subplots(figsize=(max(4.2, len(task_names) * 1.25), 5.9))
    tops: dict[tuple[str, int], float] = {}
    drawn: list[np.ndarray] = []
    for method, _method_label in METHODS:
        for index, name in enumerate(task_names):
            values = _finite(distributions[name][method].get(key, np.array([])))
            if values.size == 0:
                continue
            x = positions[index] + offsets[method]
            mean, top = _draw(ax, kind, x, values, colors[method], dots)
            tops[(method, index)] = top
            drawn.append(values)
            if kind == "bar":
                ax.text(x, top, f"{mean:.3f}", ha="center", va="bottom", fontsize=6.5,
                        color=ana.INK, zorder=8)

    # 先定 y 范围，再画括号和 n/a —— 两者都以当前范围为准
    _set_limits(ax, kind, drawn)
    for index, name in enumerate(task_names):
        present = {}
        for method, _method_label in METHODS:
            if (method, index) in tops:
                present[method] = tops[(method, index)]
            else:
                _mark_missing(ax, positions[index] + offsets[method])
        if len(present) == 2:
            p_value = significance.get(name, {}).get(key, {}).get("p_value", float("nan"))
            if np.isfinite(p_value):
                _bracket(ax, positions[index] + offsets["favor"], positions[index] + offsets["vjepaori"],
                         present["favor"], present["vjepaori"], ana._p_stars(p_value))

    ax.set_xticks(positions, task_names, rotation=30, ha="right")
    ax.set_ylabel(_axis_label(label, direction))
    zoomed = kind == "bar" and ax.get_ylim()[0] > 0
    ax.set_title(f"{label}: FaVoR vs V-JEPA 2.1 ({kind}, {len(task_names)} task"
                 f"{'' if len(task_names) == 1 else 's'})"
                 + (", y-axis zoomed (not zero-based)" if zoomed else ""))
    ax.grid(axis="y", color="#F1F1F1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _legend(ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_task_grid(kind: str, name: str, specs: list, distributions: dict,
                   significance: dict, output: Path, columns: int = 4):
    """单个任务：每个指标一个小面板，面板里两根柱子（柱 / 箱 / 小提琴）。"""
    colors = _colors()
    rows = (len(specs) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 3.0), squeeze=False)
    for position, (key, label, direction) in enumerate(specs):
        ax = axes[position // columns][position % columns]
        tops: dict[str, float] = {}
        drawn: list[np.ndarray] = []
        for method, _method_label in METHODS:
            values = _finite(distributions[name][method].get(key, np.array([])))
            if not values.size:
                continue
            slots = {"favor": 0.81, "vjepaori": 1.19}
            _mean, top = _draw(ax, kind, slots[method], values, colors[method], dots=False)
            tops[method] = top
            drawn.append(values)
        _set_limits(ax, kind, drawn)
        for method, _method_label in METHODS:
            if method not in tops:
                _mark_missing(ax, {"favor": 0.81, "vjepaori": 1.19}[method])
        if len(tops) == 2:
            p_value = significance.get(name, {}).get(key, {}).get("p_value", float("nan"))
            if np.isfinite(p_value):
                _bracket(ax, 0.81, 1.19, tops["favor"], tops["vjepaori"], ana._p_stars(p_value))
        ax.set_xticks([])
        ax.set_title(f"{label} ({'higher' if direction == 'up' else 'lower'} better)", fontsize=8)
        ax.grid(axis="y", color="#F1F1F1", linewidth=0.8)
        ax.set_axisbelow(True)
    for position in range(len(specs), rows * columns):
        axes[position // columns][position % columns].axis("off")
    handles = [Patch(facecolor=colors[method], edgecolor=ana.INK, label=label)
               for method, label in METHODS]
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"{name}: FaVoR vs V-JEPA 2.1 ({kind})", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _roc_data(run: Run):
    """返回 (指示矩阵, 分数矩阵)；单标签转 one-hot，多标签本来就有一份。"""
    if run.task_type == "classification":
        return np.eye(run.scores.shape[1])[run.truth], run.scores
    return run.truth.astype(int), run.scores


def _macro_roc(indicators: np.ndarray, scores: np.ndarray):
    """macro 平均的 one-vs-rest ROC：每条类别曲线插值到公共 FPR 网格再取均值。"""
    grid = np.linspace(0, 1, 101)
    curves, aucs = [], []
    for column in range(indicators.shape[1]):
        truth = indicators[:, column]
        if np.unique(truth).size < 2:
            continue
        fpr, tpr, _thresholds = roc_curve(truth, scores[:, column])
        curves.append(np.interp(grid, fpr, tpr))
        aucs.append(roc_auc_score(truth, scores[:, column]))
    if not curves:
        return None
    return grid, np.mean(curves, axis=0), float(np.mean(aucs))


def _macro_roc_bootstrap(indicators: np.ndarray, scores: np.ndarray, count: int, rng):
    """一次重采样同时拿到曲线和 AUC：曲线堆叠起来求逐点 95% 带，AUC 求 95% 区间。"""
    curves, aucs = [], []
    for rows in rng.integers(0, len(indicators), size=(count, len(indicators))):
        computed = _macro_roc(indicators[rows], scores[rows])
        if computed is None:
            continue
        curves.append(computed[1])
        aucs.append(computed[2])
    return np.asarray(curves), np.asarray(aucs, dtype=float)


def plot_roc(name: str, runs: dict[str, Run], output: Path, count: int, rng):
    """一个任务一张 ROC：FaVoR 与 V-JEPA 叠在一起，带 bootstrap 95% CI 带。"""
    colors = _colors()
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    drawn = 0
    for method, method_label in METHODS:
        run = runs[method]
        if not run.available or run.task_type == "regression":
            continue
        indicators, scores = _roc_data(run)
        baseline = _macro_roc(indicators, scores)
        if baseline is None:
            continue
        grid, mean_curve, mean_auc = baseline
        curves, aucs = _macro_roc_bootstrap(indicators, scores, count, rng)
        color = colors[method]
        interval = ""
        if curves.size:
            ax.fill_between(grid, np.quantile(curves, 0.025, axis=0),
                            np.quantile(curves, 0.975, axis=0),
                            color=color, alpha=0.12, linewidth=0)
        if aucs.size:
            interval = (f", 95% CI {np.quantile(aucs, 0.025):.3f}"
                        f"-{np.quantile(aucs, 0.975):.3f}")
        ax.plot(grid, mean_curve, color=color, linewidth=1.9,
                label=f"{method_label} (macro AUC={mean_auc:.3f}{interval})")
        drawn += 1
    if not drawn:
        plt.close(fig)
        return False
    ax.plot([0, 1], [0, 1], color="#959BA5", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
           title=f"{name}: macro-average one-vs-rest ROC")
    ax.grid(color="#F1F1F1", linewidth=0.8)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def analyse(tasks: dict[str, Task], output_root: Path, count: int, rng, per_task: bool, dots: bool):
    specs_by_type = {task_type: METRICS[task_type] for task_type in METRICS}
    groups: dict[str, list[str]] = {}
    for name, task in tasks.items():
        groups.setdefault(task.task_type, []).append(name)
    for names in groups.values():
        names.sort()

    significance_rows, bootstrap_rows = [], []
    raw_samples: dict[str, np.ndarray] = {}
    for task_type, names in groups.items():
        specs = specs_by_type[task_type]
        keys = [key for key, _label, _direction in specs]
        directory = output_root / TASK_TYPE_DIR[task_type]
        distributions = {name: {method: {} for method, _label in METHODS} for name in names}
        significance: dict[str, dict] = {name: {} for name in names}

        for name in names:
            task = tasks[name]
            favor, other = task.runs["favor"], task.runs["vjepaori"]
            for method, _label in METHODS:
                run = task.runs[method]
                if not run.available:
                    for key in keys:
                        distributions[name][method][key] = np.array([])
                        # 没跑过的方法也占一行（值为空），这样 Excel 里一筛就知道缺哪些
                        bootstrap_rows.append({"task": name, "task_type": task_type, "metric": key,
                                               "method": method, "note": run.note})
                    continue
                samples = bootstrap(task_type, run.truth, run.scores, count, rng)
                for key in keys:
                    distributions[name][method][key] = samples[key]
                    raw_samples[_cache_key(name, method, key)] = samples[key]
                    finite = _finite(samples[key])
                    bootstrap_rows.append({
                        "task": name, "task_type": task_type, "metric": key, "method": method,
                        "n_samples": int(len(run.truth)),
                        "mean": float(np.mean(finite)) if finite.size else None,
                        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
                        "ci95_low": float(np.quantile(finite, 0.025)) if finite.size else None,
                        "ci95_high": float(np.quantile(finite, 0.975)) if finite.size else None,
                        "source": str(run.folder or ""), "note": run.note,
                    })

            aligned = align(favor, other)
            if aligned is None:
                # 配对显著性算不了，但每一侧自己的均值还是有的（FaVoR 通常都有）——
                # 只把差值/p/星号/胜者这些"配对"列留空，别把能填的也一起丢掉。
                reason = ("vjepaori 那一侧没有可用预测" if not other.available
                          else "两个方法的评测样本对不齐")
                for key, label, direction in specs:
                    favor_mean = _mean_of(distributions[name]["favor"][key])
                    other_mean = _mean_of(distributions[name]["vjepaori"][key])
                    significance[name][key] = {"p_value": float("nan"), "note": reason}
                    significance_rows.append({
                        "task": name, "task_type": task_type, "metric": key, "label": label,
                        "better": "higher" if direction == "up" else "lower",
                        "n_paired": None, "favor_mean": favor_mean, "vjepaori_mean": other_mean,
                        "note": reason,
                    })
                print(f"  {name}: 不算显著性（{reason}）")
                continue

            truth, favor_scores, other_scores = aligned
            differences = paired_bootstrap(task_type, truth, favor_scores, other_scores, count, rng)
            for key, label, direction in specs:
                difference = differences[key]
                finite = difference[np.isfinite(difference)]
                p_value = two_sided_p(difference)
                favor_mean = _mean_of(distributions[name]["favor"][key])
                other_mean = _mean_of(distributions[name]["vjepaori"][key])
                significance[name][key] = {"p_value": p_value, "mean_difference": float(np.mean(finite))
                                           if finite.size else float("nan")}
                significance_rows.append({
                    "task": name, "task_type": task_type, "metric": key, "label": label,
                    "better": "higher" if direction == "up" else "lower",
                    "n_paired": int(len(truth)), "favor_mean": favor_mean, "vjepaori_mean": other_mean,
                    "difference": float(np.mean(finite)) if finite.size else None,
                    "ci95_low": float(np.quantile(finite, 0.025)) if finite.size else None,
                    "ci95_high": float(np.quantile(finite, 0.975)) if finite.size else None,
                    "p_value": p_value if np.isfinite(p_value) else None,
                    "stars": ana._p_stars(p_value),
                    "winner": _winner(favor_mean, other_mean, direction),
                })

        for kind in ("bar", "box", "violin"):
            for key, label, direction in specs:
                plot_metric(kind, (key, label, direction), names, distributions, significance,
                            directory / kind / f"{key}.png", dots)
        if per_task:
            for name in names:
                for kind in ("bar", "box", "violin"):
                    plot_task_grid(kind, name, specs, distributions, significance,
                                   directory / "task" / kind / f"{name}.png")
        print(f"{TASK_TYPE_DIR[task_type]}: {len(names)} 个任务 × {len(specs)} 个指标 × 3 种图"
              f" -> {directory}")

    return significance_rows, bootstrap_rows, raw_samples


def _cache_key(task: str, method: str, metric: str) -> str:
    """NPZ 里的稳定键；任务名和指标名都不使用 ``::``。"""
    return CACHE_SEPARATOR.join((task, method, metric))


def _summary_values(mean, std) -> np.ndarray:
    """用 CSV 摘要构造仅供柱状图使用的最小数组。

    ``[mean-std, mean, mean+std]`` 的样本标准差（ddof=1）恰好等于 ``std``，
    因此能无损复现现有柱状图的均值和误差棒。这些点不用于箱线图或小提琴图。
    """
    if pd.isna(mean):
        return np.array([], dtype=float)
    mean = float(mean)
    if pd.isna(std) or float(std) <= 0:
        return np.asarray([mean], dtype=float)
    std = float(std)
    return np.asarray([mean - std, mean, mean + std], dtype=float)


def redraw_cached(output_root: Path, selected_tasks: list[str] | None,
                  per_task: bool, dots: bool) -> int:
    """仅从已保存的 CSV/NPZ 重画，完全不读预测文件、不做 bootstrap。"""
    significance_path = output_root / "significance.csv"
    summary_path = output_root / "metrics_bootstrap.csv"
    raw_path = output_root / BOOTSTRAP_CACHE
    missing = [str(path) for path in (significance_path, summary_path) if not path.is_file()]
    if missing:
        print("--redraw-only 缺少缓存文件: " + ", ".join(missing), file=sys.stderr)
        print("请先不带 --redraw-only 完整运行一次。", file=sys.stderr)
        return 2

    significance_frame = pd.read_csv(significance_path)
    summary_frame = pd.read_csv(summary_path)
    if selected_tasks:
        wanted = set(selected_tasks)
        significance_frame = significance_frame[significance_frame["task"].isin(wanted)]
        summary_frame = summary_frame[summary_frame["task"].isin(wanted)]
        found = set(summary_frame["task"].dropna().astype(str))
        unknown = sorted(wanted - found)
        if unknown:
            print("缓存里找不到任务: " + ", ".join(unknown), file=sys.stderr)
            return 2
    if summary_frame.empty:
        print("缓存中没有可重画的任务。", file=sys.stderr)
        return 2

    significance: dict[str, dict] = {}
    for row in significance_frame.to_dict("records"):
        significance.setdefault(str(row["task"]), {})[str(row["metric"])] = {
            "p_value": float(row["p_value"]) if pd.notna(row.get("p_value")) else float("nan")
        }

    raw = np.load(raw_path, allow_pickle=False) if raw_path.is_file() else None
    kinds = ("bar", "box", "violin") if raw is not None else ("bar",)
    if raw is None:
        print(f"未找到 {raw_path.name}：使用 CSV 摘要精确重画柱状图；"
              "保留现有箱线图、小提琴图和 ROC。")
    else:
        print(f"已加载原始 bootstrap 缓存: {raw_path}")

    groups = (summary_frame[["task_type", "task"]].drop_duplicates()
              .groupby("task_type", sort=False)["task"].apply(list).to_dict())
    for task_type, task_names in groups.items():
        if task_type not in METRICS:
            print(f"跳过未知任务类型: {task_type}", file=sys.stderr)
            continue
        task_names = sorted(str(name) for name in task_names)
        specs = METRICS[task_type]
        directory = output_root / TASK_TYPE_DIR[task_type]
        distributions = {name: {method: {} for method, _label in METHODS} for name in task_names}
        for name in task_names:
            for method, _label in METHODS:
                for key, _metric_label, _direction in specs:
                    cache_key = _cache_key(name, method, key)
                    if raw is not None and cache_key in raw:
                        values = np.asarray(raw[cache_key], dtype=float)
                    else:
                        rows = summary_frame[
                            (summary_frame["task"] == name)
                            & (summary_frame["method"] == method)
                            & (summary_frame["metric"] == key)
                        ]
                        values = (np.array([], dtype=float) if rows.empty
                                  else _summary_values(rows.iloc[0].get("mean"), rows.iloc[0].get("std")))
                    distributions[name][method][key] = values

        for kind in kinds:
            for spec in specs:
                key = spec[0]
                plot_metric(kind, spec, task_names, distributions, significance,
                            directory / kind / f"{key}.png", dots)
        if per_task:
            for name in task_names:
                for kind in kinds:
                    plot_task_grid(kind, name, specs, distributions, significance,
                                   directory / "task" / kind / f"{name}.png")
        print(f"{TASK_TYPE_DIR[task_type]}: {len(task_names)} 个任务 × {len(specs)} 个指标"
              f" × {len(kinds)} 种图（仅重画） -> {directory}")
    if raw is not None:
        raw.close()
    print("仅重画完成：未重算指标、bootstrap、显著性或 ROC。")
    return 0


def _winner(favor_mean, other_mean, direction: str) -> str:
    if favor_mean is None or other_mean is None:
        return ""
    if favor_mean == other_mean:
        return "tie"
    favor_better = favor_mean > other_mean if direction == "up" else favor_mean < other_mean
    return "favor" if favor_better else "vjepaori"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FaVoR vs V-JEPA 2.1 的跨任务对比图（柱状图 / 箱线图 / 小提琴图 / ROC）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python PLOT/ana_bench.py\n"
            "  python PLOT/ana_bench.py --n-bootstrap 2000 --per-task\n"
            "  python PLOT/ana_bench.py --tasks RAVDESS-emotion MER2023-emotion\n"
            "  python PLOT/ana_bench.py --redraw-only --per-task\n"
        ),
    )
    parser.add_argument("--favor", default=str(PROJECT_ROOT / "OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-e5"),
                        help="FaVoR 主结果那一侧的通配符")
    parser.add_argument("--vjepaori", default=str(PROJECT_ROOT / "OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-vjepaori"),
                        help="V-JEPA 2.1 基线那一侧的通配符（没有的任务留空）")
    parser.add_argument("--tasks", nargs="+", default=None, metavar="NAME",
                        help="只画这些任务（默认全画）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help=f"输出目录（默认 {DEFAULT_OUTPUT}）")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="bootstrap 重采样次数（默认 1000，和 ana.py 一致）")
    parser.add_argument("--per-task", action="store_true",
                        help="额外为每个任务出一张「一指标一个面板」的分面图")
    parser.add_argument("--no-dots", action="store_true", help="不叠加 bootstrap 散点")
    parser.add_argument("--redraw-only", "--plot-only", action="store_true",
                        help="只读取 --out 中的缓存重画；不重算指标、bootstrap、显著性或 ROC")
    args = parser.parse_args()
    if not args.redraw_only and args.n_bootstrap < 20:
        parser.error("--n-bootstrap 至少要 20")

    if args.redraw_only:
        return redraw_cached(args.out, args.tasks, args.per_task, not args.no_dots)

    tasks = collect_tasks(args.favor, args.vjepaori, args.tasks)
    if not tasks:
        print(f"没有任务可用：{args.favor} 或 {args.vjepaori} 都没匹配到目录", file=sys.stderr)
        return 2

    rng = np.random.default_rng(ana.RNG_SEED)
    print(f"共 {len(tasks)} 个任务，bootstrap {args.n_bootstrap} 次")
    significance_rows, bootstrap_rows, raw_samples = analyse(
        tasks, args.out, args.n_bootstrap, rng, args.per_task, not args.no_dots)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(significance_rows).to_csv(args.out / "significance.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(args.out / "metrics_bootstrap.csv", index=False)
    np.savez_compressed(args.out / BOOTSTRAP_CACHE, **raw_samples)

    # ROC：只有分类/多标签有，回归没有 ROC 这一步
    roc_drawn = []
    for name, task in tasks.items():
        if task.task_type == "regression":
            continue
        if plot_roc(name, task.runs, args.out / "roc" / f"{name}.png", args.n_bootstrap, rng):
            roc_drawn.append(name)
    if roc_drawn:
        plot_roc_overview(roc_drawn, tasks, args.out / "roc" / "overview.png",
                          args.n_bootstrap, rng)

    (args.out / "README.txt").write_text(_readme(tasks, args.n_bootstrap, roc_drawn), encoding="utf-8")
    print(f"\n显著性 -> {args.out / 'significance.csv'}")
    print(f"bootstrap 分布 -> {args.out / 'metrics_bootstrap.csv'}")
    print(f"bootstrap 原始缓存 -> {args.out / BOOTSTRAP_CACHE}")
    if roc_drawn:
        print(f"ROC -> {args.out / 'roc'}（{len(roc_drawn)} 个分类任务 + overview.png）")
    missing = [name for name, task in tasks.items() if not task.runs["vjepaori"].available]
    if missing:
        print(f"\nvjepaori 那侧还缺 {len(missing)} 个任务，这些位置在图上留空: "
              + ", ".join(sorted(missing)))
    return 0


def plot_roc_overview(names: list[str], tasks: dict[str, Task], output: Path, count: int, rng):
    """所有分类任务的 ROC 缩略网格，方便一眼扫完。"""
    columns = min(4, len(names))
    rows = (len(names) + columns - 1) // columns
    colors = _colors()
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.6, rows * 3.5), squeeze=False)
    for position, name in enumerate(names):
        ax = axes[position // columns][position % columns]
        for method, method_label in METHODS:
            run = tasks[name].runs[method]
            if not run.available:
                continue
            indicators, scores = _roc_data(run)
            computed = _macro_roc(indicators, scores)
            if computed is None:
                continue
            grid, curve, auc = computed
            ax.plot(grid, curve, color=colors[method], linewidth=1.5,
                    label=f"{method_label} ({auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#959BA5", linestyle="--", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), title=name, xlabel="FPR", ylabel="TPR")
        ax.grid(color="#F1F1F1", linewidth=0.7)
        ax.legend(fontsize=6.5, loc="lower right")
    for position in range(len(names), rows * columns):
        axes[position // columns][position % columns].axis("off")
    fig.suptitle("Macro-average one-vs-rest ROC (macro AUC in legend)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _readme(tasks: dict[str, Task], count: int, roc_drawn: list[str]) -> str:
    lines = [
        "ana_bench.py 产出说明",
        "",
        f"bootstrap 次数: {count}（非参数重采样，有放回，每次抽满整个评测集）",
        "采样口径: 每个任务各自评测集的预测文件（logs/eval_best_predict.csv）。",
        "         所有指标都由 ana_bench.py 在同一条代码路径上从逐样本预测重算，"
        "两个方法完全一致；",
        "         因此数值可能与 history.csv / best.csv 里训练时记下的略有出入"
        "（口径不同），但方法之间的对比是严格的。",
        "显著性: 配对 bootstrap——同一个重采样下标同时作用于两个方法，按样本 path 对齐后再取差；",
        "       p 值是两侧的 2*min(P(diff<=0), P(diff>=0))（与 ana.py 一致），",
        "       stars 记法 ns / * / ** / *** / ****（0.05 / 0.01 / 0.001 / 0.0001）。",
        "留空规则: 缺 vjepaori 一侧、或某指标在该 bootstrap 子样本里无定义（如某类一次没出现，"
        "macro AUC 未定义），",
        "         图里对应槽位空着并标浅灰 n/a，CSV 里对应单元格留空。",
        "ROC: macro 平均的 one-vs-rest；曲线是把每条类别曲线插值到公共 FPR 网格后取均值，",
        "     阴影带是 bootstrap 95% 区间。回归任务没有 ROC。",
        "",
        "每个任务的可用情况:",
    ]
    for name in sorted(tasks):
        task = tasks[name]
        parts = []
        for method, method_label in METHODS:
            run = task.runs[method]
            parts.append(f"{method_label}={'有' if run.available else '缺(' + run.note + ')'}")
        lines.append(f"  {name:28} {task.task_type:28} " + "  ".join(parts))
    if roc_drawn:
        lines += ["", f"已画 ROC 的任务（{len(roc_drawn)}）: " + ", ".join(roc_drawn)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
