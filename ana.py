#!/usr/bin/env python3
"""Aggregate fine-tuning results and create paired statistical reports.

Example:
    python ana.py --tasks finetune_v/vitl16/MER242526-emotion \
        finetune_v/vitl16/RAVDESS-emotion \
        --main_result vitl16-favor-e5-4layer-data48-8-newnewopt \
        FaVoR-112px-48f-8fps-e5

Each task is resolved below ``OUTPUT`` unless an absolute path is supplied.
The analysis never changes the original experiment directories; its files are
written below ``OUTPUT/ana/<task-name>/``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
)


# Category colours follow color.md: the Primary palette first (main_result keeps the
# leading colour #ED66A4), then the matching Secondary shades once a task exceeds the
# seven usable primary colours. #F1F1F1/#D5D5D5 stay reserved for backgrounds/grids.
PRIMARY_PALETTE = ["#ED66A4", "#8690C2", "#8CC7A1", "#F59E42", "#959BA5", "#C4B080", "#9E7BBC"]
SECONDARY_PALETTE = ["#C94F8B", "#6673A8", "#6FA987", "#D17E32", "#707780", "#A98F5C", "#805F9F"]
# Ink for structural elements (bar outlines, brackets, annotation text), not a data category colour.
INK = "#333333"
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
RNG_SEED = 20260909


def _category_color(index: int) -> str:
    """Return the colour for the configuration at ``index``, extending into the
    Secondary palette once the usable Primary colours are exhausted."""
    palette = PRIMARY_PALETTE + SECONDARY_PALETTE
    return palette[index % len(palette)]


def _ordered(names: list[str], main_name: str) -> list[str]:
    """Place ``main_name`` first and the remaining names in character order."""
    ordered = [main_name] if main_name in names else []
    ordered.extend(sorted(name for name in names if name != main_name))
    return ordered


def _p_stars(p_value: float) -> str:
    """Map a two-sided p-value to the conventional star annotation."""
    if not np.isfinite(p_value) or p_value >= 0.05:
        return "ns"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def _read_best_csv(path: Path) -> dict[str, float]:
    """Read both the current compact classification report and one-row reports."""
    rows = list(csv.reader(path.open(encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError("empty file")
    # Regression and future tabular reports are a normal header/value table.
    if len(rows) == 2 and rows[0] and rows[0][0].lower() in {"epoch", "metric"}:
        result = {}
        for name, value in zip(rows[0], rows[1]):
            if name != "epoch" and value.strip():
                result[name] = float(value.strip().rstrip("%")) / (100 if value.strip().endswith("%") else 1)
        return result

    result = {}
    # Compact layout: each name row is followed by a same-width values row.
    for names, values in zip(rows[1::2], rows[2::2]):
        for group, name, value in zip(rows[0], names, values):
            name, value = name.strip(), value.strip()
            if not name or not value:
                continue
            suffix = "" if group == "main" else f"_{group}"
            result[f"{name}{suffix}"] = float(value.rstrip("%")) / (100 if value.endswith("%") else 1)
    return result


def _find_predictions(result_dir: Path) -> Path | None:
    for name in ("best_predict.csv", "eval_best_predict.csv"):
        candidate = result_dir / "logs" / name
        if candidate.is_file():
            return candidate
    return None


def _load_result(result_dir: Path) -> dict:
    best = result_dir / "logs" / "best.csv"
    if not best.is_file():
        raise FileNotFoundError(f"missing {best}")
    prediction_path = _find_predictions(result_dir)
    return {
        "name": result_dir.name,
        "metrics": _read_best_csv(best),
        "predictions": pd.read_csv(prediction_path) if prediction_path else None,
        "source": result_dir,
    }


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    return sorted((column for column in frame if re.fullmatch(r"prob_\d+", column)), key=lambda x: int(x[5:]))


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.arange(probabilities.shape[1])
    y_pred = probabilities.argmax(axis=1)
    answer = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, labels=labels, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, labels=labels, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    actual, predicted, total = matrix.sum(axis=1), matrix.sum(axis=0), matrix.sum()
    specificity = []
    for class_index in labels:
        true_positive = matrix[class_index, class_index]
        false_positive = predicted[class_index] - true_positive
        false_negative = actual[class_index] - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        specificity.append(true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0)
    answer["specificity_macro"] = float(np.mean(specificity))
    answer["specificity_weighted"] = float(np.average(specificity, weights=actual)) if actual.sum() else float("nan")
    try:
        answer["log_loss"] = log_loss(y_true, probabilities, labels=labels)
        one_hot = np.eye(len(labels))[y_true]
        answer["auc_ovr_macro"] = roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        answer["auc_ovr_weighted"] = roc_auc_score(one_hot, probabilities, average="weighted", multi_class="ovr")
        answer["average_precision_macro"] = average_precision_score(one_hot, probabilities, average="macro")
        answer["average_precision_weighted"] = average_precision_score(one_hot, probabilities, average="weighted")
    except ValueError:
        pass
    return answer


def _bootstrap_metrics(y_true, probabilities, keys, count, rng):
    samples = {key: [] for key in keys}
    for indices in rng.integers(0, len(y_true), size=(count, len(y_true))):
        values = _classification_metrics(y_true[indices], probabilities[indices])
        for key in keys:
            samples[key].append(values.get(key, np.nan))
    return {key: np.asarray(values, dtype=float) for key, values in samples.items()}


def _paired_bootstrap_differences(y_true, main_prob, candidate_prob, keys, count, rng):
    """Return matched-sample bootstrap differences for every reported metric."""
    differences = {key: [] for key in keys}
    for indices in rng.integers(0, len(y_true), size=(count, len(y_true))):
        main_values = _classification_metrics(y_true[indices], main_prob[indices])
        candidate_values = _classification_metrics(y_true[indices], candidate_prob[indices])
        for key in keys:
            differences[key].append(main_values.get(key, np.nan) - candidate_values.get(key, np.nan))
    return {key: np.asarray(value, dtype=float) for key, value in differences.items()}


def _align_classification(reference: pd.DataFrame, other: pd.DataFrame):
    columns = _probability_columns(reference)
    if not columns or columns != _probability_columns(other):
        return None
    required = ["path", "true_label", *columns]
    if any(column not in reference or column not in other for column in required):
        return None
    joined = reference[required].merge(other[["path", "true_label", *columns]], on="path", suffixes=("_main", "_other"))
    joined = joined[joined.true_label_main == joined.true_label_other]
    if joined.empty:
        return None
    y = joined.true_label_main.to_numpy(dtype=int)
    main = joined[[f"{column}_main" for column in columns]].to_numpy(float)
    other_values = joined[[f"{column}_other" for column in columns]].to_numpy(float)
    return y, main, other_values


def _plot_distribution(kind, metric, distributions, output, main_name, best_name=None, p_value=float("nan")):
    names = _ordered(list(distributions), main_name)
    values = [distributions[name][metric] for name in names]
    values = [value[np.isfinite(value)] for value in values]
    fig, ax = plt.subplots(figsize=(max(7.5, len(names) * 0.8), 5.6))
    positions = np.arange(1, len(names) + 1)
    colors = [_category_color(index) for index in range(len(names))]
    if kind == "violin":
        artists = ax.violinplot(values, positions=positions, showmeans=True, showextrema=False, widths=0.75)
        for body, color in zip(artists["bodies"], colors):
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.6)
        artists["cmeans"].set_color(INK); artists["cmeans"].set_linewidth(1.0)
    elif kind == "bar":
        means = [np.mean(value) for value in values]
        errors = [np.std(value, ddof=1) if len(value) > 1 else 0 for value in values]
        ax.bar(positions, means, yerr=errors, capsize=3, color=colors, edgecolor=INK, linewidth=0.7)
    else:
        artists = ax.boxplot(values, positions=positions, patch_artist=True, showfliers=False, widths=0.6)
        for box, color in zip(artists["boxes"], colors):
            box.set_facecolor(color); box.set_alpha(0.65)
    rng = np.random.default_rng(RNG_SEED)
    for position, value, color in zip(positions, values, colors):
        ax.scatter(position + rng.uniform(-0.14, 0.14, len(value)), value, s=10, color=color, alpha=0.28,
                   edgecolors="none", zorder=3)
    main_index = names.index(main_name)
    # Reference line at main_result's mean metric, drawn in its colour on every plot kind.
    if len(values[main_index]):
        ax.axhline(float(np.mean(values[main_index])), color=colors[main_index], linestyle="--",
                   linewidth=1.1, zorder=1)
    # Paired-bootstrap significance bracket between main_result and the best remaining configuration.
    if best_name is not None and best_name in names and len(values[main_index]):
        best_index = names.index(best_name)
        if len(values[best_index]):
            merged = np.concatenate([value for value in values if value.size])
            span = float(np.max(merged) - np.min(merged))
            step = span * 0.07 if span > 0 else 1e-6
            main_top = float(np.max(values[main_index]))
            best_top = float(np.max(values[best_index]))
            # Clear every configuration's replicates so no scatter dot crosses the bracket.
            bracket_y = float(np.max(merged)) + step
            x1, x2 = positions[main_index], positions[best_index]
            ax.plot([x1, x2], [bracket_y, bracket_y], color=INK, linewidth=1.1, zorder=6)
            for top, x in ((main_top, x1), (best_top, x2)):
                ax.plot([x, x], [top, bracket_y], color=INK, linewidth=1.1, zorder=6)
            text_y = bracket_y + 0.7 * step
            ax.text((x1 + x2) / 2, text_y, _p_stars(p_value), ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=INK, zorder=7)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, max(top, text_y + step))
    ax.set_xticks(positions, names, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: bootstrap distributions (main: {main_name})")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(results, output, bootstrap_count, rng):
    fig, ax = plt.subplots(figsize=(7, 6.2))
    for index, item in enumerate(results):
        frame, cols = item["predictions"], _probability_columns(item["predictions"])
        y = frame.true_label.to_numpy(dtype=int)
        probabilities = frame[cols].to_numpy(float)
        one_hot = np.eye(len(cols))[y]
        flat_y, flat_prob = one_hot.ravel(), probabilities.ravel()
        fpr, tpr, thresholds = roc_curve(flat_y, flat_prob)
        auc = roc_auc_score(flat_y, flat_prob)
        grid = np.linspace(0, 1, 101)
        curves, aucs = [], []
        for sample in rng.integers(0, len(y), size=(bootstrap_count, len(y))):
            y_b, p_b = y[sample], probabilities[sample]
            y_hot = np.eye(len(cols))[y_b].ravel()
            if np.unique(y_hot).size < 2:
                continue
            boot_fpr, boot_tpr, _ = roc_curve(y_hot, p_b.ravel())
            curves.append(np.interp(grid, boot_fpr, boot_tpr))
            aucs.append(roc_auc_score(y_hot, p_b.ravel()))
        color = _category_color(index)
        label = f"{item['name']} (AUC={auc:.3f}, 95% CI {np.quantile(aucs, .025):.3f}-{np.quantile(aucs, .975):.3f})"
        ax.plot(fpr, tpr, color=color, linewidth=1.9, label=label)
        ax.fill_between(grid, np.quantile(curves, .025, axis=0), np.quantile(curves, .975, axis=0), color=color, alpha=.12)
        for marker, threshold in zip(MARKERS[:4], (0.90, 0.75, 0.50, 0.25)):
            mask = thresholds >= threshold
            if mask.any():
                point = np.flatnonzero(mask)[-1]
                ax.scatter(fpr[point], tpr[point], color=color, marker=marker, s=35, zorder=4)
    ax.plot([0, 1], [0, 1], color="#959BA5", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate", title="Micro-average ROC with bootstrap 95% CI")
    ax.grid(color="#F1F1F1", linewidth=.8)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _task_output_name(task: Path) -> str:
    return "-".join(task.parts[-3:]) if len(task.parts) >= 3 else task.name


def analyse_task(task_dir: Path, main_name: str, root: Path, n_bootstrap: int):
    result_dirs = sorted(path.parent.parent for path in task_dir.glob("*/logs/best.csv"))
    results, skipped = [], []
    for directory in result_dirs:
        try:
            results.append(_load_result(directory))
        except (ValueError, OSError) as error:
            skipped.append(f"{directory.name}: {error}")
    if not results:
        raise RuntimeError(f"no readable best.csv beneath {task_dir}")
    output = root / "OUTPUT" / "ana" / _task_output_name(task_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_name = {item["name"]: item for item in results}
    ordered_overall = _ordered([item["name"] for item in results], main_name)
    pd.DataFrame([{"configuration": name, **by_name[name]["metrics"]} for name in ordered_overall]).to_csv(output / "overall.csv", index=False)

    main = next((item for item in results if item["name"] == main_name), None)
    if main is None:
        raise ValueError(f"main result {main_name!r} is not a readable configuration in {task_dir}")
    if main["predictions"] is None or not _probability_columns(main["predictions"]):
        (output / "README.txt").write_text("Regression or predictions unavailable: wrote overall.csv; ROC and classification distribution plots were skipped.\n", encoding="utf-8")
        return output, len(results), skipped

    # Limit graph metrics to those actually reproducible from all classification predictions.
    valid = [item for item in results if item["predictions"] is not None and _probability_columns(item["predictions"])]
    ordered_valid = _ordered([item["name"] for item in valid], main_name)
    keys = list(_classification_metrics(main["predictions"].true_label.to_numpy(int), main["predictions"][_probability_columns(main["predictions"])].to_numpy(float)))
    rng = np.random.default_rng(RNG_SEED)
    distributions = {}
    for name in ordered_valid:
        item = by_name[name]
        frame, cols = item["predictions"], _probability_columns(item["predictions"])
        distributions[name] = _bootstrap_metrics(frame.true_label.to_numpy(int), frame[cols].to_numpy(float), keys, n_bootstrap, rng)

    # Paired bootstrap differences (and their two-sided p-values) provide a compact,
    # auditable significance table; the same p-values annotate the distribution plots.
    rows = []
    paired_p = {}
    for name in ordered_valid:
        if name == main_name:
            continue
        aligned = _align_classification(main["predictions"], by_name[name]["predictions"])
        if aligned is None:
            continue
        y, main_prob, candidate_prob = aligned
        paired_differences = _paired_bootstrap_differences(
            y, main_prob, candidate_prob, keys, n_bootstrap, rng
        )
        paired_p[name] = {}
        for metric in keys:
            diff = paired_differences[metric]
            p_value = 2 * min(np.nanmean(diff <= 0), np.nanmean(diff >= 0))
            paired_p[name][metric] = p_value
            rows.append({"configuration": name, "metric": metric, "n_paired": len(y),
                         "main_minus_configuration": np.nanmean(diff), "ci95_low": np.nanquantile(diff, .025),
                         "ci95_high": np.nanquantile(diff, .975),
                         "bootstrap_two_sided_p": p_value})
    pd.DataFrame(rows).to_csv(output / "significance_vs_main.csv", index=False)

    # The bracket only compares main_result against the best of the remaining configurations.
    significance = {}
    for metric in keys:
        candidates = [name for name in paired_p if np.isfinite(np.nanmean(distributions[name][metric]))]
        if not candidates:
            continue
        best = max(candidates, key=lambda name: float(np.nanmean(distributions[name][metric])))
        significance[metric] = (best, paired_p[best][metric])
    for kind in ("violin", "bar", "box"):
        folder = output / kind
        folder.mkdir(exist_ok=True)
        for metric in keys:
            best, p_value = significance.get(metric, (None, float("nan")))
            _plot_distribution(kind, metric, distributions, folder / f"{metric}.png", main_name,
                               best_name=best, p_value=p_value)
    _plot_roc([by_name[name] for name in ordered_valid], output / "roc_micro.png", n_bootstrap, rng)
    (output / "README.txt").write_text(
        "overall.csv contains every metric reported by each best.csv (main_result first, then alphabetical). Distribution plots show non-parametric bootstrap replicates; dots are individual bootstrap estimates. The dashed horizontal line is main_result's mean and the bracket (with ns/*/**/***/****) tests main_result against the best remaining configuration via the paired bootstrap two-sided p-value. significance_vs_main.csv reports the paired bootstrap differences after matching samples by path. roc_micro.png pools one-vs-rest classes; four marker shapes show thresholds 0.90, 0.75, 0.50, and 0.25.\n"
        + ("Skipped results:\n" + "\n".join(skipped) + "\n" if skipped else ""), encoding="utf-8")
    return output, len(results), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", required=True, help="Task directories relative to OUTPUT, or absolute paths.")
    parser.add_argument("--main_result", nargs="+", required=True, help="Main configuration(s), in the same order as --tasks, or one name for every task.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap replicates per configuration (default: 1000).")
    args = parser.parse_args()
    if args.n_bootstrap < 20:
        parser.error("--n-bootstrap must be at least 20")
    if len(args.main_result) not in (1, len(args.tasks)):
        parser.error("--main_result needs one name or exactly one name per task")
    root = Path(__file__).resolve().parent
    main_names = args.main_result if len(args.main_result) > 1 else args.main_result * len(args.tasks)
    for task_text, main_name in zip(args.tasks, main_names):
        task = Path(task_text)
        task_dir = task if task.is_absolute() else root / "OUTPUT" / task
        # Older command notes abbreviated the ViT-L directory as ``vit16``;
        # experiment folders in this repository use the explicit ``vitl16``.
        if not task_dir.is_dir() and not task.is_absolute() and "vit16" in task.parts:
            task_dir = root / "OUTPUT" / Path(*["vitl16" if part == "vit16" else part for part in task.parts])
        output, count, skipped = analyse_task(task_dir, main_name, root, args.n_bootstrap)
        print(f"{task_dir}: {count} configurations -> {output}" + (f" ({len(skipped)} skipped)" if skipped else ""))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ana.py: error: {error}", file=sys.stderr)
        raise SystemExit(2)
