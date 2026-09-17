"""Evaluation, reporting, and plotting helpers for fine-tuning."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(function, *args, default=float("nan"), **kwargs):
    try:
        return float(function(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return default


def classification_metrics(truth, logits, num_classes):
    """Return scalar/per-class metrics and the integer confusion matrix."""
    truth = np.asarray(truth, dtype=int)
    logits = np.asarray(logits, dtype=np.float64)
    predictions = logits.argmax(axis=1)
    labels = np.arange(num_classes)
    # Stable softmax is needed for log-loss, AUC, and average precision.
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    matrix = confusion_matrix(truth, predictions, labels=labels)

    metrics = {
        "accuracy": _safe_metric(accuracy_score, truth, predictions, default=0.0),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, truth, predictions, default=0.0),
        "precision_macro": _safe_metric(precision_score, truth, predictions, labels=labels,
                                         average="macro", zero_division=0, default=0.0),
        "precision_weighted": _safe_metric(precision_score, truth, predictions, labels=labels,
                                            average="weighted", zero_division=0, default=0.0),
        "precision_micro": _safe_metric(precision_score, truth, predictions, labels=labels,
                                         average="micro", zero_division=0, default=0.0),
        "recall_macro": _safe_metric(recall_score, truth, predictions, labels=labels,
                                      average="macro", zero_division=0, default=0.0),
        "recall_weighted": _safe_metric(recall_score, truth, predictions, labels=labels,
                                         average="weighted", zero_division=0, default=0.0),
        "recall_micro": _safe_metric(recall_score, truth, predictions, labels=labels,
                                      average="micro", zero_division=0, default=0.0),
        "f1_macro": _safe_metric(f1_score, truth, predictions, labels=labels,
                                  average="macro", zero_division=0, default=0.0),
        "f1_weighted": _safe_metric(f1_score, truth, predictions, labels=labels,
                                     average="weighted", zero_division=0, default=0.0),
        "f1_micro": _safe_metric(f1_score, truth, predictions, labels=labels,
                                  average="micro", zero_division=0, default=0.0),
        "jaccard_macro": _safe_metric(jaccard_score, truth, predictions, labels=labels,
                                       average="macro", zero_division=0, default=0.0),
        "jaccard_weighted": _safe_metric(jaccard_score, truth, predictions, labels=labels,
                                          average="weighted", zero_division=0, default=0.0),
        "cohen_kappa": _safe_metric(cohen_kappa_score, truth, predictions, labels=labels, default=0.0),
        "matthews_corrcoef": _safe_metric(matthews_corrcoef, truth, predictions, default=0.0),
        "hamming_loss": _safe_metric(hamming_loss, truth, predictions, default=0.0),
        "log_loss": _safe_metric(log_loss, truth, probabilities, labels=labels, default=float("nan")),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0
    )
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    specificities = []
    for index in labels:
        true_positive = matrix[index, index]
        false_positive = predicted[index] - true_positive
        false_negative = actual[index] - true_positive
        true_negative = matrix.sum() - true_positive - false_positive - false_negative
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
        specificities.append(specificity)
        metrics[f"class_{index}_precision"] = float(precision[index])
        metrics[f"class_{index}_recall"] = float(recall[index])
        metrics[f"class_{index}_f1"] = float(f1[index])
        metrics[f"class_{index}_specificity"] = float(specificity)
        metrics[f"class_{index}_support"] = int(support[index])
    metrics["specificity_macro"] = float(np.mean(specificities))
    metrics["specificity_weighted"] = float(np.average(specificities, weights=actual)) if actual.sum() else 0.0

    binary_truth = (truth[:, None] == labels).astype(int)
    auc_values = []
    ap_values = []
    for index in labels:
        has_both = binary_truth[:, index].min() != binary_truth[:, index].max()
        auc = _safe_metric(roc_auc_score, binary_truth[:, index], probabilities[:, index], default=float("nan")) \
            if has_both else float("nan")
        ap = _safe_metric(average_precision_score, binary_truth[:, index], probabilities[:, index], default=float("nan")) \
            if binary_truth[:, index].sum() else float("nan")
        metrics[f"class_{index}_auc"] = auc
        metrics[f"class_{index}_average_precision"] = ap
        auc_values.append(auc)
        ap_values.append(ap)
    metrics["auc_ovr_macro"] = float(np.nanmean(auc_values)) if not np.all(np.isnan(auc_values)) else float("nan")
    metrics["average_precision_macro"] = float(np.nanmean(ap_values)) if not np.all(np.isnan(ap_values)) else float("nan")
    valid_auc = [(value, actual[index]) for index, value in enumerate(auc_values) if not np.isnan(value)]
    valid_ap = [(value, actual[index]) for index, value in enumerate(ap_values) if not np.isnan(value)]
    metrics["auc_ovr_weighted"] = (float(np.average([value for value, _ in valid_auc],
                                                     weights=[weight for _, weight in valid_auc]))
                                    if valid_auc else float("nan"))
    metrics["average_precision_weighted"] = (float(np.average([value for value, _ in valid_ap],
                                                                weights=[weight for _, weight in valid_ap]))
                                               if valid_ap else float("nan"))
    metrics["f1"] = metrics["f1_macro"]
    return metrics, matrix, predictions, probabilities


def multilabel_metrics(truth, logits, num_classes, threshold=0.5):
    """Return scalar/per-class metrics for multi-label targets.

    ``truth`` is a ``[N, num_classes]`` 0/1 matrix and ``logits`` an
    ``[N, num_classes]`` array of raw scores.  Predictions use a fixed
    ``threshold`` on the sigmoid probabilities; there is no argmax because
    classes are not mutually exclusive.
    """
    truth = np.asarray(truth, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if truth.shape != logits.shape:
        raise ValueError(f"truth shape {truth.shape} does not match logits shape {logits.shape}")

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= threshold).astype(int)
    labels = np.arange(num_classes)
    # Subset (exact-match) accuracy: every one of the C labels must agree.
    subset_accuracy = float(np.all(predictions == truth, axis=1).mean())

    # sklearn's log_loss treats a 2D y_pred as a multiclass distribution and
    # renormalizes every row to sum to one, which is meaningless for
    # independent sigmoids.  Compute the elementwise BCE instead.
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    metrics = {
        "accuracy": subset_accuracy,
        "hamming_loss": _safe_metric(hamming_loss, truth, predictions, default=0.0),
        "log_loss": float(-(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped)).mean()),
    }
    for average in ("macro", "weighted", "micro"):
        # ``micro`` over a multi-label indicator is the aggregate TP/FP/FN view.
        metrics[f"precision_{average}"] = _safe_metric(
            precision_score, truth, predictions, average=average, zero_division=0, default=0.0)
        metrics[f"recall_{average}"] = _safe_metric(
            recall_score, truth, predictions, average=average, zero_division=0, default=0.0)
        metrics[f"f1_{average}"] = _safe_metric(
            f1_score, truth, predictions, average=average, zero_division=0, default=0.0)
        metrics[f"jaccard_{average}"] = _safe_metric(
            jaccard_score, truth, predictions, average=average, zero_division=0, default=0.0)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0
    )
    auc_values, ap_values = [], []
    for index in labels:
        column = truth[:, index]
        metrics[f"class_{index}_precision"] = float(precision[index])
        metrics[f"class_{index}_recall"] = float(recall[index])
        metrics[f"class_{index}_f1"] = float(f1[index])
        metrics[f"class_{index}_support"] = int(support[index])
        has_both = column.min() != column.max()
        auc = _safe_metric(roc_auc_score, column, probabilities[:, index], default=float("nan")) \
            if has_both else float("nan")
        ap = _safe_metric(average_precision_score, column, probabilities[:, index], default=float("nan")) \
            if column.sum() else float("nan")
        metrics[f"class_{index}_auc"] = auc
        metrics[f"class_{index}_average_precision"] = ap
        auc_values.append(auc)
        ap_values.append(ap)
    metrics["auc_ovr_macro"] = float(np.nanmean(auc_values)) if not np.all(np.isnan(auc_values)) else float("nan")
    metrics["average_precision_macro"] = float(np.nanmean(ap_values)) if not np.all(np.isnan(ap_values)) else float("nan")
    valid_auc = [(value, int(support[index])) for index, value in enumerate(auc_values) if not np.isnan(value)]
    valid_ap = [(value, int(support[index])) for index, value in enumerate(ap_values) if not np.isnan(value)]
    metrics["auc_ovr_weighted"] = (float(np.average([value for value, _ in valid_auc],
                                                     weights=[weight for _, weight in valid_auc]))
                                    if valid_auc else float("nan"))
    metrics["average_precision_weighted"] = (float(np.average([value for value, _ in valid_ap],
                                                                weights=[weight for _, weight in valid_ap]))
                                               if valid_ap else float("nan"))
    metrics["f1"] = metrics["f1_macro"]
    return metrics, predictions, probabilities


def _format_label_set(vector):
    """Render a 0/1 vector as the pipe-separated one-based form used by the CSV labels."""
    return "|".join(str(index + 1) for index, value in enumerate(vector) if value)


def write_multilabel_predictions_csv(output, paths, truth, predictions, logits, probabilities):
    """Write per-sample multi-hot labels, logits, and probabilities."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    class_count = logits.shape[1]
    fieldnames = ["path", "true_label", "predicted_label", "correct", "confidence"]
    fieldnames += [f"logit_{index}" for index in range(class_count)]
    fieldnames += [f"prob_{index}" for index in range(class_count)]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path, true, prediction, sample_logits, sample_probabilities in zip(
                paths, truth, predictions, logits, probabilities):
            row = {
                "path": path,
                "true_label": _format_label_set(true),
                "predicted_label": _format_label_set(prediction),
                "correct": int(bool(np.all(np.asarray(true) == np.asarray(prediction)))),
                "confidence": float(np.mean(sample_probabilities)),
            }
            row.update({f"logit_{index}": float(value) for index, value in enumerate(sample_logits)})
            row.update({f"prob_{index}": float(value) for index, value in enumerate(sample_probabilities)})
            writer.writerow(row)


def save_per_class_f1_chart(metrics, num_classes, output, epoch):
    """Bar chart of per-class F1, exposing which of the C labels are learned."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [metrics.get(f"class_{index}_f1", 0.0) for index in range(num_classes)]
    fig, ax = plt.subplots(figsize=(max(7, num_classes * 0.5), 4.5))
    ax.bar(range(num_classes), values, color="#4C72B0")
    ax.set(xticks=range(num_classes), xlabel="Class index", ylabel="F1",
           ylim=(0, 1), title=f"Best per-class F1 (epoch {epoch})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_multilabel_best_reports(output_dir, epoch, metrics, paths, truth, predictions, logits, probabilities):
    """Write best.csv, best_predict.csv, and the per-class F1 chart for a multi-label run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scalars = {key: value for key, value in metrics.items() if not key.startswith("class_")}
    with (output_dir / "best.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *scalars])
        writer.writeheader()
        writer.writerow({"epoch": epoch, **scalars})
    write_multilabel_predictions_csv(output_dir / "best_predict.csv", paths, truth, predictions, logits, probabilities)
    save_per_class_f1_chart(metrics, logits.shape[1], output_dir / "per_class_f1_best.png", epoch)


def write_metrics_csv(output, metrics, epoch=None):
    """Write the compact sample.csv-style classification report.

    The detailed per-sample predictions remain in ``best_predict.csv``. This
    report intentionally contains only aggregate metrics and uses percentages
    so it can be compared directly with the supplied sample.csv template.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("accuracy", "precision", "precision", "precision",
         "accuracy", "precision_macro", "precision_weighted", "precision_micro"),
        ("balanced_accuracy", "recall", "recall", "recall",
         "balanced_accuracy", "recall_macro", "recall_weighted", "recall_micro"),
        ("cohen_kappa", "f1", "f1", "f1",
         "cohen_kappa", "f1_macro", "f1_weighted", "f1_micro"),
        ("", "specificity", "specificity", "",
         "", "specificity_macro", "specificity_weighted", ""),
        ("", "auc_ovr", "auc_ovr", "",
         "", "auc_ovr_macro", "auc_ovr_weighted", ""),
        ("", "average_precision", "average_precision", "",
         "", "average_precision_macro", "average_precision_weighted", ""),
    ]

    def percentage(key):
        value = metrics.get(key, float("nan"))
        return "" if value is None or not np.isfinite(value) else f"{100.0 * value:.2f}%"

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["main", "macro", "weighted", "micro"])
        for main, macro, weighted, micro, main_key, macro_key, weighted_key, micro_key in rows:
            writer.writerow([main, macro, weighted, micro])
            writer.writerow([
                percentage(main_key) if main_key else "",
                percentage(macro_key) if macro_key else "",
                percentage(weighted_key) if weighted_key else "",
                percentage(micro_key) if micro_key else "",
            ])

def write_predictions_csv(output, paths, truth, predictions, logits, probabilities):
    """Write per-sample labels, logits, and probabilities."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    class_count = logits.shape[1]
    fieldnames = ["path", "true_label", "predicted_label", "correct", "confidence"]
    fieldnames += [f"logit_{index}" for index in range(class_count)]
    fieldnames += [f"prob_{index}" for index in range(class_count)]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path, true, prediction, sample_logits, sample_probabilities in zip(
                paths, truth, predictions, logits, probabilities):
            row = {
                "path": path,
                "true_label": int(true),
                "predicted_label": int(prediction),
                "correct": int(int(true) == int(prediction)),
                "confidence": float(np.max(sample_probabilities)),
            }
            row.update({f"logit_{index}": float(value) for index, value in enumerate(sample_logits)})
            row.update({f"prob_{index}": float(value) for index, value in enumerate(sample_probabilities)})
            writer.writerow(row)


def save_evaluation_reports(outmetric, outpredict, outcm, metrics, matrix, paths,
                            truth, predictions, logits, probabilities, epoch=None):
    """Write metrics CSV, per-sample prediction CSV, and a confusion matrix."""
    write_metrics_csv(outmetric, metrics, epoch=epoch)
    write_predictions_csv(outpredict, paths, truth, predictions, logits, probabilities)
    save_confusion_matrix(matrix, outcm, epoch)


def save_best_reports(output_dir, epoch, metrics, matrix, paths, truth, predictions, logits, probabilities):
    """Write best.csv, best_predict.csv, and the best confusion-matrix figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_reports(
        output_dir / "best.csv",
        output_dir / "best_predict.csv",
        output_dir / "confusion_matrix_best.png",
        metrics, matrix, paths, truth, predictions, logits, probabilities, epoch,
    )


def save_confusion_matrix(matrix, output, epoch):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[0] * 0.8), max(5, matrix.shape[0] * 0.7)))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Count")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    labels = np.arange(matrix.shape[0])
    ax.set(xticks=labels, yticks=labels, xlabel="Predicted label", ylabel="True label",
           title=f"Best confusion matrix (epoch {epoch})")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_metric_curves(history, output):
    """Plot the training history into one report figure."""
    if not history:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = [key for key in history[0] if key not in {"epoch", "matrix"}
            and not key.startswith("cm_")
            and "class_" not in key]
    columns = 3
    rows = math.ceil(len(keys) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, max(4, rows * 3.2)), squeeze=False)
    epochs = [item["epoch"] for item in history]
    for axis, key in zip(axes.flat, keys):
        axis.plot(epochs, [item[key] for item in history], marker="o", markersize=2, linewidth=1.3)
        axis.set_title(key)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(keys):]:
        axis.axis("off")
    fig.suptitle("Training and validation metrics")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_history_csv(history, output):
    """Write one row per epoch, including the validation confusion matrix."""
    if not history:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # history 行可能跨越不同 schema 版本：旧版把混淆矩阵存成单个 `matrix`
    # 列，当前代码则拆成每格一列的 cm_{r}_{c}。若以 history[0] 的键为
    # fieldnames，新追加的含 cm_* 的行会让 DictWriter 抛
    # "dict contains fields not in fieldnames"。改为取全行键的并集（先见序），
    # 纯新目录（各行键一致）下结果与原来完全相同。
    fieldnames, seen = [], set()
    for row in history:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def history_from_csv(path):
    """Load history rows and reconstruct each epoch's confusion matrix."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No history records found in {path}")

    matrix_columns = {}
    for name in rows[0]:
        if name.startswith("cm_"):
            try:
                row, column = (int(value) for value in name[3:].split("_", 1))
            except ValueError:
                continue
            matrix_columns[(row, column)] = name
    if matrix_columns:
        size = max(max(row, column) for row, column in matrix_columns) + 1
    else:
        size = 0

    history = []
    for row in rows:
        item = {}
        for name, value in row.items():
            if name.startswith("cm_"):
                continue
            if name == "epoch":
                item[name] = int(value)
            else:
                item[name] = float(value)
        matrix = np.zeros((size, size), dtype=int)
        for (matrix_row, matrix_column), name in matrix_columns.items():
            matrix[matrix_row, matrix_column] = int(float(row[name]))
        item["matrix"] = matrix if size else None
        history.append(item)
    return history
