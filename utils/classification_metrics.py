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
    fieldnames = list(history[0])
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
