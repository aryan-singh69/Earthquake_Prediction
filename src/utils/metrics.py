"""
Evaluation metrics for earthquake detection models.

Provides:
  - compute_all_metrics()       : Full suite of classification metrics
  - save_metrics_to_json()      : Persist results as JSON
  - plot_confusion_matrix()     : Save confusion matrix heatmap as PNG
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Compute full suite of binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1), array-like.
        y_pred: Predicted labels (0 or 1), array-like.
        y_prob: Predicted probabilities for positive class (optional, for ROC-AUC).

    Returns:
        Dict containing accuracy, precision, recall, f1, confusion_matrix,
        classification_report, and roc_auc (if y_prob provided).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["Noise", "Earthquake"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # ROC-AUC requires probability scores
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError as e:
            # Can happen if only one class present in y_true
            metrics["roc_auc"] = None
            metrics["roc_auc_error"] = str(e)
    else:
        metrics["roc_auc"] = None

    return metrics


def save_metrics_to_json(metrics, output_path):
    """
    Save metrics dict to a JSON file.

    Args:
        metrics:     Dict from compute_all_metrics().
        output_path: Destination file path (e.g., models/metrics/evaluation_report.json).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path,
                          labels=None, title="Confusion Matrix"):
    """
    Plot and save a confusion matrix heatmap.

    Args:
        y_true:      Ground truth labels.
        y_pred:      Predicted labels.
        output_path: Destination PNG path (e.g., models/metrics/confusion_matrix.png).
        labels:      Class label names (default: ["Noise", "Earthquake"]).
        title:       Plot title.
    """
    if labels is None:
        labels = ["Noise", "Earthquake"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="gray",
        annot_kws={"size": 14},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"  Confusion matrix saved → {output_path}")


def print_metrics_summary(metrics):
    """
    Print a clean, readable metrics summary to the terminal.

    Args:
        metrics: Dict from compute_all_metrics().
    """
    print("\n" + "=" * 60)
    print("         EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")

    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    else:
        roc_err = metrics.get("roc_auc_error", "probabilities not provided")
        print(f"  ROC-AUC   : N/A ({roc_err})")

    cm = metrics["confusion_matrix"]
    print("\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Noise   Earthquake")
    print(f"  Actual Noise      {cm[0][0]:>6d}   {cm[0][1]:>6d}")
    print(f"  Actual Earthquake {cm[1][0]:>6d}   {cm[1][1]:>6d}")

    report = metrics["classification_report"]
    print("\n  Per-Class Report:")
    print(f"  {'Class':<14s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"  {'-'*54}")
    for cls_name in ["Noise", "Earthquake"]:
        cls = report[cls_name]
        print(f"  {cls_name:<14s} {cls['precision']:>10.4f} {cls['recall']:>10.4f} "
              f"{cls['f1-score']:>10.4f} {cls['support']:>10.0f}")

    print("=" * 60 + "\n")
