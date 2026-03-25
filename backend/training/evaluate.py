"""
Model Evaluation — Metrics & Reporting
=======================================
Computes: Accuracy, Precision, Recall, F1, Confusion Matrix.
This is interview gold — shows you understand model assessment.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def evaluate_model(model, dataloader: DataLoader, device, class_labels: list,
                   is_temporal: bool = False):
    """
    Full evaluation with all standard metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    num_classes = len(class_labels)

    print_report(all_labels, all_preds, class_labels)


def print_report(y_true: np.ndarray, y_pred: np.ndarray, class_labels: list):
    """Print comprehensive evaluation report."""
    num_classes = len(class_labels)

    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"\n  Overall Accuracy: {accuracy:.4f} ({np.sum(y_true == y_pred)}/{len(y_true)})")

    # Per-class metrics
    print(f"\n  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 62)

    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(num_classes):
        tp = np.sum((y_pred == i) & (y_true == i))
        fp = np.sum((y_pred == i) & (y_true != i))
        fn = np.sum((y_pred != i) & (y_true == i))
        support = np.sum(y_true == i)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        print(f"  {class_labels[i]:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")

    # Weighted averages
    total = sum(supports)
    if total > 0:
        w_precision = sum(p * s for p, s in zip(precisions, supports)) / total
        w_recall = sum(r * s for r, s in zip(recalls, supports)) / total
        w_f1 = sum(f * s for f, s in zip(f1s, supports)) / total
        print("  " + "-" * 62)
        print(f"  {'Weighted Avg':<20} {w_precision:>10.4f} {w_recall:>10.4f} {w_f1:>10.4f} {total:>10d}")

    # Macro averages
    m_precision = np.mean(precisions)
    m_recall = np.mean(recalls)
    m_f1 = np.mean(f1s)
    print(f"  {'Macro Avg':<20} {m_precision:>10.4f} {m_recall:>10.4f} {m_f1:>10.4f} {total:>10d}")

    # Confusion Matrix
    print(f"\n  Confusion Matrix:")
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # Header
    header = "  " + " " * 18
    for label in class_labels:
        header += f"{label[:8]:>10}"
    print(header)
    print("  " + "-" * (18 + 10 * num_classes))

    for i in range(num_classes):
        row = f"  {class_labels[i]:<18}"
        for j in range(num_classes):
            val = cm[i][j]
            marker = " ✓" if i == j and val > 0 else ""
            row += f"{val:>10}"
        print(row)

    return {
        "accuracy": accuracy,
        "precision_per_class": precisions,
        "recall_per_class": recalls,
        "f1_per_class": f1s,
        "confusion_matrix": cm.tolist(),
    }
