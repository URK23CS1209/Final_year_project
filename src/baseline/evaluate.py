"""
Evaluation Metrics & Analysis Module
------------------------------------
Calculates fold-wise metrics, confusion matrices, sensitivity, specificity,
F1-score, ROC-AUC, and parent-smear-level aggregated diagnostic decisions.
Strictly segregates Full Test Evaluation and Clean-Subset Sensitivity Analysis.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.50) -> Dict[str, Any]:
    """
    Computes standard binary classification metrics from ground truth labels and predicted probabilities.
    Class 1 = Anemic, Class 0 = Healthy.
    Threshold: Documented decision boundary (default 0.50).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Threshold probability to discrete class label
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity (True Positive Rate)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Specificity = TN / (TN + FP)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Balanced Accuracy = (Sensitivity + Specificity) / 2
    bal_acc = (rec + spec) / 2.0

    # ROC-AUC (valid only if both classes are present in y_true)
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = float('nan')
    else:
        roc_auc = float('nan')

    return {
        "accuracy": float(acc),
        "sensitivity": float(rec),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "balanced_accuracy": float(bal_acc),
        "roc_auc": float(roc_auc),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_samples": int(len(y_true))
    }


def compute_parent_smear_level_metrics(
    records: List[Dict],
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Parent-Smear-Level Evaluation:
    Aggregates crop-level predictions to parent-smear diagnoses by averaging
    the 12 predicted crop probabilities belonging to the same parent smear.
      p_smear = (1/12) * sum(p_crop_k)
      y_smear = 1 if p_smear >= threshold else 0
    """
    # Group by original_image_id
    parent_probs = {}
    parent_labels = {}

    for r in records:
        pid = r["original_image_id"]
        prob = float(r["pred_prob"])
        label = int(r["true_label"])

        if pid not in parent_probs:
            parent_probs[pid] = []
            parent_labels[pid] = label
        parent_probs[pid].append(prob)

    pids = sorted(list(parent_probs.keys()))
    y_true_patient = np.array([parent_labels[pid] for pid in pids], dtype=int)
    # Average the 12 crop probabilities
    y_prob_patient = np.array([np.mean(parent_probs[pid]) for pid in pids], dtype=float)

    metrics = compute_binary_metrics(y_true_patient, y_prob_patient, threshold=threshold)
    metrics["num_parent_smears"] = len(pids)
    metrics["evaluation_level"] = "Parent-Smear-Level (Aggregated 12 crops per smear)"
    return metrics


def evaluate_test_predictions(
    predictions: List[Dict],
    threshold: float = 0.50,
    conflicts_csv: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of test-set predictions:
      1. Full Test Set: Crop-level & Parent-smear-level metrics.
      2. Clean Non-Conflicting Subset: Separate sensitivity analysis
         excluding the 71 conflicting duplicate parents.
    """
    # 1. Full Test Set
    y_true_all = [p["true_label"] for p in predictions]
    y_prob_all = [p["pred_prob"] for p in predictions]

    full_crop_metrics = compute_binary_metrics(y_true_all, y_prob_all, threshold=threshold)
    full_smear_metrics = compute_parent_smear_level_metrics(predictions, threshold=threshold)

    results = {
        "full_test_crop_level": full_crop_metrics,
        "full_test_parent_smear_level": full_smear_metrics,
        "clean_subset_evaluated": False
    }

    # 2. Clean Subset Sensitivity Analysis
    if conflicts_csv and Path(conflicts_csv).exists():
        with open(conflicts_csv, 'r', encoding='utf-8') as f:
            conf_rows = list(csv.DictReader(f))
        conflicting_parents = set()
        for r in conf_rows:
            conflicting_parents.update(r["parent_ids"].split("; "))

        clean_preds = [p for p in predictions if p["original_image_id"] not in conflicting_parents]

        if len(clean_preds) > 0:
            y_true_clean = [p["true_label"] for p in clean_preds]
            y_prob_clean = [p["pred_prob"] for p in clean_preds]

            clean_crop_metrics = compute_binary_metrics(y_true_clean, y_prob_clean, threshold=threshold)
            clean_smear_metrics = compute_parent_smear_level_metrics(clean_preds, threshold=threshold)

            results["clean_subset_crop_level"] = clean_crop_metrics
            results["clean_subset_parent_smear_level"] = clean_smear_metrics
            results["clean_subset_excluded_parents"] = len(set(p["original_image_id"] for p in predictions) & conflicting_parents)
            results["clean_subset_evaluated"] = True

    return results


def format_evaluation_summary(results: Dict[str, Any], title: str = "Evaluation Summary") -> str:
    """Format evaluation results into a clean structured report."""
    lines = []
    lines.append(f"## {title}")
    lines.append("")

    full_crop = results["full_test_crop_level"]
    full_smear = results["full_test_parent_smear_level"]

    lines.append("### [A] Full Test Set Evaluation (All Held-Out Samples)")
    lines.append(f"• Decision Threshold: $\\tau = {full_crop['threshold']:.2f}$")
    lines.append(f"• Total Crops Evaluated: {full_crop['total_samples']:,} | Parent Smears: {full_smear['num_parent_smears']}")
    lines.append("")
    lines.append("| Metric | Crop-Level Score | Parent-Smear-Level Score | Details |")
    lines.append("| :--- | :---: | :---: | :--- |")
    lines.append(f"| **Accuracy** | {full_crop['accuracy']:.4f} | {full_smear['accuracy']:.4f} | Overall diagnostic accuracy |")
    lines.append(f"| **Sensitivity (Recall)** | {full_crop['sensitivity']:.4f} | {full_smear['sensitivity']:.4f} | Anemia detection rate ($TP / (TP + FN)$) |")
    lines.append(f"| **Specificity** | {full_crop['specificity']:.4f} | {full_smear['specificity']:.4f} | Healthy detection rate ($TN / (TN + FP)$) |")
    lines.append(f"| **Precision** | {full_crop['precision']:.4f} | {full_smear['precision']:.4f} | Positive predictive value ($TP / (TP + FP)$) |")
    lines.append(f"| **F1-Score** | {full_crop['f1_score']:.4f} | {full_smear['f1_score']:.4f} | Harmonic mean of Precision & Sensitivity |")
    lines.append(f"| **Balanced Accuracy** | {full_crop['balanced_accuracy']:.4f} | {full_smear['balanced_accuracy']:.4f} | $(\\text{{Sens}} + \\text{{Spec}}) / 2$ |")
    lines.append(f"| **ROC-AUC** | {full_crop['roc_auc']:.4f} | {full_smear['roc_auc']:.4f} | Area under ROC curve |")
    lines.append(f"| **Confusion Matrix** | TP={full_crop['tp']}, FP={full_crop['fp']}, TN={full_crop['tn']}, FN={full_crop['fn']} | TP={full_smear['tp']}, FP={full_smear['fp']}, TN={full_smear['tn']}, FN={full_smear['fn']} | Diagnostic counts |")
    lines.append("")

    if results.get("clean_subset_evaluated"):
        clean_crop = results["clean_subset_crop_level"]
        clean_smear = results["clean_subset_parent_smear_level"]
        lines.append("### [B] Clean-Subset Sensitivity Analysis (Excluding Conflicting Duplicate Parents)")
        lines.append(f"• Excluded Conflicting Parents in Test Fold: {results.get('clean_subset_excluded_parents', 0)}")
        lines.append(f"• Remaining Clean Crops: {clean_crop['total_samples']:,} | Remaining Clean Smears: {clean_smear['num_parent_smears']}")
        lines.append("")
        lines.append("| Metric | Clean Crop-Level | Clean Parent-Smear-Level | Details |")
        lines.append("| :--- | :---: | :---: | :--- |")
        lines.append(f"| **Accuracy** | {clean_crop['accuracy']:.4f} | {clean_smear['accuracy']:.4f} | Accuracy on unambiguous data |")
        lines.append(f"| **Sensitivity** | {clean_crop['sensitivity']:.4f} | {clean_smear['sensitivity']:.4f} | Clean anemia detection rate |")
        lines.append(f"| **Specificity** | {clean_crop['specificity']:.4f} | {clean_smear['specificity']:.4f} | Clean healthy detection rate |")
        lines.append(f"| **F1-Score** | {clean_crop['f1_score']:.4f} | {clean_smear['f1_score']:.4f} | Clean F1-Score |")
        lines.append(f"| **ROC-AUC** | {clean_crop['roc_auc']:.4f} | {clean_smear['roc_auc']:.4f} | Clean ROC-AUC |")
        lines.append("")

    return "\n".join(lines)
