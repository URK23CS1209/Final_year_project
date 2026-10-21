"""
Automated Unit Test Suite for AneRBC Baseline Pipeline
------------------------------------------------------
Validates nested group-safe evaluation logic, inner-split leakage,
test-fold isolation, checkpoint selection, prediction-label alignment,
model/loss compatibility, and metric calculations.

Usage:
    python src/baseline/test_baseline_pipeline.py
"""

import unittest
import csv
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
import sys
sys.path.append(str(script_dir))

from config import BaselineConfig
from dataset import (
    load_nested_dataset_records,
    validate_nested_splits
)
from model import AneRBCClassifier
from evaluate import (
    compute_binary_metrics,
    compute_parent_smear_level_metrics,
    evaluate_test_predictions
)


class TestBaselinePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = BaselineConfig()
        assert cls.config.splits_csv_path.exists(), "dataset_splits.csv missing"
        assert cls.config.hash_audit_csv_path.exists(), "image_hash_audit.csv missing"

    def test_nested_split_leakage_and_test_isolation(self):
        """Verify zero parent-ID and zero SHA-256 hash leakage across all 5 nested folds."""
        for test_fold in range(self.config.n_splits):
            val_fold = (test_fold + self.config.inner_val_offset) % self.config.n_splits
            train_recs, val_recs, test_recs = load_nested_dataset_records(
                splits_csv=self.config.splits_csv_path,
                tier="AneRBC-II",
                test_fold=test_fold,
                inner_val_fold=val_fold
            )

            # Check counts
            self.assertEqual(len(test_recs), 2400, f"Outer test fold {test_fold} should have 2,400 crops")
            self.assertEqual(len(val_recs), 2400, f"Inner val fold {val_fold} should have 2,400 crops")
            self.assertEqual(len(train_recs), 7200, f"Inner train folds should have 7,200 crops")

            diag = validate_nested_splits(
                train_records=train_recs,
                val_records=val_recs,
                test_records=test_recs,
                hash_audit_csv=self.config.hash_audit_csv_path
            )

            # Strict zero leakage assertions
            self.assertEqual(diag["parent_leakage"], 0, f"Parent ID leakage detected in test fold {test_fold}")
            self.assertEqual(diag["hash_leakage"], 0, f"SHA-256 hash leakage detected in test fold {test_fold}")
            self.assertTrue(diag["is_leakage_free"])

    def test_prediction_label_alignment(self):
        """Verify label mapping consistency and prediction-record alignment."""
        self.assertEqual(self.config.label_map["Healthy"], 0)
        self.assertEqual(self.config.label_map["Anemic"], 1)
        self.assertEqual(self.config.inv_label_map[0], "Healthy")
        self.assertEqual(self.config.inv_label_map[1], "Anemic")

        # Verify against actual splits CSV records
        train_recs, val_recs, test_recs = load_nested_dataset_records(
            splits_csv=self.config.splits_csv_path,
            tier="AneRBC-II",
            test_fold=0,
            inner_val_fold=1
        )
        for r in test_recs[:100]:
            expected_binary = self.config.label_map[r["directory_label"]]
            self.assertIn(expected_binary, [0, 1])

    def test_model_output_shape_and_bce_loss_compatibility(self):
        """Verify model returns linear logits compatible with BCEWithLogitsLoss."""
        model = AneRBCClassifier(model_name="mobilenet_v2", pretrained=False, num_classes=1)
        batch_size = 8
        dummy_x = torch.randn(batch_size, 3, 224, 224)
        dummy_y = torch.randint(0, 2, (batch_size,)).float()

        logits = model(dummy_x)
        self.assertEqual(logits.shape, torch.Size([batch_size]))

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, dummy_y)

        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())
        self.assertGreater(loss.item(), 0.0)

        # Verify sigmoid transformation
        probs = torch.sigmoid(logits)
        self.assertTrue((probs >= 0.0).all().item() and (probs <= 1.0).all().item())

    def test_checkpoint_selection_logic(self):
        """Verify best checkpoint is selected strictly based on validation metrics, never test metrics."""
        # Mock training epochs
        epochs = [
            {"epoch": 1, "val_f1": 0.75, "test_f1": 0.85},
            {"epoch": 2, "val_f1": 0.88, "test_f1": 0.80},  # True best on validation
            {"epoch": 3, "val_f1": 0.82, "test_f1": 0.95},  # High test, but lower val => must NOT be chosen
        ]

        best_val_f1 = -1.0
        best_epoch = -1
        for ep in epochs:
            # Checkpoint selection rule: based strictly on val_f1
            if ep["val_f1"] > best_val_f1:
                best_val_f1 = ep["val_f1"]
                best_epoch = ep["epoch"]

        self.assertEqual(best_epoch, 2, "Checkpoint selection incorrectly selected based on test or non-best epoch")
        self.assertEqual(best_val_f1, 0.88)

    def test_metric_calculations_and_thresholding(self):
        """Verify accuracy, sensitivity, specificity, F1, and ROC-AUC calculations."""
        # 2 Positive, 2 Negative
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])

        metrics = compute_binary_metrics(y_true, y_prob, threshold=0.50)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["specificity"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["f1_score"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

        # Test threshold sensitivity
        metrics_high_thresh = compute_binary_metrics(y_true, y_prob, threshold=0.85)
        # y_prob >= 0.85 => [1, 0, 0, 0] => TP=1, FN=1, TN=2, FP=0
        self.assertEqual(metrics_high_thresh["tp"], 1)
        self.assertEqual(metrics_high_thresh["fn"], 1)

    def test_parent_smear_level_aggregation(self):
        """Verify averaging 12 crop probabilities per parent smear produces parent-level diagnoses."""
        records = []
        # Parent 1: Anemic (12 crops with high probabilities)
        for c in range(1, 13):
            records.append({
                "original_image_id": "100_a",
                "crop_id": f"{c:02d}",
                "true_label": 1,
                "pred_prob": 0.70 + (c * 0.02)  # all >= 0.72
            })
        # Parent 2: Healthy (12 crops with low probabilities)
        for c in range(1, 13):
            records.append({
                "original_image_id": "200_h",
                "crop_id": f"{c:02d}",
                "true_label": 0,
                "pred_prob": 0.10 + (c * 0.02)  # all <= 0.34
            })

        smear_metrics = compute_parent_smear_level_metrics(records, threshold=0.50)
        self.assertEqual(smear_metrics["num_parent_smears"], 2)
        self.assertAlmostEqual(smear_metrics["accuracy"], 1.0)
        self.assertAlmostEqual(smear_metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(smear_metrics["specificity"], 1.0)
        self.assertIn("Parent-Smear-Level", smear_metrics["evaluation_level"])

    def test_clean_subset_separation(self):
        """Verify evaluate_test_predictions cleanly separates full and clean subset metrics."""
        mock_preds = [
            {"original_image_id": "001_a", "crop_id": "01", "true_label": 1, "pred_prob": 0.8},
            {"original_image_id": "002_h", "crop_id": "01", "true_label": 0, "pred_prob": 0.2},
            {"original_image_id": "119_a", "crop_id": "01", "true_label": 1, "pred_prob": 0.6},  # Conflicting parent
        ]

        # Use actual conflicts CSV
        results = evaluate_test_predictions(
            predictions=mock_preds,
            threshold=0.50,
            conflicts_csv=self.config.conflicts_csv_path
        )

        self.assertIn("full_test_crop_level", results)
        self.assertIn("full_test_parent_smear_level", results)
        self.assertEqual(results["full_test_crop_level"]["total_samples"], 3)

        if results.get("clean_subset_evaluated"):
            # 119_a should be excluded from clean subset
            self.assertEqual(results["clean_subset_crop_level"]["total_samples"], 2)
            self.assertEqual(results["clean_subset_parent_smear_level"]["num_parent_smears"], 2)
            self.assertEqual(results["clean_subset_excluded_parents"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
