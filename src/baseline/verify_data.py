"""
Pre-Training Pipeline & Architecture Validator
----------------------------------------------
Verifies data integrity, nested fold partitioning, image loading from ZIP,
label mappings, tensor shapes, model forward pass, and metric computations.

Usage:
    python src/baseline/verify_data.py
"""

import sys
import random
from pathlib import Path
import torch

sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

from config import BaselineConfig
from dataset import (
    AneRBCDataset,
    get_train_transforms,
    get_eval_transforms,
    load_nested_dataset_records,
    validate_nested_splits
)
from model import AneRBCClassifier
from evaluate import compute_binary_metrics, compute_parent_smear_level_metrics


def verify_pipeline():
    print("=" * 80)
    print("ANERBC BASELINE PRE-TRAINING PIPELINE VALIDATION")
    print("=" * 80)

    config = BaselineConfig()
    print(f"Project Root : {config.project_root}")
    print(f"Splits CSV   : {config.splits_csv_path}")
    print(f"Dataset ZIP  : {config.dataset_zip_path}")
    print(f"Model Target : {config.model_name.upper()}\n")

    # Step 1: Check File Existence
    assert config.splits_csv_path.exists(), f"Splits CSV missing: {config.splits_csv_path}"
    assert config.dataset_zip_path.exists(), f"Dataset ZIP missing: {config.dataset_zip_path}"
    assert config.hash_audit_csv_path.exists(), f"Hash audit CSV missing: {config.hash_audit_csv_path}"
    print("[1] File Existence Checks: PASSED.")

    # Step 2: Test Nested 5-Fold Partitioning & Zero Leakage Across ALL 5 Folds
    print("\n[2] Testing Nested Group-Safe Partitioning Across All 5 Outer Test Folds:")
    for test_f in range(5):
        val_f = (test_f + config.inner_val_offset) % config.n_splits
        train_recs, val_recs, test_recs = load_nested_dataset_records(
            splits_csv=config.splits_csv_path,
            tier="AneRBC-II",
            test_fold=test_f,
            inner_val_fold=val_f
        )
        assert len(test_recs) == 2400, f"Test fold {test_f} expected 2,400 crops, got {len(test_recs)}"
        assert len(val_recs) == 2400, f"Val fold {val_f} expected 2,400 crops, got {len(val_recs)}"
        assert len(train_recs) == 7200, f"Train folds expected 7,200 crops, got {len(train_recs)}"

        diag = validate_nested_splits(
            train_records=train_recs,
            val_records=val_recs,
            test_records=test_recs,
            hash_audit_csv=config.hash_audit_csv_path
        )
        assert diag["parent_leakage"] == 0, f"Parent leakage in fold {test_f}"
        assert diag["hash_leakage"] == 0, f"Hash leakage in fold {test_f}"
        print(f"  • Outer Test Fold {test_f} (Val={val_f}, Train=remaining): "
              f"Train={len(train_recs):,} | Val={len(val_recs):,} | Test={len(test_recs):,} | Zero Leakage: PASSED")

    print("  -> All 5 Nested Partitions Validated: 100% SUCCESS.")

    # Step 3: Test Direct Image Reading from ZIP
    sample_records, _, _ = load_nested_dataset_records(
        splits_csv=config.splits_csv_path,
        tier="ALL",
        test_fold=0,
        inner_val_fold=1
    )
    random.seed(42)
    sample_sub = random.sample(sample_records, 10)

    dataset = AneRBCDataset(
        records=sample_sub,
        zip_path=config.dataset_zip_path,
        transform=get_train_transforms(config.image_size),
        label_map=config.label_map
    )

    print("\n[3] Testing Direct ZIP Image Decompression & Transformation:")
    for i in range(len(dataset)):
        item = dataset[i]
        tensor = item["image"]
        lbl = item["label"]
        pid = item["original_image_id"]
        tier = item["dataset_tier"]
        dir_lbl = item["directory_label"]

        assert tensor.shape == torch.Size([3, 224, 224]), f"Unexpected tensor shape: {tensor.shape}"
        assert lbl.item() in [0.0, 1.0], f"Invalid label: {lbl}"
        assert config.label_map[dir_lbl] == int(lbl.item()), f"Label mapping mismatch for {dir_lbl}"
        print(f"  • Sample {i+1:02d}: Tier={tier:<9} PID={pid:<7} Label={lbl.item():.0f} ({dir_lbl:<7}) Tensor Shape={tuple(tensor.shape)}")

    dataset.close()
    print("  -> Direct In-Memory ZIP Loading & Consistent Label Mapping: 100% SUCCESS.")

    # Step 4: Model Architecture & Loss Compatibility Check
    print("\n[4] Testing Model Architecture & BCEWithLogitsLoss Compatibility:")
    model = AneRBCClassifier(model_name=config.model_name, pretrained=False, num_classes=1)
    dummy_input = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    criterion = torch.nn.BCEWithLogitsLoss()
    logits = model(dummy_input)

    assert logits.shape == torch.Size([4]), f"Expected logits shape [4], got {logits.shape}"
    loss = criterion(logits, dummy_labels)
    assert not torch.isnan(loss) and not torch.isinf(loss), "BCEWithLogitsLoss returned non-finite value"
    probs = torch.sigmoid(logits)

    print(f"  • Dummy Input Shape  : {tuple(dummy_input.shape)}")
    print(f"  • Output Logits Shape: {tuple(logits.shape)} (Raw Linear Logits)")
    print(f"  • BCEWithLogitsLoss  : {loss.item():.4f} (Finite & Compatible)")
    print(f"  • Sigmoid Probs      : {[round(p, 4) for p in probs.tolist()]}")
    print("  -> Model & Loss Compatibility: 100% SUCCESS.")

    # Step 5: Test Metric Computations & Parent Aggregation
    print("\n[5] Testing Metric Calculations & Parent-Smear Aggregation:")
    # Create mock 24 crop predictions for 2 parent smears (12 crops each)
    mock_preds = []
    for c in range(1, 13):
        mock_preds.append({
            "original_image_id": "001_a",
            "crop_id": f"{c:02d}",
            "true_label": 1,
            "pred_prob": 0.85 if c > 3 else 0.40  # 9 of 12 are positive => mean ~ 0.7375
        })
    for c in range(1, 13):
        mock_preds.append({
            "original_image_id": "002_h",
            "crop_id": f"{c:02d}",
            "true_label": 0,
            "pred_prob": 0.20 if c > 2 else 0.60  # 10 of 12 are negative => mean ~ 0.2667
        })

    smear_metrics = compute_parent_smear_level_metrics(mock_preds, threshold=0.50)
    assert smear_metrics["accuracy"] == 1.0, "Parent aggregation failed to classify parents correctly"
    assert smear_metrics["num_parent_smears"] == 2, "Expected 2 parent smears"
    print(f"  • Parent Smears Evaluated : {smear_metrics['num_parent_smears']}")
    print(f"  • Parent Diagnostic Acc   : {smear_metrics['accuracy']:.4f}")
    print(f"  • Parent Sensitivity      : {smear_metrics['sensitivity']:.4f}")
    print(f"  • Parent Specificity      : {smear_metrics['specificity']:.4f}")
    print("  -> Parent-Smear Evaluation: 100% SUCCESS.")

    print("\n" + "=" * 80)
    print("ALL PRE-TRAINING VALIDATION CHECKS PASSED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    verify_pipeline()
