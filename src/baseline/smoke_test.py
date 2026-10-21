"""
AneRBC Baseline Pipeline Smoke Test
-----------------------------------
Runs a short, self-contained one-fold smoke test:
  1. Partitions data into nested groups (Outer Test Fold 0, Inner Val Fold 1, Inner Train).
  2. Subsets to a small parent-group cohort (5 train parents = 60 crops, 2 val parents = 24 crops, 2 test parents = 24 crops).
  3. Trains for 1 epoch on CPU/GPU.
  4. Verifies that the training loss is finite.
  5. Saves checkpoint to disk.
  6. Reloads checkpoint into a fresh model instance.
  7. Evaluates on held-out test data and computes both crop-level and parent-smear-level metrics.
  8. Labels output strictly as a pipeline smoke test (NOT a scientific result).

Usage:
    python src/baseline/smoke_test.py
"""

import sys
import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
from evaluate import (
    compute_binary_metrics,
    compute_parent_smear_level_metrics,
    evaluate_test_predictions,
    format_evaluation_summary
)


def run_smoke_test():
    print("=" * 80)
    print("ANERBC BASELINE PIPELINE SMOKE TEST")
    print("WARNING: THIS IS A FUNCTIONAL PIPELINE TEST ONLY — NOT A SCIENTIFIC RESULT")
    print("=" * 80)

    config = BaselineConfig()
    smoke_dir = config.project_root / "experiments" / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = smoke_dir

    device = torch.device("cpu")  # CPU is sufficient and fast for small smoke test
    print(f"Device       : {device}")
    print(f"Artifact Dir : {smoke_dir}\n")

    # Step 1: Load nested records
    train_recs, val_recs, test_recs = load_nested_dataset_records(
        splits_csv=config.splits_csv_path,
        tier="AneRBC-II",
        test_fold=0,
        inner_val_fold=1
    )

    # Subset to distinct parent groups (all 12 crops per parent kept together)
    train_pids = sorted(list(set(r["original_image_id"] for r in train_recs)))[:5]  # 5 parents = 60 crops
    val_pids = sorted(list(set(r["original_image_id"] for r in val_recs)))[:2]      # 2 parents = 24 crops
    test_pids = sorted(list(set(r["original_image_id"] for r in test_recs)))[:2]    # 2 parents = 24 crops

    sub_train = [r for r in train_recs if r["original_image_id"] in train_pids]
    sub_val = [r for r in val_recs if r["original_image_id"] in val_pids]
    sub_test = [r for r in test_recs if r["original_image_id"] in test_pids]

    print(f"[1] Cohort Selection: Train={len(sub_train)} crops ({len(train_pids)} parents), "
          f"Val={len(sub_val)} crops ({len(val_pids)} parents), "
          f"Test={len(sub_test)} crops ({len(test_pids)} parents)")

    # Step 2: Validate zero leakage on smoke subset
    diag = validate_nested_splits(sub_train, sub_val, sub_test, config.hash_audit_csv_path)
    assert diag["parent_leakage"] == 0, "Parent leakage in smoke test!"
    assert diag["hash_leakage"] == 0, "Hash leakage in smoke test!"
    print("[2] Leakage Verification: PASSED (Zero parent leakage, Zero hash leakage).")

    # Step 3: Create Datasets & Loaders
    train_ds = AneRBCDataset(sub_train, zip_path=config.dataset_zip_path, transform=get_train_transforms(config.image_size))
    val_ds = AneRBCDataset(sub_val, zip_path=config.dataset_zip_path, transform=get_eval_transforms(config.image_size))
    test_ds = AneRBCDataset(sub_test, zip_path=config.dataset_zip_path, transform=get_eval_transforms(config.image_size))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Step 4: Model & 1-Epoch Training
    model = AneRBCClassifier("mobilenet_v2", pretrained=False, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n[3] Executing 1 Training Epoch...")
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        imgs, lbls = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"  • Training Loss: {avg_loss:.4f}")
    assert not torch.isnan(torch.tensor(avg_loss)), "Loss is NaN!"
    assert not torch.isinf(torch.tensor(avg_loss)), "Loss is Inf!"
    assert avg_loss > 0.0, "Loss must be positive!"
    print("  -> Loss is FINITE and VALID: PASSED.")

    # Step 5: Save Checkpoint
    checkpoint_file = smoke_dir / "smoke_checkpoint.pt"
    torch.save({
        "epoch": 1,
        "loss": avg_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_file)
    assert checkpoint_file.exists(), "Checkpoint file was not created!"
    print(f"\n[4] Saved Checkpoint: {checkpoint_file.name} ({checkpoint_file.stat().st_size:,} bytes): PASSED.")

    # Step 6: Reload into Fresh Model Instance
    fresh_model = AneRBCClassifier("mobilenet_v2", pretrained=False, num_classes=1).to(device)
    loaded_ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
    fresh_model.load_state_dict(loaded_ckpt["model_state_dict"])
    fresh_model.eval()
    print("[5] Reloaded Checkpoint into Fresh Model: PASSED.")

    # Step 7: Evaluate on Held-Out Test Data
    print("\n[6] Evaluating Reloaded Model on Held-Out Test Subset (24 crops)...")
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            logits = fresh_model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(probs)):
                test_preds.append({
                    "original_image_id": batch["original_image_id"][i],
                    "crop_id": batch["crop_id"][i],
                    "true_label": int(lbls[i].item()),
                    "pred_prob": float(probs[i]),
                    "pred_label": int(probs[i] >= 0.50)
                })

    assert len(test_preds) == len(sub_test), f"Expected {len(sub_test)} predictions, got {len(test_preds)}"
    crop_metrics = compute_binary_metrics([p["true_label"] for p in test_preds], [p["pred_prob"] for p in test_preds])
    smear_metrics = compute_parent_smear_level_metrics(test_preds)

    print(f"  • Crop-Level Accuracy        : {crop_metrics['accuracy']:.4f}")
    print(f"  • Crop-Level Sensitivity     : {crop_metrics['sensitivity']:.4f}")
    print(f"  • Crop-Level Specificity     : {crop_metrics['specificity']:.4f}")
    print(f"  • Parent-Smear Accuracy      : {smear_metrics['accuracy']:.4f}")
    print(f"  • Parent Smears Evaluated    : {smear_metrics['num_parent_smears']}")

    # Clean up
    train_ds.close()
    val_ds.close()
    test_ds.close()

    print("\n" + "=" * 80)
    print("PIPELINE SMOKE TEST COMPLETED SUCCESSFULLY (100% FUNCTIONAL)")
    print("NOTE: METRICS ABOVE ARE FROM 1 SMOKE EPOCH AND ARE NOT A SCIENTIFIC RESULT")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
