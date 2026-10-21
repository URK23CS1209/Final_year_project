"""
Training Pipeline for AneRBC Nested Group-Safe Cross-Validation
---------------------------------------------------------------
Enforces strict nested group-safe evaluation:
  • Outer Test Fold    : Strictly held out for final model evaluation. Never used for
                         early stopping, threshold selection, or hyperparameter tuning.
  • Inner Val Fold     : Used exclusively for monitoring, early stopping, and checkpoint selection.
  • Inner Train Folds  : Used exclusively for gradient optimization.
  • Zero Leakage Proof : Validates zero parent-ID and zero SHA-256 hash leakage before training.
"""

import sys
import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure UTF-8 output on Windows
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


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.50).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, List[Dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total = 0
    predictions = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        true_labels = labels.cpu().numpy()

        for i in range(len(probs)):
            predictions.append({
                "original_image_id": batch["original_image_id"][i],
                "crop_id": batch["crop_id"][i],
                "dataset_tier": batch["dataset_tier"][i],
                "image_path": batch["image_path"][i],
                "directory_label": batch["directory_label"][i],
                "fold": int(batch["fold"][i]),
                "true_label": int(true_labels[i]),
                "pred_prob": float(probs[i]),
                "pred_label": int(probs[i] >= 0.50)
            })

    val_loss = total_loss / total
    return val_loss, predictions


def run_nested_fold(
    test_fold: int,
    config: BaselineConfig,
    device: torch.device,
    inner_val_fold: Optional[int] = None,
    subset_limit: Optional[int] = None
) -> Dict[str, Any]:
    if inner_val_fold is None:
        inner_val_fold = (test_fold + config.inner_val_offset) % config.n_splits

    print("\n" + "=" * 80)
    print(f"NESTED GROUP-SAFE CROSS-VALIDATION: OUTER TEST FOLD {test_fold}")
    print("=" * 80)
    print(f"• Outer Test Fold (Held Out) : Fold {test_fold}")
    print(f"• Inner Validation Fold      : Fold {inner_val_fold} (used ONLY for early stopping & checkpoint selection)")
    remaining_folds = [f for f in range(config.n_splits) if f not in (test_fold, inner_val_fold)]
    print(f"• Inner Training Folds       : Folds {remaining_folds}")
    print(f"• Model Architecture         : {config.model_name.upper()}")
    print(f"• Target Device              : {device}")

    seed_everything(config.seed + test_fold)

    # 1. Load nested partitions
    train_recs, val_recs, test_recs = load_nested_dataset_records(
        splits_csv=config.splits_csv_path,
        tier=config.dataset_tier,
        test_fold=test_fold,
        inner_val_fold=inner_val_fold
    )

    if subset_limit:
        print(f"[INFO] Subsetting for smoke test: {subset_limit} train samples")
        train_recs = train_recs[:subset_limit]
        val_recs = val_recs[:min(len(val_recs), subset_limit // 2)]
        test_recs = test_recs[:min(len(test_recs), subset_limit // 2)]

    # 2. Formal pre-training zero-leakage verification
    diag = validate_nested_splits(
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        hash_audit_csv=config.hash_audit_csv_path
    )
    print(f"[LEAKAGE AUDIT] Zero parent leakage: {diag['parent_leakage'] == 0} | Zero hash leakage: {diag['hash_leakage'] == 0}")
    print(f"[PARTITION SIZES] Train: {len(train_recs):,} crops ({diag['train_parents']} parents) | "
          f"Val: {len(val_recs):,} crops ({diag['val_parents']} parents) | "
          f"Test: {len(test_recs):,} crops ({diag['test_parents']} parents)")

    # 3. Create Datasets & Loaders
    train_dataset = AneRBCDataset(
        records=train_recs,
        zip_path=config.dataset_zip_path,
        extracted_dir=config.extracted_data_dir,
        transform=get_train_transforms(config.image_size),
        label_map=config.label_map
    )
    val_dataset = AneRBCDataset(
        records=val_recs,
        zip_path=config.dataset_zip_path,
        extracted_dir=config.extracted_data_dir,
        transform=get_eval_transforms(config.image_size),
        label_map=config.label_map
    )
    test_dataset = AneRBCDataset(
        records=test_recs,
        zip_path=config.dataset_zip_path,
        extracted_dir=config.extracted_data_dir,
        transform=get_eval_transforms(config.image_size),
        label_map=config.label_map
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # 4. Model, Loss, Optimizer
    model = AneRBCClassifier(
        model_name=config.model_name,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
        num_classes=config.num_classes
    ).to(device)

    if config.freeze_backbone:
        model.freeze_backbone()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 5. Training Loop with Inner Validation Selection
    best_val_f1 = -1.0
    best_val_epoch = -1
    best_val_metrics = {}
    best_val_preds = []
    checkpoint_path = config.output_dir / f"best_model_testfold_{test_fold}.pt"
    history = []

    print("\n--- BEGIN INNER TRAINING & VALIDATION MONITORING ---")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        y_true_val = [p["true_label"] for p in val_preds]
        y_prob_val = [p["pred_prob"] for p in val_preds]
        val_metrics = compute_binary_metrics(y_true_val, y_prob_val, threshold=config.classification_threshold)
        curr_val_f1 = val_metrics["f1_score"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_metrics["accuracy"]),
            "val_f1": float(curr_val_f1),
            "val_roc_auc": float(val_metrics["roc_auc"])
        }
        history.append(epoch_record)

        print(f"Epoch {epoch:02d}/{config.epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {curr_val_f1:.4f} AUC: {val_metrics['roc_auc']:.4f}")

        # Checkpoint selection based strictly on INNER VALIDATION metric
        if curr_val_f1 > best_val_f1:
            best_val_f1 = curr_val_f1
            best_val_epoch = epoch
            best_val_metrics = val_metrics
            best_val_preds = val_preds
            patience_counter = 0

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "test_fold": test_fold,
                "inner_val_fold": inner_val_fold,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "seed": config.seed + test_fold,
                "config": {
                    "model_name": config.model_name,
                    "dataset_tier": config.dataset_tier,
                    "image_size": config.image_size,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "classification_threshold": config.classification_threshold
                }
            }, checkpoint_path)
            print(f"  -> [SAVED BEST CHECKPOINT] Epoch {epoch} (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch} (patience={config.early_stopping_patience}).")
                break

    # 6. Final Evaluation on Isolated Outer Test Fold
    print("\n--- LOADING BEST CHECKPOINT FOR HELD-OUT OUTER TEST EVALUATION ---")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[INFO] Loaded model checkpoint from Epoch {checkpoint['epoch']} (Best Val F1: {checkpoint['val_metrics']['f1_score']:.4f})")

    test_loss, test_preds = evaluate_epoch(model, test_loader, criterion, device)

    # Full and clean-subset evaluation
    test_eval_results = evaluate_test_predictions(
        predictions=test_preds,
        threshold=config.classification_threshold,
        conflicts_csv=config.conflicts_csv_path
    )
    test_eval_results["test_loss"] = float(test_loss)
    test_eval_results["best_val_epoch"] = best_val_epoch
    test_eval_results["best_val_metrics"] = best_val_metrics

    # Print clean markdown summary
    summary_md = format_evaluation_summary(test_eval_results, title=f"Outer Test Fold {test_fold} Final Evaluation")
    print("\n" + summary_md)

    # 7. Save outputs
    # Save test predictions CSV
    test_df = pd.DataFrame(test_preds)
    test_csv_path = config.output_dir / f"test_predictions_fold_{test_fold}.csv"
    test_df.to_csv(test_csv_path, index=False)

    # Save validation predictions CSV
    val_df = pd.DataFrame(best_val_preds)
    val_csv_path = config.output_dir / f"val_predictions_fold_{test_fold}.csv"
    val_df.to_csv(val_csv_path, index=False)

    # Save training history & metrics JSON
    run_log = {
        "test_fold": test_fold,
        "inner_val_fold": inner_val_fold,
        "training_folds": remaining_folds,
        "best_epoch": best_val_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_eval_results,
        "training_history": history,
        "seed": config.seed + test_fold,
        "leakage_audit": diag
    }
    log_json_path = config.output_dir / f"run_log_fold_{test_fold}.json"
    with open(log_json_path, 'w', encoding='utf-8') as f:
        json.dump(run_log, f, indent=2)

    # Clean up dataset handles
    train_dataset.close()
    val_dataset.close()
    test_dataset.close()

    print(f"[INFO] Saved test predictions to : {test_csv_path.name}")
    print(f"[INFO] Saved run log to          : {log_json_path.name}")

    return run_log


def main():
    parser = argparse.ArgumentParser(description="AneRBC Nested Group-Safe Baseline Classifier")
    parser.add_argument("--test-fold", type=int, default=0, help="Outer test fold index (0-4)")
    parser.add_argument("--inner-val-fold", type=int, default=None, help="Inner validation fold (default: (test_fold + 1) % 5)")
    parser.add_argument("--run-all-folds", action="store_true", help="Execute full 5-fold cross-validation")
    parser.add_argument("--model", type=str, default="mobilenet_v2", help="Model architecture")
    parser.add_argument("--tier", type=str, default="AneRBC-II", help="AneRBC-I or AneRBC-II")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    config = BaselineConfig(
        model_name=args.model,
        dataset_tier=args.tier,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")
    print(f"[INFO] Output Directory: {config.output_dir}")

    if args.run_all_folds:
        folds_to_run = list(range(config.n_splits))
    else:
        folds_to_run = [args.test_fold]

    results = []
    for f in folds_to_run:
        res = run_nested_fold(
            test_fold=f,
            config=config,
            device=device,
            inner_val_fold=args.inner_val_fold
        )
        results.append(res)

    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETE - SUMMARY REPORT")
    print("=" * 80)
    for res in results:
        f = res["test_fold"]
        tc = res["test_metrics"]["full_test_crop_level"]
        ts = res["test_metrics"]["full_test_parent_smear_level"]
        print(f"Test Fold {f}: Crop Acc={tc['accuracy']:.4f}, Crop F1={tc['f1_score']:.4f} | "
              f"Parent-Smear Acc={ts['accuracy']:.4f}, Parent-Smear F1={ts['f1_score']:.4f}")


if __name__ == "__main__":
    main()
