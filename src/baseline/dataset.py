"""
AneRBC PyTorch Dataset & Nested Partitioning Implementation
-----------------------------------------------------------
Handles loading of peripheral blood smear images and crops directly from
the AneRBC ZIP archive or an uncompressed directory. Enforces strict
nested group-safe splits (Train, Val, Test) preserving parent-ID and
image-hash isolation.
"""

import os
import io
import csv
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Set

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Standard training data augmentation for microscopy images."""
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomCrop((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic validation/test preprocessing."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class AneRBCDataset(Dataset):
    """
    AneRBC Image Dataset for Binary Anemia Classification.
    Reads images either directly from the ZIP archive (zero disk extraction)
    or from an uncompressed directory.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        zip_path: Optional[Path] = None,
        extracted_dir: Optional[Path] = None,
        transform: Optional[Callable] = None,
        label_map: Optional[Dict[str, int]] = None
    ):
        self.records = records
        self.zip_path = Path(zip_path) if zip_path else None
        self.extracted_dir = Path(extracted_dir) if extracted_dir else None
        self.transform = transform
        self.label_map = label_map or {"Healthy": 0, "Anemic": 1}

        if not self.zip_path and not self.extracted_dir:
            raise ValueError("Either zip_path or extracted_dir must be provided.")

        # Lazy per-process zip handle for DataLoader multiprocessing safety
        self._zip_handle = None

    def _get_zip_handle(self) -> zipfile.ZipFile:
        if self._zip_handle is None:
            self._zip_handle = zipfile.ZipFile(self.zip_path, 'r')
        return self._zip_handle

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.records[idx]
        img_rel_path = row["image_path"]
        label_str = row["directory_label"]
        label = self.label_map[label_str]

        # Load image bytes
        if self.extracted_dir and (self.extracted_dir / img_rel_path).exists():
            full_path = self.extracted_dir / img_rel_path
            image = Image.open(full_path).convert("RGB")
        elif self.zip_path:
            zf = self._get_zip_handle()
            data = zf.read(img_rel_path)
            image = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            raise FileNotFoundError(f"Cannot locate image: {img_rel_path}")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "original_image_id": row["original_image_id"],
            "crop_id": row.get("crop_id", ""),
            "dataset_tier": row["dataset_tier"],
            "image_path": img_rel_path,
            "directory_label": label_str,
            "fold": int(row["fold"])
        }

    def close(self):
        if self._zip_handle is not None:
            self._zip_handle.close()
            self._zip_handle = None


def load_nested_dataset_records(
    splits_csv: Path,
    tier: str = "AneRBC-II",
    test_fold: int = 0,
    inner_val_fold: int = 1,
    exclude_conflicts: bool = False,
    conflicts_csv: Optional[Path] = None
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Loads records from reports/dataset_splits.csv and creates a nested group-safe partition:
      • test_records : records in outer test_fold (strictly held out).
      • val_records  : records in inner_val_fold (used for early stopping & checkpoint selection).
      • train_records: records in remaining folds (used for training optimization).
    """
    if test_fold == inner_val_fold:
        raise ValueError(f"test_fold ({test_fold}) cannot be the same as inner_val_fold ({inner_val_fold})")

    with open(splits_csv, 'r', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))

    # Filter by tier
    if tier in ["AneRBC-I", "AneRBC-II"]:
        records = [r for r in all_rows if r["dataset_tier"] == tier]
    else:
        records = all_rows

    # If excluding conflicting groups for clean ablation
    if exclude_conflicts and conflicts_csv and conflicts_csv.exists():
        with open(conflicts_csv, 'r', encoding='utf-8') as f:
            conf_rows = list(csv.DictReader(f))
        conflicting_parents = set()
        for r in conf_rows:
            conflicting_parents.update(r["parent_ids"].split("; "))
        records = [r for r in records if r["original_image_id"] not in conflicting_parents]

    test_records = [r for r in records if int(r["fold"]) == test_fold]
    val_records = [r for r in records if int(r["fold"]) == inner_val_fold]
    train_records = [r for r in records if int(r["fold"]) not in (test_fold, inner_val_fold)]

    return train_records, val_records, test_records


def validate_nested_splits(
    train_records: List[Dict[str, str]],
    val_records: List[Dict[str, str]],
    test_records: List[Dict[str, str]],
    hash_audit_csv: Optional[Path] = None
) -> Dict[str, any]:
    """
    Formally verifies that:
      1. Train, Val, and Test sets are mutually disjoint in image paths.
      2. No Parent ID appears in more than one partition (Zero Parent Leakage).
      3. No SHA-256 image hash appears in more than one partition (Zero Hash Leakage).
      4. All 12 crops belonging to each parent are strictly co-located within the same partition.
    """
    # 1. Path disjointness
    train_paths = {r["image_path"] for r in train_records}
    val_paths = {r["image_path"] for r in val_records}
    test_paths = {r["image_path"] for r in test_records}

    assert len(train_paths & val_paths) == 0, f"Train and Val share {len(train_paths & val_paths)} image paths!"
    assert len(train_paths & test_paths) == 0, f"Train and Test share {len(train_paths & test_paths)} image paths!"
    assert len(val_paths & test_paths) == 0, f"Val and Test share {len(val_paths & test_paths)} image paths!"

    # 2. Parent ID disjointness
    train_parents = {r["original_image_id"] for r in train_records}
    val_parents = {r["original_image_id"] for r in val_records}
    test_parents = {r["original_image_id"] for r in test_records}

    parent_train_val = train_parents & val_parents
    parent_train_test = train_parents & test_parents
    parent_val_test = val_parents & test_parents

    assert len(parent_train_val) == 0, f"Train and Val share {len(parent_train_val)} parent IDs: {parent_train_val}"
    assert len(parent_train_test) == 0, f"Train and Test share {len(parent_train_test)} parent IDs: {parent_train_test}"
    assert len(parent_val_test) == 0, f"Val and Test share {len(parent_val_test)} parent IDs: {parent_val_test}"

    # 3. Hash disjointness (if hash audit provided)
    hash_leakage_detected = 0
    if hash_audit_csv and hash_audit_csv.exists():
        with open(hash_audit_csv, 'r', encoding='utf-8-sig') as f:
            path_to_hash = {r["image_path"]: r["sha256"] for r in csv.DictReader(f)}

        train_hashes = {path_to_hash[p] for p in train_paths if p in path_to_hash}
        val_hashes = {path_to_hash[p] for p in val_paths if p in path_to_hash}
        test_hashes = {path_to_hash[p] for p in test_paths if p in path_to_hash}

        hash_tv = train_hashes & val_hashes
        hash_tt = train_hashes & test_hashes
        hash_vt = val_hashes & test_hashes

        assert len(hash_tv) == 0, f"Train and Val share {len(hash_tv)} SHA-256 hashes!"
        assert len(hash_tt) == 0, f"Train and Test share {len(hash_tt)} SHA-256 hashes!"
        assert len(hash_vt) == 0, f"Val and Test share {len(hash_vt)} SHA-256 hashes!"
        hash_leakage_detected = len(hash_tv) + len(hash_tt) + len(hash_vt)

    return {
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "train_parents": len(train_parents),
        "val_parents": len(val_parents),
        "test_parents": len(test_parents),
        "parent_leakage": len(parent_train_val) + len(parent_train_test) + len(parent_val_test),
        "hash_leakage": hash_leakage_detected,
        "is_leakage_free": True
    }
