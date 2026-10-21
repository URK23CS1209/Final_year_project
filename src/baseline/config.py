"""
Baseline Configuration for AneRBC Binary Classification
-------------------------------------------------------
Defines model architectures, data paths, hyper-parameters, and evaluation settings.
Implements nested group-safe cross-validation where an outer fold is strictly
held out as the test set, an inner validation fold is used for model selection,
and the remaining folds are used for training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict


@dataclass
class BaselineConfig:
    # Project paths
    project_root: Path = Path(r"C:\Users\sharo\Desktop\FINAL YEAR PROJECT\AneRBC-MedGemma-Research")
    dataset_zip_path: Path = Path(
        r"C:\Users\sharo\Downloads\AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. httpsdoi.org10.1093databasebaae120.zip"
    )
    splits_csv_path: Path = Path(r"C:\Users\sharo\Desktop\FINAL YEAR PROJECT\AneRBC-MedGemma-Research\reports\dataset_splits.csv")
    hash_audit_csv_path: Path = Path(r"C:\Users\sharo\Desktop\FINAL YEAR PROJECT\AneRBC-MedGemma-Research\reports\image_hash_audit.csv")
    conflicts_csv_path: Path = Path(r"C:\Users\sharo\Desktop\FINAL YEAR PROJECT\AneRBC-MedGemma-Research\reports\conflicting_duplicate_investigation.csv")
    output_dir: Path = Path(r"C:\Users\sharo\Desktop\FINAL YEAR PROJECT\AneRBC-MedGemma-Research\experiments\baseline_mobilenetv2")

    # Dataset settings
    dataset_tier: str = "AneRBC-II"  # 'AneRBC-II' (12,000 crops) or 'AneRBC-I' (1,000 smears) or 'ALL'
    extracted_data_dir: Optional[Path] = None  # If unzipped on Colab/disk, specify directory; otherwise reads directly from zip
    image_size: int = 224  # Input resolution (224x224 for MobileNetV2/ResNet/VGG16)
    num_classes: int = 1   # Binary classification (raw linear logits for BCEWithLogitsLoss)

    # Model architecture
    model_name: str = "mobilenet_v2"  # 'mobilenet_v2', 'resnet18', 'resnet50', 'vgg16'
    pretrained: bool = True           # Use ImageNet pre-trained weights
    dropout_rate: float = 0.2
    freeze_backbone: bool = False     # False for full fine-tuning, True to freeze feature extractor

    # Training hyper-parameters
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 20                  # Baseline epochs for Colab GPU
    early_stopping_patience: int = 5  # Stop if validation macro F1 does not improve
    seed: int = 42

    # Nested Group-Safe Cross-Validation settings
    n_splits: int = 5
    active_test_folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # Inner validation fold offset: val_fold = (test_fold + inner_val_offset) % n_splits
    inner_val_offset: int = 1

    # Evaluation threshold
    classification_threshold: float = 0.50  # Documented decision boundary: p >= 0.50 => Anemic (1), else Healthy (0)

    # Class mapping (strictly consistent throughout training and evaluation)
    label_map: Dict[str, int] = field(default_factory=lambda: {"Healthy": 0, "Anemic": 1})
    inv_label_map: Dict[int, str] = field(default_factory=lambda: {0: "Healthy", 1: "Anemic"})

    # Smoke test mode
    smoke_test: bool = False
    smoke_test_samples: int = 120  # Total training crops in smoke test

    def __post_init__(self):
        self.project_root = Path(self.project_root)
        self.dataset_zip_path = Path(self.dataset_zip_path)
        self.splits_csv_path = Path(self.splits_csv_path)
        self.hash_audit_csv_path = Path(self.hash_audit_csv_path)
        self.conflicts_csv_path = Path(self.conflicts_csv_path)
        self.output_dir = Path(self.output_dir)
        if self.extracted_data_dir is not None:
            self.extracted_data_dir = Path(self.extracted_data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
