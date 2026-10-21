# AneRBC Baseline Binary Anemia Classification Pipeline

Reproducible, leakage-safe baseline experiment for binary anemia classification on the **AneRBC** peripheral blood smear image dataset.

---

## 1. Overview & Baseline Purpose

This pipeline establishes the quantitative baseline for our final-year research project:
1. **Benchmark Context**: Implements standard Convolutional Neural Networks (**MobileNetV2**, **ResNet18/50**, **VGG16**) on AneRBC-II cropped RBC images ($306 \times 320$ pixels) and AneRBC-I whole smears ($1224 \times 960$ pixels).
2. **Leakage-Safe Protocol**: Replaces the authors' arbitrary non-grouped split with our mathematically verified **5-fold stratified group cross-validation** (`reports/dataset_splits.csv`), enforcing:
   - **Zero cross-fold parent-ID leakage** (all 12 crops strictly co-located with their parent smear).
   - **Zero cross-fold SHA-256 hash leakage** (identical image files kept in the same fold).
   - **Exact 50.00% class balance** across all 5 folds.
3. **Role in Research**: This CNN establishes the pure 2D computer vision performance boundary before advancing to our final contribution: multimodal clinical grounding with **MedGemma 1.5 4B IT**.

---

## 2. Directory Structure

```
src/baseline/
├── config.py                  # Centralized hyperparameters & paths
├── dataset.py                 # PyTorch Dataset (direct ZIP reading or extracted dir) & nested CV split loaders
├── model.py                   # CNN architectures (MobileNetV2, ResNet, VGG16) with raw linear logits
├── evaluate.py                # Metrics, patient-level aggregation, clean-subset ablation
├── train.py                   # Nested group-safe CV training loop & checkpointing
├── verify_data.py             # Pre-training pipeline & data validator
├── test_baseline_pipeline.py  # Automated unit test suite (7 tests)
└── smoke_test.py              # 1-epoch lightweight functional pipeline smoke test
```

---

## 3. Nested Group-Safe Cross-Validation Protocol

To eliminate data leakage and ensure rigorous generalization estimates:
1. **Outer Test Set**: In fold $k \in \{0, 1, 2, 3, 4\}$, fold $k$ is reserved strictly for test evaluation. It is never used for early stopping, model checkpointing, or threshold tuning.
2. **Inner Validation Set**: Fold $(k+1) \bmod 5$ is allocated as the validation set, used exclusively for monitoring validation loss and saving the checkpoint with highest validation macro-F1.
3. **Inner Training Set**: The remaining 3 folds form the training set.
4. **Group Separation**:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
   This partition holds simultaneously across parent smear IDs and raw SHA-256 hashes, with exact 50.00% binary class balance in all three partitions.

---

## 4. Modeling, Loss & Aggregation Specifications

- **Model Output**: Raw linear logits $\mathbb{R}^B$ with shape `(B,)`.
- **Loss Function**: `torch.nn.BCEWithLogitsLoss()`, with binary float targets $y \in \{0.0, 1.0\}$ (`Healthy: 0`, `Anemic: 1`).
- **Inference Probabilities**: $\hat{p} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$.
- **Decision Threshold**: Documented default $\tau = 0.50$ ($\hat{y} = 1 \iff \hat{p} \ge 0.50$).
- **Parent-Smear (Patient-Level) Diagnosis**: Aggregated across all 12 crops belonging to each parent smear via mean predicted probability:
  $$\hat{p}_{\text{smear}} = \frac{1}{12} \sum_{i=1}^{12} \hat{p}_{\text{crop}_i}, \quad \hat{y}_{\text{smear}} = \mathbb{I}(\hat{p}_{\text{smear}} \ge 0.50)$$
- **Segregated Evaluations**:
  1. **Full Outer Test Set**: Comprehensive evaluation on all 2,400 crops (200 parent smears) in the test fold.
  2. **Clean-Subset Sensitivity Analysis**: Ablation evaluating only unambiguous smears (excluding the 71 parent smears involved in conflicting-label duplicate groups).

---

## 5. Environment & Dependencies

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## 6. Verification & Automated Unit Tests

### A. Run Automated Unit Test Suite
Verify inner split leakage isolation, test-fold isolation, checkpoint reload logic, prediction-label alignment, metric computations, and clean subset filtering:
```bash
python -m unittest src/baseline/test_baseline_pipeline.py
```

### B. Run Pre-Training Pipeline Validator
Validates image loading directly from the ZIP, label mappings, and model forward pass across all 5 nested folds:
```bash
python src/baseline/verify_data.py
```

### C. Run Standalone Pipeline Smoke Test
Runs a 1-epoch lightweight run on a small parent cohort (5 train parents = 60 crops, 2 val parents = 24 crops, 2 test parents = 24 crops) to verify finite loss, checkpoint save/reload, and held-out test evaluation:
```bash
python src/baseline/smoke_test.py
```
> [!NOTE]
> The smoke test is strictly a functional pipeline verification tool and does not represent a scientific result.

---

## 7. Execution Instructions (Full Training)

> [!WARNING]
> Full training runs should only be initiated when computational resources (e.g. GPU) are confirmed and ready.

### A. Run Single Fold (e.g. Fold 0)
```bash
python src/baseline/train.py --fold 0 --model mobilenet_v2 --epochs 20 --batch-size 32 --lr 1e-4
```

### B. Run Complete 5-Fold Cross-Validation
```bash
python src/baseline/train.py --run-all-folds --model mobilenet_v2 --epochs 20 --batch-size 32 --lr 1e-4
```

### C. Running on Google Colab (Free T4 GPU)
1. Upload `dataset_splits.csv` and `conflicting_duplicate_investigation.csv` to Colab.
2. Mount Google Drive or upload the AneRBC dataset ZIP.
3. In a Colab cell:
```python
!pip install -r requirements.txt
!python src/baseline/train.py --run-all-folds --model mobilenet_v2 --epochs 20 --batch-size 64
```

---

## 8. Generated Outputs & Checkpoints

All training outputs are saved under `experiments/baseline_<model_name>/`:
- `best_model_fold_<k>.pt`: PyTorch model checkpoint selected strictly by inner validation macro-F1.
- `predictions_fold_<k>.csv`: Per-crop predictions, ground truth, and predicted probabilities aligned by `record_id`.
- `fold_<k>_results.json`: Run log recording random seeds, hyperparameters, epoch losses, validation metrics, and test evaluation metrics.
- Metrics reported:
  - **Crop-level**: Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, Balanced Accuracy, ROC-AUC, Confusion Matrix.
  - **Parent-level**: Predictions aggregated across the 12 crops of each parent smear via mean probability.
  - **Clean-subset ablation**: Evaluated separately on unambiguous parent smears excluding the 71 conflicting duplicate parents.

