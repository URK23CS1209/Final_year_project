# AneRBC-MedGemma-Research

**Final-year research project | Current stage: Baseline development and evaluation**

## Project overview

This project investigates computer-aided anemia classification using peripheral blood smear images from the AneRBC dataset. The plan is to establish a reliable image-classification baseline first, then explore explainability and the feasibility of adding MedGemma for clinical-context support.

This is a research prototype, not a clinically validated diagnostic system.

## Current progress

### Completed / available

- Inspected the AneRBC dataset archive and metadata.
- Developed dataset quality-control scripts, including image-duplicate and dataset-link audits.
- Developed a group-aware dataset-splitting workflow intended to reduce leakage between related images and crops from the same parent smear.
- Trained baseline experiments and saved five fold-specific checkpoints for each of:
  - **MobileNetV2**
  - **Vision Transformer (ViT-B/16)**
- Stored experiment artifacts in Google Drive under `AneRBC-Research-Results/`.

### Preliminary results

The current working summary reports the following preliminary metrics:

| Model | Crop-level accuracy | Smear-level accuracy | Smear-level AUC |
|---|---:|---:|---:|
| MobileNetV2 | 81.67% | 82.50% | 0.9185 |
| ViT-B/16 | 81.12% | 82.50% | 0.9189 |

**Important:** These numbers are preliminary and must be checked against the saved evaluation outputs and exact evaluation protocol before being treated as final results. They do not establish clinical readiness.

## Current stage

The project is currently at the **baseline verification and demonstration stage**.

Immediate work:
1. Load a saved checkpoint with its matching model definition and preprocessing.
2. Run and document a reproducible prediction on a dataset image.
3. Verify reported metrics and confirm the evaluation split and grouping protocol.
4. Compare the baseline models and document limitations.

## Planned work

- Investigate Explainable AI (XAI) methods and assess whether explanations are appropriate and reliable.
- Explore MedGemma integration only if suitable compute and memory resources are available. Current GPU/resource limitations may prevent running the model locally or in the available Colab environment; MedGemma is **not yet presented as an integrated, validated component**.
- Evaluate any proposed extensions with a clear protocol and report results without targeting a score at the expense of validity.

## Dataset and evaluation notes

AneRBC-I contains 1,000 original smear images (500 labelled anemic and 500 labelled healthy). The dataset also includes cropped images and related reports. Related crops must be kept grouped by their parent smear during splitting to avoid leakage between training and test data.

## Repository guide

- `src/inspect_archive.py`, `src/inspect_metadata.py`: dataset and metadata inspection.
- `src/audit_dataset_links.py`, `src/audit_image_hashes.py`: dataset relationship and duplicate audits.
- `src/create_dataset_splits.py`: dataset split generation.
- `src/baseline/`: baseline dataset, model, training, evaluation, validation, and smoke-test code.
- `README_BASELINE.md`: baseline pipeline details.
- Trained checkpoints and experiment outputs are stored separately in Google Drive and are not expected to be committed to GitHub.

## Limitations

- The reported metrics are preliminary until verified against the corresponding saved evaluation files.
- A successful single-image demo demonstrates inference functionality, not overall model performance.
- XAI and MedGemma integration are future work unless separately implemented and evaluated.
- The project is for research and educational use; it must not be used to make clinical decisions.

## Environment

The experiments use Python and PyTorch, with Google Colab/Google Drive used for training artifacts and experimentation. Refer to `requirements.txt` and the source configuration for the exact dependencies and settings.
