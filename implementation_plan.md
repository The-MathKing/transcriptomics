# Benchmarking Uncertainty Calibration in Spatial Transcriptomics Models

This project will systematically evaluate whether spatial transcriptomics deconvolution models produce well-calibrated uncertainty estimates, particularly under distribution shifts (cross-platform, cross-tissue, cross-species), and investigate lightweight post-hoc recalibration methods.

## User Review Required

Please review the proposed order of execution. The project is split into environment setup/model validation, data acquisition/preprocessing, and evaluation. We can tackle these in whatever order you prefer, but starting with a single model end-to-end is recommended to establish the pipeline.

## Open Questions

> [!IMPORTANT]  
> **Where should we start today?**
> 1. **Option A (Recommended):** Set up the Python environment, clone a probabilistic model (e.g., DestVI or Starfysh), and run it end-to-end on its provided demo dataset to validate the tooling.
> 2. **Option B:** Focus on data acquisition. Identify, download, and preprocess the specific paired datasets for our distribution shifts (e.g., 10x Visium human brain vs. Slide-seqV2).

> [!NOTE]
> **Model Selection for Initial Phase:** If we start with Option A, do you have a preference between DestVI and Starfysh for the initial proof-of-concept? DestVI is part of scvi-tools (very stable ecosystem), making it a strong first candidate.

## Proposed Changes / Phases

### Phase 1: Infrastructure and Validation
- Initialize project directory structure (`data/`, `models/`, `notebooks/`, `src/`).
- Create a `requirements.txt` / `environment.yml` with `scanpy`, `squidpy`, `scvi-tools` (for DestVI), `torch`, and calibration libraries (`netcal`).
- Clone/install the first model (e.g., DestVI).
- Run the chosen model on a tiny demo dataset to verify inference and uncertainty extraction (posterior variance).

### Phase 2: Data Acquisition & Preprocessing
- **In-distribution:** Download a standard reference (e.g., 10x Visium DLPFC).
- **Shifted:** Download matched tissue from a different platform (e.g., Slide-seqV2).
- Preprocess using `scanpy`: normalize, filter, and harmonize gene panels.
- Generate pseudo-spots from scRNA-seq references to establish ground-truth cell type proportions.

### Phase 3: Benchmarking Calibration
- Train/run inference on the in-distribution dataset.
- Extract uncertainty and compute ECE (Expected Calibration Error).
- Generate Reliability Diagrams.
- Run inference on the shifted dataset (zero-shot / no retraining) and measure calibration degradation.

### Phase 4: Recalibration
- Implement Split Conformal Prediction or Temperature Scaling on spatial proportions.
- Apply to a small calibration split of the shifted data.
- Measure improvement in Prediction Interval Coverage and ECE.

## Verification Plan

### Automated Tests
- Unit tests for the calibration metrics (ECE, Coverage) against known dummy data to ensure our metric implementations are correct before applying them to model outputs.

### Manual Verification
- Visual inspection of Spatial plots (e.g., mapping prediction confidence directly onto spatial coordinates to see if models are confidently wrong in specific tissue regions).
- Review of reliability diagrams to confirm visually whether confidence matches accuracy.
