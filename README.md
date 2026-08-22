# Spatial Calibration Benchmark

This repository contains the fully reproducible code for benchmarking confidence calibration of spatial transcriptomics deconvolution models under deployment-time distribution shift.

## Overview
We evaluate the calibration of DestVI under simulated cross-platform shift (via binomial dropout of mRNA capture efficiencies). Crucially, the model and post-hoc calibrators are fit **exclusively on clean data** and evaluated out-of-distribution without retraining.

## Contents
- `src/benchmark_v2.py`: The core benchmarking pipeline. It orchestrates data splitting, synthetic shift generation, model training (up to 200 epochs with ELBO early stopping), out-of-sample inference, and Expected Calibration Error (ECE) calculation.
- `src/calibrators.py`: Implementations of the post-hoc calibrators, including Temperature Scaling and Isotonic Regression.
- `src/data.py`: Utilities for downloading the Mouse Cortex reference via `squidpy` and synthesizing pseudo-spots.

## Reproducibility
The pipeline is designed for strict reproducibility:
- Random seeds are fixed.
- Hardware is constrained to CPU to ensure deterministic behavior in PyTorch and `scvi-tools`.

### Running the Benchmark
To reproduce the full 10-seed experiment (warning: this may take several hours):
```bash
python src/benchmark_v2.py
```

To run a rapid smoke-test (subsampled data, 2 seeds, 3 epochs) to verify your environment setup:
```bash
SMOKE_TEST=1 python src/benchmark_v2.py
```
This will output `results_smoketest.json` and generate an ECE degradation plot.
