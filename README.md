# Post-Hoc Reliability Calibration for Spatial Transcriptomics Deconvolution

This repository provides the fully reproducible benchmark codebase for evaluating post-hoc confidence calibration algorithms (Temperature Scaling, Isotonic Regression, Beta Calibration) applied to spatial transcriptomics deconvolution models across architectures, taxonomic granularities, and technical distribution shifts.

## Repository Overview

- **Multi-Model Deconvolution**: Supports amortized variational inference (`DestVI` / `scvi-tools`) and per-spot Poisson likelihood maximization (`cell2location`).
- **Multi-Granularity Auditing**: Evaluates well-posed coarse taxonomic regimes ($k=4$) alongside fine-grained subclass regimes ($k=23$) subject to compositional ties and base-rate collapse.
- **Technical Shift Robustness**: Benchmarks stability under capture efficiency dropout (binomial thinning) and ambient background RNA contamination (Poisson noise).
- **Simplex Probability Redistribution**: Implements proportional multi-class probability redistribution with simplex stability clamps for proper scoring (continuous simplex Brier score and NLL).
- **Synthetic Density Mechanism**: 12-condition battery evaluating parametric vs. non-parametric calibrator degradation under marginal score distribution shift $P(S)$ with invariant conditional accuracy $P(Y=1 \mid S=s)$.

---

## Directory Structure

```
spatial_calibration_benchmark/
├── src/
│   ├── benchmark_v2.py                 # Core DestVI fine-grained (k=23) benchmark pipeline
│   ├── benchmark_c2l.py                # cell2location fine-grained (k=23) benchmark
│   ├── benchmark_c2l_coarse.py         # cell2location coarse-grained (k=4) benchmark
│   ├── diagnostic_k4_confusion.py      # Hungarian label alignment & confusion diagnostics
│   ├── run_c2l_alpha_sensitivity.py    # cell2location prior dispersion (alpha) sweep
│   ├── run_density_sweep_comprehensive.py # Synthetic marginal density shift battery
│   ├── compute_stats_and_multiplicity.py  # Bootstrap CIs, paired tests & Holm-Bonferroni correction
│   ├── simulate_noise_floor.py         # Dirichlet finite-sample noise floor simulation
│   ├── plot_knots.py                   # Generates Figure 2 (density & active knot mappings)
│   ├── plot_multimodel.py              # Generates Figure 3 (DestVI vs. cell2location comparisons)
│   └── plot_results.py                 # Generates Figure 1 & Figure 4 (ECE & shift degradation)
├── figures/                            # High-resolution benchmark figures
├── results/                            # Raw benchmark metric logs (JSON)
├── requirements.txt                    # Python environment specifications
├── pyproject.toml                      # Package installation configuration
└── README.md
```

---

## Installation & Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Benchmark Results

### 1. Rapid Smoke Test
To verify the full evaluation pipeline in ~2 minutes with subsampled data and 3 epochs:
```bash
SMOKE_TEST=1 python3 src/benchmark_v2.py
```

### 2. Coarse Granularity ($k=4$) Benchmark
Run Hungarian optimal label alignment and evaluate DestVI and `cell2location` across 5 holdout splits:
```bash
python3 src/diagnostic_k4_confusion.py
python3 src/benchmark_c2l_coarse.py
```

### 3. Fine Granularity ($k=23$) Negative Control Benchmark
```bash
python3 src/benchmark_v2.py
python3 src/benchmark_c2l.py
```

### 4. Synthetic Marginal Density Shift Battery
Evaluate 12 experimental conditions ($N=5000$, $n=10$ replicates):
```bash
python3 src/run_density_sweep_comprehensive.py
```

### 5. Multiplicity Correction & Statistical Tests
Compute paired Wilcoxon tests, bootstrap 95% confidence intervals, and Holm-Bonferroni adjusted $p$-values:
```bash
python3 src/compute_stats_and_multiplicity.py
```

### 6. Generate All Figures
```bash
python3 src/plot_knots.py
python3 src/plot_results.py
python3 src/plot_multimodel.py
```

---

## License & Citation
Code is released under the MIT License for research reproducibility.
