"""
Corrected calibration-under-shift benchmark.

DESIGN CHANGE FROM v9 (documented here so the paper text can cite it accurately):
  Old design: for each shift fraction, a FRESH DestVI model was trained directly on
  the shifted data, then split 50/50 into a calibration/test set drawn from that same
  shifted distribution. This meant (a) the model was never actually deployed on a
  distribution different from what it was fit to, so it wasn't testing calibration
  under deployment-time shift, and (b) calibration and test were IID samples of the
  same shifted distribution, which structurally favors a flexible non-parametric
  calibrator like Isotonic Regression over a 1-parameter method like Temperature
  Scaling regardless of anything specific to spatial transcriptomics.

  New design: ONE DestVI model is trained once per seed on a held-out CLEAN training
  split. Temperature Scaling and Isotonic Regression are fit ONCE on a held-out CLEAN
  calibration split (never shifted). That single frozen model and those fixed
  calibration mappings are then evaluated, unchanged, on a third held-out TEST split
  that gets shifted across the fraction sweep (1.0 = clean baseline, down to 0.2 =
  80% dropout). This is now an actual test of whether calibration learned on clean
  data generalizes as deployment-time shift increases -- the claim the paper's title
  and framing were always meant to support.

  Side benefit: the old "ID ECE" was computed by evaluating the model on the SAME
  data it was trained on (no held-out ID test set existed at all). The new fraction
  1.0 case fixes this too -- ID ECE is now genuinely held-out, using the same
  train/cal/test split machinery as every OOD fraction.

SPLIT: pseudo-spots are split 50% train / 25% calibration / 25% test per seed
(reshuffled per seed for robustness). With n_spots=2000 that's 1000/500/500.

Before committing to the full 10-seed x 25-epoch run, run with SMOKE_TEST=True
below (or `SMOKE_TEST=1 python src/benchmark.py`) to sanity-check the pipeline
end-to-end in a couple minutes: 2 seeds, 3 epochs, and a subsampled dataset.
The one line most likely to need adjustment on your machine is the
`model.get_proportions(adata=...)` out-of-sample-inference call in
`eval_on_split` -- if your installed scvi-tools version errors there, see the
comment next to it for what to check.
"""

import os
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netcal.metrics import ECE
from sklearn.isotonic import IsotonicRegression
from scipy.stats import ttest_rel
import json
import torch
import pytorch_lightning as pl

# Configure scvi data loader workers for faster I/O
scvi.settings.num_workers = 12  # faster I/O on multi‑core machines

# Enable cuDNN auto‑tuner (helps on CUDA GPUs)
import torch
torch.backends.cudnn.benchmark = True


def downsample_counts(adata, fraction):
    # Simulate lower capture rate via binomial dropout.
    # fraction=1.0 is deterministically a no-op (binomial(n, p=1) == n), so this
    # is safe to call uniformly across the whole fraction sweep including the
    # clean baseline -- no special-casing needed.
    new_adata = adata.copy()
    if hasattr(new_adata.X, "toarray"):
        counts = new_adata.X.toarray()
    else:
        counts = new_adata.X.copy()

    counts = counts.astype(int)
    downsampled = np.random.binomial(counts, fraction)
    new_adata.X = downsampled.astype(np.float32)
    return new_adata


def generate_pseudo_spots(adata_sc, n_spots=2000, cells_per_spot=10, cell_type_col="cell_subclass", seed=42):
    import anndata as ad
    rng = np.random.RandomState(seed)
    cell_types = adata_sc.obs[cell_type_col].unique()
    pseudo_counts = np.zeros((n_spots, adata_sc.n_vars))
    pseudo_props = np.zeros((n_spots, len(cell_types)))
    
    type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    for i in range(n_spots):
        sampled_indices = rng.choice(adata_sc.n_obs, size=cells_per_spot, replace=True)
        sampled_cells = adata_sc[sampled_indices]
        
        if hasattr(sampled_cells.X, "toarray"):
            pseudo_counts[i, :] = sampled_cells.X.sum(axis=0).A1
        else:
            pseudo_counts[i, :] = sampled_cells.X.sum(axis=0)
            
        types = sampled_cells.obs[cell_type_col].values
        for ct in types:
            pseudo_props[i, type_to_idx[ct]] += 1
            
    pseudo_props = pseudo_props / cells_per_spot
    
    adata_pseudo = ad.AnnData(X=pseudo_counts)
    adata_pseudo.var_names = adata_sc.var_names
    adata_pseudo.obs_names = [f"pseudo_{i}" for i in range(n_spots)]
    
    sanitized_cell_types = [str(ct).replace("/", "_") for ct in cell_types]
    prop_df = pd.DataFrame(pseudo_props, index=adata_pseudo.obs_names, columns=sanitized_cell_types)
    adata_pseudo.obsm["proportions"] = prop_df
    return adata_pseudo


def get_calibration_stats(true_p, pred_p, temp=1.0, iso_reg=None):
    eps = 1e-7
    logits = np.log(pred_p + eps)
    scaled_logits = logits / temp

    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    cal_pred_p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    conf = np.max(cal_pred_p, axis=1)
    pred_max_idx = np.argmax(cal_pred_p, axis=1)
    acc = (true_p[np.arange(len(true_p)), pred_max_idx] == np.max(true_p, axis=1)).astype(int)

    if iso_reg is not None:
        conf = iso_reg.predict(conf)
        conf = np.clip(conf, 0, 1)

    ece = ECE(bins=10).measure(conf, acc)
    brier = float(np.mean((conf - acc) ** 2))
    return conf, acc, ece, brier, cal_pred_p


def optimize_temperature(true_p, pred_p):
    best_t = 1.0
    best_ece = float('inf')
    for t in np.linspace(0.5, 3.0, 50):
        _, _, ece, _, _ = get_calibration_stats(true_p, pred_p, temp=t)
        if ece < best_ece:
            best_ece = ece
            best_t = t
    return best_t, best_ece


def get_ood_proportions(model, adata):
    """
    Out-of-sample inference: run the FROZEN trained model's encoder on new data
    it was never trained on, without retraining. Standard scvi-tools pattern --
    most BaseModelClass get_* methods accept adata= and internally transfer the
    training-time field registry onto the new AnnData (same var_names required).
    If this raises an AnnDataManager / registry error on your scvi-tools version,
    that's the one thing to look up first (search "scvi-tools out-of-sample
    inference transfer_fields" for your installed version's equivalent call).
    """
    # DestVI.get_proportions doesn't accept an adata argument in scvi-tools v1.5,
    # so we must register the fields and temporarily swap the model's adata attribute.
    if "_indices" not in adata.obs:
        import numpy as np
        adata.obs["_indices"] = np.arange(adata.n_obs)
    model.transfer_fields(adata)
    old_adata = model.adata
    model.adata = adata
    try:
        props_df = model.get_proportions()
        # Reindex to ensure all expected cell type columns are present
        props_df = props_df.reindex(columns=adata.obsm['proportions'].columns, fill_value=0)
        props = props_df[adata.obsm['proportions'].columns].values
    finally:
        model.adata = old_adata
    return props


def run_seed_pipeline(adata_sc, seed, fractions, epochs):
    # Set random seeds for reproducibility
    scvi.settings.seed = seed
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(seed)

    # Choose accelerator based on available hardware – use GPU only if CUDA is present.
    if torch.cuda.is_available():
        accelerator = 'gpu'          # NVIDIA/AMD CUDA GPU
    elif torch.backends.mps.is_available():
        # MPS (Apple Silicon) is not fully supported by scvi; fall back to CPU.
        accelerator = 'cpu'
    else:
        accelerator = 'cpu'

    # Identify the cell type column in the AnnData
    cell_type_col = None
    for col in ["cell_subclass", "cluster", "cell_type", "labels"]:
        if col in adata_sc.obs.columns:
            cell_type_col = col
            break

    # --- Split single‑cell reference for independent pseudo‑spot generation ---
    n_sc = adata_sc.n_obs
    rng = np.random.RandomState(seed)
    perm_sc = rng.permutation(n_sc)
    n_sc_train = int(0.5 * n_sc)
    sc_train_idx = perm_sc[:n_sc_train]
    sc_test_idx = perm_sc[n_sc_train:]

    adata_sc_train = adata_sc[sc_train_idx].copy()
    adata_sc_test = adata_sc[sc_test_idx].copy()

    # --- Train CondSCVI prior ONLY on the SC training split ---
    scvi.model.CondSCVI.setup_anndata(adata_sc_train, labels_key=cell_type_col)
    sc_model = scvi.model.CondSCVI(adata_sc_train, weight_obs=False)
    # Set prior for the model
    sc_model.init_params_["kwargs"] = {"module_kwargs": {"prior": "lognorm"}}
    # Train CondSCVI model using the chosen accelerator
    sc_model.train(max_epochs=epochs, accelerator=accelerator, precision=16, early_stopping=True, train_size=0.9)
    sc_model.is_trained_ = True

    # --- Generate fresh pseudo‑spots ONLY from the SC test split ---
    adata_st = generate_pseudo_spots(adata_sc_test, n_spots=2000, cells_per_spot=10, cell_type_col=cell_type_col, seed=seed)

    # --- Split pseudo‑spots into train / calibration / test (clean, per‑seed) ---
    n_spots = adata_st.n_obs
    perm = rng.permutation(n_spots)
    n_train = int(0.5 * n_spots)
    n_cal = int(0.25 * n_spots)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train + n_cal]
    test_idx = perm[n_train + n_cal:]

    adata_train = adata_st[train_idx].copy()
    adata_cal = adata_st[cal_idx].copy()
    adata_test_base = adata_st[test_idx].copy()

    # Initialize DestVI using the trained CondSCVI model
    st_model = scvi.model.DestVI.from_rna_model(adata_train, sc_model)
    # Train DestVI model using the chosen accelerator
    st_model.train(max_epochs=epochs, accelerator=accelerator, precision=16, early_stopping=True, train_size=0.9)

    # --- Fit calibration (Temperature Scaling, Isotonic Regression) ONCE on the clean calibration split ---
    true_props_cal = adata_cal.obsm["proportions"].values
    pred_props_cal = get_ood_proportions(st_model, adata_cal)
    best_t, _ = optimize_temperature(true_props_cal, pred_props_cal)

    conf_cal = np.max(pred_props_cal, axis=1)
    acc_cal = (np.argmax(pred_props_cal, axis=1) == np.argmax(true_props_cal, axis=1)).astype(int)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(conf_cal, acc_cal)

    # --- Evaluate the frozen model + fixed calibration across the shift sweep ---
    seed_results = {"fractions": {}}
    for frac in fractions:
        print(f"    Evaluating fraction {frac}...")
        np.random.seed(seed)
        adata_test_frac = downsample_counts(adata_test_base, fraction=frac)
        true_props_test = adata_test_frac.obsm["proportions"].values
        pred_props_test = get_ood_proportions(st_model, adata_test_frac)
        _, acc_raw, ece_raw, brier_raw, _ = get_calibration_stats(true_props_test, pred_props_test)
        _, _, ece_temp, brier_temp, _ = get_calibration_stats(true_props_test, pred_props_test, temp=best_t)
        _, _, ece_iso, brier_iso, _ = get_calibration_stats(true_props_test, pred_props_test, iso_reg=iso)
        seed_results["fractions"][frac] = {
            "acc": float(np.mean(acc_raw)),
            "ece_ood": ece_raw,
            "brier_ood": brier_raw,
            "ece_temp": ece_temp,
            "brier_temp": brier_temp,
            "ece_iso": ece_iso,
            "brier_iso": brier_iso,
            "best_t": best_t,
        }
    return seed_results


def main():
    print("Loading data...")
    adata_sc = sc.read_h5ad("data/processed_sc_reference.h5ad")

    if SMOKE_TEST:
        print("*** SMOKE_TEST=1: reduced seeds/epochs/data for a fast sanity check ***")
        seeds = [42, 123]
        epochs = 3
        sc.pp.subsample(adata_sc, n_obs=min(2000, adata_sc.n_obs), random_state=0)
    else:
        seeds = [42, 123, 2026, 777, 999, 1001, 2002, 3003, 4004, 5005]
        epochs = 200

    # 1.0 = clean baseline (replaces the old separately-computed "ID ECE"; it now
    # goes through the exact same held-out-split / frozen-model / frozen-calibration
    # pipeline as every OOD fraction, just with no dropout applied).
    fractions = [1.0, 0.8, 0.6, 0.4, 0.2]

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"--- Running Replicate {i+1}/{len(seeds)} (Seed {seed}) ---")
        res = run_seed_pipeline(adata_sc, seed, fractions, epochs)
        all_results.append(res)

    final_output = {
        "seeds": seeds,
        "fractions": fractions,
        "split": {"train_frac": 0.5, "cal_frac": 0.25, "test_frac": 0.25},
        "epochs": epochs,
        "smoke_test": SMOKE_TEST,
    }

    for frac in fractions:
        f_str = str(frac)
        final_output[f_str] = {
            "acc": [r["fractions"][frac]["acc"] for r in all_results],
            "ece_ood": [r["fractions"][frac]["ece_ood"] for r in all_results],
            "brier_ood": [r["fractions"][frac]["brier_ood"] for r in all_results],
            "ece_temp": [r["fractions"][frac]["ece_temp"] for r in all_results],
            "brier_temp": [r["fractions"][frac]["brier_temp"] for r in all_results],
            "ece_iso": [r["fractions"][frac]["ece_iso"] for r in all_results],
            "brier_iso": [r["fractions"][frac]["brier_iso"] for r in all_results],
        }
        mean_acc = float(np.mean(final_output[f_str]["acc"]))
        std_acc = float(np.std(final_output[f_str]["acc"]))
        mean_ood = float(np.mean(final_output[f_str]["ece_ood"]))
        std_ood = float(np.std(final_output[f_str]["ece_ood"]))
        mean_brier_ood = float(np.mean(final_output[f_str]["brier_ood"]))
        std_brier_ood = float(np.std(final_output[f_str]["brier_ood"]))
        mean_temp = float(np.mean(final_output[f_str]["ece_temp"]))
        std_temp = float(np.std(final_output[f_str]["ece_temp"]))
        mean_brier_temp = float(np.mean(final_output[f_str]["brier_temp"]))
        std_brier_temp = float(np.std(final_output[f_str]["brier_temp"]))
        mean_iso = float(np.mean(final_output[f_str]["ece_iso"]))
        std_iso = float(np.std(final_output[f_str]["ece_iso"]))
        mean_brier_iso = float(np.mean(final_output[f_str]["brier_iso"]))
        std_brier_iso = float(np.std(final_output[f_str]["brier_iso"]))

        t_stat_iso, p_val_iso = ttest_rel(final_output[f_str]["ece_temp"], final_output[f_str]["ece_iso"])

        final_output[f_str].update({
            "mean_acc": mean_acc, "std_acc": std_acc,
            "mean_ood": mean_ood, "std_ood": std_ood,
            "mean_brier_ood": mean_brier_ood, "std_brier_ood": std_brier_ood,
            "mean_temp": mean_temp, "std_temp": std_temp,
            "mean_brier_temp": mean_brier_temp, "std_brier_temp": std_brier_temp,
            "mean_iso": mean_iso, "std_iso": std_iso,
            "mean_brier_iso": mean_brier_iso, "std_brier_iso": std_brier_iso,
            "ttest_temp_vs_iso": {"t_stat": float(t_stat_iso), "p_val": float(p_val_iso)},
        })

        if frac != 1.0:
            t_stat, p_val = ttest_rel(final_output["1.0"]["ece_ood"], final_output[f_str]["ece_ood"])
            final_output[f_str]["ttest_vs_id"] = {"t_stat": float(t_stat), "p_val": float(p_val)}

        print(f"\nFraction {frac}:")
        print(f"  Top-1 Accuracy:   {mean_acc:.3f} +/- {std_acc:.3f}")
        print(f"  Uncalibrated ECE: {mean_ood:.3f} +/- {std_ood:.3f} (Brier: {mean_brier_ood:.3f} +/- {std_brier_ood:.3f})")
        print(f"  Temp Scaling ECE: {mean_temp:.3f} +/- {std_temp:.3f} (Brier: {mean_brier_temp:.3f} +/- {std_brier_temp:.3f})")
        print(f"  Isotonic ECE: {mean_iso:.3f} +/- {std_iso:.3f} (Brier: {mean_brier_iso:.3f} +/- {std_brier_iso:.3f}) (p={p_val_iso:.4e} vs Temp ECE)")

    os.makedirs("figures", exist_ok=True)
    out_name = "results_smoketest.json" if SMOKE_TEST else "results.json"
    with open(out_name, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Metrics saved to {out_name}")

    print("Generating dose-response figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    x_plot = sorted(fractions, reverse=True)  # 1.0 -> 0.2

    y_ood = [final_output[str(f)]["mean_ood"] for f in x_plot]
    err_ood = [final_output[str(f)]["std_ood"] for f in x_plot]
    y_temp = [final_output[str(f)]["mean_temp"] for f in x_plot]
    err_temp = [final_output[str(f)]["std_temp"] for f in x_plot]
    y_iso = [final_output[str(f)]["mean_iso"] for f in x_plot]
    err_iso = [final_output[str(f)]["std_iso"] for f in x_plot]

    ax.errorbar(x_plot, y_ood, yerr=err_ood, label='Uncalibrated', marker='o', capsize=5)
    ax.errorbar(x_plot, y_temp, yerr=err_temp, label='Temperature Scaling', marker='s', capsize=5)
    ax.errorbar(x_plot, y_iso, yerr=err_iso, label='Isotonic Regression', marker='^', capsize=5)

    ax.set_xlabel('Capture Efficiency Fraction (1.0 = clean, held-out baseline)')
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title(f'Calibration Under Deployment-Time Shift (frozen model, n={len(seeds)})')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig_name = "figures/ece_degradation_smoketest.png" if SMOKE_TEST else "figures/ece_degradation.png"
    plt.savefig(fig_name, dpi=300)
    print(f"Saved dose-response figure to {fig_name}")


if __name__ == "__main__":
    main()
