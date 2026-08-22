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


def get_calibration_stats(true_p, pred_p, temp=1.0, iso_reg=None):
    eps = 1e-7
    logits = np.log(pred_p + eps)
    scaled_logits = logits / temp

    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    cal_pred_p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    conf = np.max(cal_pred_p, axis=1)
    acc = (np.argmax(cal_pred_p, axis=1) == np.argmax(true_p, axis=1)).astype(int)

    if iso_reg is not None:
        conf = iso_reg.predict(conf)
        conf = np.clip(conf, 0, 1)

    ece = ECE(bins=10).measure(conf, acc)
    return conf, acc, ece, cal_pred_p


def optimize_temperature(true_p, pred_p):
    best_t = 1.0
    best_ece = float('inf')
    for t in np.linspace(0.5, 3.0, 50):
        _, _, ece, _ = get_calibration_stats(true_p, pred_p, temp=t)
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
        props = model.get_proportions().values
    finally:
        model.adata = old_adata
    return props


def run_seed_pipeline(adata_sc, adata_st, seed, fractions, epochs):
    scvi.settings.seed = seed
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(seed)

    cell_type_col = None
    for col in ["cell_subclass", "cluster", "cell_type", "labels"]:
        if col in adata_sc.obs.columns:
            cell_type_col = col
            break

    # --- Train CondSCVI prior on the (unshifted) single-cell reference ---
    scvi.model.CondSCVI.setup_anndata(adata_sc, labels_key=cell_type_col)
    sc_model = scvi.model.CondSCVI(adata_sc, weight_obs=False)
    sc_model.train(max_epochs=epochs, accelerator='cpu', early_stopping=True, train_size=0.9)

    # --- Split pseudo-spots into train / calibration / test (clean, per-seed) ---
    n_spots = adata_st.n_obs
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_spots)
    n_train = int(0.5 * n_spots)
    n_cal = int(0.25 * n_spots)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train + n_cal]
    test_idx = perm[n_train + n_cal:]

    adata_train = adata_st[train_idx].copy()
    adata_cal = adata_st[cal_idx].copy()
    adata_test_base = adata_st[test_idx].copy()

    # --- Train ONE DestVI model, only on the clean training split ---
    scvi.model.DestVI.setup_anndata(adata_train)
    st_model = scvi.model.DestVI.from_rna_model(adata_train, sc_model)
    st_model.train(max_epochs=epochs, accelerator='cpu', early_stopping=True, train_size=0.9)

    # --- Fit calibration (Temperature Scaling, Isotonic Regression) ONCE,
    #     on the clean held-out calibration split. Never re-fit per fraction. ---
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
        np.random.seed(seed)  # same reseeding convention as before, for determinism
        adata_test_frac = downsample_counts(adata_test_base, fraction=frac)

        true_props_test = adata_test_frac.obsm["proportions"].values
        pred_props_test = get_ood_proportions(st_model, adata_test_frac)

        _, acc_raw, ece_raw, _ = get_calibration_stats(true_props_test, pred_props_test)
        _, _, ece_temp, _ = get_calibration_stats(true_props_test, pred_props_test, temp=best_t)
        _, _, ece_iso, _ = get_calibration_stats(true_props_test, pred_props_test, iso_reg=iso)

        seed_results["fractions"][frac] = {
            "acc": float(np.mean(acc_raw)),
            "ece_ood": ece_raw,
            "ece_temp": ece_temp,
            "ece_iso": ece_iso,
            "best_t": best_t,
        }

    return seed_results


def main():
    print("Loading data...")
    adata_sc = sc.read_h5ad("data/processed_sc_reference.h5ad")
    adata_st = sc.read_h5ad("data/processed_pseudospots.h5ad")

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
        res = run_seed_pipeline(adata_sc, adata_st, seed, fractions, epochs)
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
            "ece_temp": [r["fractions"][frac]["ece_temp"] for r in all_results],
            "ece_iso": [r["fractions"][frac]["ece_iso"] for r in all_results],
        }
        mean_acc = float(np.mean(final_output[f_str]["acc"]))
        std_acc = float(np.std(final_output[f_str]["acc"]))
        mean_ood = float(np.mean(final_output[f_str]["ece_ood"]))
        std_ood = float(np.std(final_output[f_str]["ece_ood"]))
        mean_temp = float(np.mean(final_output[f_str]["ece_temp"]))
        std_temp = float(np.std(final_output[f_str]["ece_temp"]))
        mean_iso = float(np.mean(final_output[f_str]["ece_iso"]))
        std_iso = float(np.std(final_output[f_str]["ece_iso"]))

        t_stat_iso, p_val_iso = ttest_rel(final_output[f_str]["ece_temp"], final_output[f_str]["ece_iso"])

        final_output[f_str].update({
            "mean_acc": mean_acc, "std_acc": std_acc,
            "mean_ood": mean_ood, "std_ood": std_ood,
            "mean_temp": mean_temp, "std_temp": std_temp,
            "mean_iso": mean_iso, "std_iso": std_iso,
            "ttest_temp_vs_iso": {"t_stat": float(t_stat_iso), "p_val": float(p_val_iso)},
        })

        if frac != 1.0:
            t_stat, p_val = ttest_rel(final_output["1.0"]["ece_ood"], final_output[f_str]["ece_ood"])
            final_output[f_str]["ttest_vs_id"] = {"t_stat": float(t_stat), "p_val": float(p_val)}

        print(f"\nFraction {frac}:")
        print(f"  Top-1 Accuracy:   {mean_acc:.3f} +/- {std_acc:.3f}")
        print(f"  Uncalibrated ECE: {mean_ood:.3f} +/- {std_ood:.3f}")
        print(f"  Temp Scaling ECE: {mean_temp:.3f} +/- {std_temp:.3f}")
        print(f"  Isotonic ECE: {mean_iso:.3f} +/- {std_iso:.3f} (p={p_val_iso:.4e} vs Temp)")

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
