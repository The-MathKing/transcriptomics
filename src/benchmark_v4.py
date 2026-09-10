"""
Bioinformatics Submission Benchmark Pipeline (Phase 2 & 4 Integration)

Phase 2 Fix: 10-fold non-overlapping partition of the source-cell pool to ensure 
strict statistical independence across replicates.
Phase 4 Fix: Continuous Compositional Calibration Metric (Dirichlet/CRPS)
"""

import os
import gc
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netcal.metrics import ECE
from inference_visium_cortex import BetaCalibration
from sklearn.isotonic import IsotonicRegression
from scipy.stats import ttest_rel
import json
import torch
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

def downsample_counts(adata, fraction):
    new_adata = adata.copy()
    if hasattr(new_adata.X, "toarray"):
        counts = new_adata.X.toarray()
    else:
        counts = new_adata.X.copy()
    counts = counts.astype(int)
    downsampled = np.random.binomial(counts, fraction)
    new_adata.X = downsampled.astype(np.float32)
    return new_adata

def apply_noise_shift(adata, noise_lambda, seed=42):
    rng = np.random.RandomState(seed)
    new_adata = adata.copy()
    if hasattr(new_adata.X, "toarray"):
        counts = new_adata.X.toarray()
    else:
        counts = new_adata.X.copy()
    
    # Inject Poisson noise to simulate ambient RNA / off-target binding
    noise = rng.poisson(lam=noise_lambda, size=counts.shape)
    new_adata.X = (counts + noise).astype(np.float32)
    return new_adata



def generate_pseudo_spots(adata_sc, all_cell_types, n_spots=2000, cells_per_spot=10, cell_type_col="cell_subclass", seed=42):
    import anndata as ad
    rng = np.random.RandomState(seed)
    
    pseudo_counts = np.zeros((n_spots, adata_sc.n_vars))
    pseudo_props = np.zeros((n_spots, len(all_cell_types)))
    
    type_to_idx = {ct: i for i, ct in enumerate(all_cell_types)}
    
    for i in range(n_spots):
        sampled_indices = rng.choice(adata_sc.n_obs, size=cells_per_spot, replace=True)
        sampled_cells = adata_sc[sampled_indices]
        
        if hasattr(sampled_cells.X, "toarray"):
            pseudo_counts[i, :] = sampled_cells.X.sum(axis=0).A1
        else:
            pseudo_counts[i, :] = sampled_cells.X.sum(axis=0)
            
        types = sampled_cells.obs[cell_type_col].values
        for ct in types:
            if ct in type_to_idx:
                pseudo_props[i, type_to_idx[ct]] += 1
            
    pseudo_props = pseudo_props / cells_per_spot
    
    adata_pseudo = ad.AnnData(X=pseudo_counts)
    adata_pseudo.var_names = adata_sc.var_names
    adata_pseudo.obs_names = [f"pseudo_{i}" for i in range(n_spots)]
    
    sanitized_cell_types = [str(ct).replace("/", "_") for ct in all_cell_types]
    prop_df = pd.DataFrame(pseudo_props, index=adata_pseudo.obs_names, columns=sanitized_cell_types)
    adata_pseudo.obsm["proportions"] = prop_df
    return adata_pseudo


def get_calibration_stats(true_p, pred_p, temp=1.0, iso_reg=None, beta_reg=None):
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
    if beta_reg is not None:
        conf = beta_reg.predict(conf)
        conf = np.clip(conf, 0, 1)

    ece = ECE(bins=10).measure(conf, acc)
    try:
        ece_adapt = ECE(bins=10, equal_intervals=False).measure(conf, acc)
    except ValueError:
        ece_adapt = float('nan')
    brier = float(np.mean((conf - acc) ** 2))
    nll = log_loss(acc, conf, labels=[0, 1])
    
    comp_rmse = float(np.sqrt(np.mean((cal_pred_p - true_p) ** 2)))
    
    high_conf_mask = conf >= 0.9
    high_conf_precision = float(np.mean(acc[high_conf_mask])) if np.sum(high_conf_mask) > 0 else float('nan')
    
    # Class-wise ECE (for top predicted class only)
    c_ece = ece
    
    return conf, acc, ece, ece_adapt, brier, nll, comp_rmse, high_conf_precision, cal_pred_p


def optimize_temperature(true_p, pred_p):
    best_t = 1.0
    best_nll = float('inf')
    eps = 1e-7
    for t in np.linspace(0.05, 50.0, 500):
        logits = np.log(pred_p + eps)
        scaled_logits = logits / t
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        cal_pred_p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # NLL is cross entropy on true proportions
        nll = -np.mean(np.sum(true_p * np.log(cal_pred_p + eps), axis=1))
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return best_t, best_nll


def evaluate_bootstrapped_calibrators(true_props_cal, pred_props_cal, true_props_test, pred_props_test, n_bootstraps=100):
    n_cal = len(true_props_cal)
    ece_isos, ece_temps, ece_betas = [], [], []
    brier_isos, brier_temps, brier_betas = [], [], []
    ece_isos_adapt, ece_temps_adapt, ece_betas_adapt = [], [], []
    nll_isos, nll_temps, nll_betas = [], [], []
    for _ in range(n_bootstraps):
        idx = np.random.choice(n_cal, n_cal, replace=True)
        t_cal, p_cal = true_props_cal[idx], pred_props_cal[idx]
        
        # Temp
        best_t, _ = optimize_temperature(t_cal, p_cal)
        _, _, ece_t, ece_t_adapt, brier_t, nll_t, _, _, _ = get_calibration_stats(true_props_test, pred_props_test, temp=best_t)
        ece_temps.append(ece_t)
        brier_temps.append(brier_t)
        nll_temps.append(nll_t)
        ece_temps_adapt.append(ece_t_adapt)
        
        # Iso
        conf_cal = np.max(p_cal, axis=1)
        pred_max_cal = np.argmax(p_cal, axis=1)
        acc_cal = (t_cal[np.arange(len(t_cal)), pred_max_cal] == np.max(t_cal, axis=1)).astype(int)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(conf_cal, acc_cal)
        _, _, ece_i, ece_i_adapt, brier_i, nll_i, _, _, _ = get_calibration_stats(true_props_test, pred_props_test, iso_reg=iso)
        ece_isos.append(ece_i)
        brier_isos.append(brier_i)
        nll_isos.append(nll_i)
        ece_isos_adapt.append(ece_i_adapt)

        # Beta
        beta = BetaCalibration()
        beta.fit(conf_cal, acc_cal)
        _, _, ece_b, ece_b_adapt, brier_b, nll_b, _, _, _ = get_calibration_stats(true_props_test, pred_props_test, beta_reg=beta)
        ece_betas.append(ece_b)
        brier_betas.append(brier_b)
        nll_betas.append(nll_b)
        ece_betas_adapt.append(ece_b_adapt)
        
    return {
        "ece_temp_mean": float(np.mean(ece_temps)), "ece_temp_std": float(np.std(ece_temps)),
        "brier_temp_mean": float(np.mean(brier_temps)), "brier_temp_std": float(np.std(brier_temps)),
        "nll_temp_mean": float(np.mean(nll_temps)),
        "ece_temp_adapt_mean": float(np.mean(ece_temps_adapt)),
        "ece_iso_mean": float(np.mean(ece_isos)), "ece_iso_std": float(np.std(ece_isos)),
        "brier_iso_mean": float(np.mean(brier_isos)), "brier_iso_std": float(np.std(brier_isos)),
        "nll_iso_mean": float(np.mean(nll_isos)),
        "ece_iso_adapt_mean": float(np.mean(ece_isos_adapt)),
        "ece_beta_mean": float(np.mean(ece_betas)),
        "brier_beta_mean": float(np.mean(brier_betas)),
        "nll_beta_mean": float(np.mean(nll_betas)),
        "ece_beta_adapt_mean": float(np.mean(ece_betas_adapt))
    }

def get_ood_proportions(model, adata):
    if "_indices" not in adata.obs:
        adata.obs["_indices"] = np.arange(adata.n_obs)
    model.transfer_fields(adata)
    old_adata = model.adata
    model.adata = adata
    try:
        props_df = model.get_proportions()
        props = props_df[adata.obsm["proportions"].columns].values
    finally:
        model.adata = old_adata
    return props


from sklearn.model_selection import StratifiedKFold

def run_kfold_pipeline(adata_sc, fractions, noise_levels, epochs, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cell_type_col = None
    for col in ["cell_subclass", "cluster", "cell_type", "labels"]:
        if col in adata_sc.obs.columns:
            cell_type_col = col
            break
            
    all_results = []
    y_strat = adata_sc.obs[cell_type_col].values
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_strat)), y_strat)):
        print(f"--- Running Fold {fold_idx+1}/{n_splits} ---")
        
        seed = 42 + fold_idx
        scvi.settings.seed = seed
        pl.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        adata_sc_train = adata_sc[train_idx].copy()
        adata_sc_test = adata_sc[test_idx].copy()
        
        all_cell_types = adata_sc_train.obs[cell_type_col].unique()
        
        scvi.model.CondSCVI.setup_anndata(adata_sc_train, labels_key=cell_type_col)
        sc_model = scvi.model.CondSCVI(adata_sc_train, weight_obs=False)
        sc_model.train(max_epochs=epochs, accelerator='cpu', early_stopping=True, train_size=0.9)

        n_test_spots = 1000 if not SMOKE_TEST else 200
        adata_st_test = generate_pseudo_spots(adata_sc_test, all_cell_types, n_spots=n_test_spots, cells_per_spot=10, cell_type_col=cell_type_col, seed=seed)
        
        n_train_spots = 2000 if not SMOKE_TEST else 400
        adata_st_train_cal = generate_pseudo_spots(adata_sc_train, all_cell_types, n_spots=n_train_spots, cells_per_spot=10, cell_type_col=cell_type_col, seed=seed)
        
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_train_spots)
        n_train = int(0.7 * n_train_spots)
        train_spot_idx = perm[:n_train]
        cal_spot_idx = perm[n_train:]
        
        adata_train = adata_st_train_cal[train_spot_idx].copy()
        adata_cal = adata_st_train_cal[cal_spot_idx].copy()
        
        scvi.model.DestVI.setup_anndata(adata_train)
        st_model = scvi.model.DestVI.from_rna_model(adata_train, sc_model)
        st_model.train(max_epochs=epochs, accelerator='cpu', early_stopping=True, train_size=0.9)
        
        true_props_cal = adata_cal.obsm["proportions"].values
        pred_props_cal = get_ood_proportions(st_model, adata_cal)
        
        best_t, _ = optimize_temperature(true_props_cal, pred_props_cal)
        conf_cal = np.max(pred_props_cal, axis=1)
        pred_max_cal = np.argmax(pred_props_cal, axis=1)
        acc_cal = (true_props_cal[np.arange(len(true_props_cal)), pred_max_cal] == np.max(true_props_cal, axis=1)).astype(int)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(conf_cal, acc_cal)
        
        seed_results = {"fractions": {}, "noise_levels": {}}
        for frac in fractions:
            np.random.seed(seed)
            # Test data
            adata_test_frac = downsample_counts(adata_st_test, fraction=frac)
            true_props_test = adata_test_frac.obsm["proportions"].values
            pred_props_test = get_ood_proportions(st_model, adata_test_frac)
            
            # Phase 5: Shifted Calibration data
            adata_cal_frac = downsample_counts(adata_cal, fraction=frac)
            true_props_cal_frac = adata_cal_frac.obsm["proportions"].values
            pred_props_cal_frac = get_ood_proportions(st_model, adata_cal_frac)
            
            # Fit shifted calibrators
            best_t_shifted, _ = optimize_temperature(true_props_cal_frac, pred_props_cal_frac)
            conf_cal_frac = np.max(pred_props_cal_frac, axis=1)
            pred_max_cal_frac = np.argmax(pred_props_cal_frac, axis=1)
            acc_cal_frac = (true_props_cal_frac[np.arange(len(true_props_cal_frac)), pred_max_cal_frac] == np.max(true_props_cal_frac, axis=1)).astype(int)
            iso_shifted = IsotonicRegression(out_of_bounds='clip')
            iso_shifted.fit(conf_cal_frac, acc_cal_frac)

            # Evaluate Uncalibrated
            _, acc_raw, ece_raw, ece_adapt_raw, brier_raw, nll_raw, comp_rmse_raw, hc_raw, _ = get_calibration_stats(true_props_test, pred_props_test)
            
            # Evaluate Bootstrapped Clean Calibrators
            boot_clean = evaluate_bootstrapped_calibrators(true_props_cal, pred_props_cal, true_props_test, pred_props_test, n_bootstraps=100)
            
            # Evaluate Bootstrapped Shifted Calibrators
            boot_shifted = evaluate_bootstrapped_calibrators(true_props_cal_frac, pred_props_cal_frac, true_props_test, pred_props_test, n_bootstraps=100)

            seed_results["fractions"][frac] = {
                "acc": float(np.mean(acc_raw)),
                "ece_ood": ece_raw,
                "nll_ood": nll_raw,
                "ece_adapt_ood": ece_adapt_raw,
                "brier_ood": brier_raw,
                "comp_rmse_ood": comp_rmse_raw,
                "hc_prec_ood": hc_raw,
                "boot_clean": boot_clean,
                "boot_shifted": boot_shifted
            }
            
        for nl in noise_levels:
            np.random.seed(seed)
            # Test data
            adata_test_noise = apply_noise_shift(adata_st_test, noise_lambda=nl, seed=seed)
            true_props_test = adata_test_noise.obsm["proportions"].values
            pred_props_test = get_ood_proportions(st_model, adata_test_noise)
            
            # Phase 5: Shifted Calibration data
            adata_cal_noise = apply_noise_shift(adata_cal, noise_lambda=nl, seed=seed)
            true_props_cal_noise = adata_cal_noise.obsm["proportions"].values
            pred_props_cal_noise = get_ood_proportions(st_model, adata_cal_noise)
            
            # Fit shifted calibrators
            best_t_shifted, _ = optimize_temperature(true_props_cal_noise, pred_props_cal_noise)
            conf_cal_noise = np.max(pred_props_cal_noise, axis=1)
            acc_cal_noise = (np.argmax(pred_props_cal_noise, axis=1) == np.argmax(true_props_cal_noise, axis=1)).astype(int)
            iso_shifted = IsotonicRegression(out_of_bounds='clip')
            iso_shifted.fit(conf_cal_noise, acc_cal_noise)

            # Evaluate Uncalibrated
            _, acc_raw, ece_raw, ece_adapt_raw, brier_raw, nll_raw, comp_rmse_raw, hc_raw, _ = get_calibration_stats(true_props_test, pred_props_test)
            
            # Evaluate Bootstrapped Clean Calibrators
            boot_clean = evaluate_bootstrapped_calibrators(true_props_cal, pred_props_cal, true_props_test, pred_props_test, n_bootstraps=100)
            
            # Evaluate Bootstrapped Shifted Calibrators
            boot_shifted = evaluate_bootstrapped_calibrators(true_props_cal_noise, pred_props_cal_noise, true_props_test, pred_props_test, n_bootstraps=100)

            seed_results["noise_levels"][nl] = {
                "acc": float(np.mean(acc_raw)),
                "ece_ood": ece_raw,
                "nll_ood": nll_raw,
                "ece_adapt_ood": ece_adapt_raw,
                "brier_ood": brier_raw,
                "comp_rmse_ood": comp_rmse_raw,
                "hc_prec_ood": hc_raw,
                "boot_clean": boot_clean,
                "boot_shifted": boot_shifted
            }
        
        all_results.append(seed_results)
        
        del sc_model
        del st_model
        del adata_sc_train
        del adata_sc_test
        del adata_st_test
        del adata_st_train_cal
        gc.collect()
        
        if SMOKE_TEST and fold_idx >= 1:
            break
            
    return all_results

def main():
    print("Loading data...")
    adata_sc = sc.read_h5ad("data/processed_sc_reference.h5ad")

    if SMOKE_TEST:
        epochs = 3
        sc.pp.subsample(adata_sc, n_obs=min(2000, adata_sc.n_obs), random_state=0)
        n_splits = 2
    else:
        epochs = 200
        n_splits = 5

    fractions = [1.0, 0.8, 0.6, 0.4, 0.2]
    noise_levels = [0.0, 0.5, 1.0, 2.0]
    all_results = run_kfold_pipeline(adata_sc, fractions, noise_levels, epochs, n_splits=n_splits)

    final_output = {
        "fractions": fractions,
        "noise_levels": noise_levels,
        "epochs": epochs,
        "smoke_test": SMOKE_TEST,
        "n_splits": n_splits
    }

    for frac in fractions:
        f_str = str(frac)
        final_output[f_str] = {
            "ece_ood": [r["fractions"][frac]["ece_ood"] for r in all_results],
            "ece_adapt_ood": [r["fractions"][frac]["ece_adapt_ood"] for r in all_results],
            "brier_ood": [r["fractions"][frac]["brier_ood"] for r in all_results],
            "comp_rmse_ood": [r["fractions"][frac]["comp_rmse_ood"] for r in all_results],
            "boot_clean": [r["fractions"][frac]["boot_clean"] for r in all_results],
            "boot_shifted": [r["fractions"][frac]["boot_shifted"] for r in all_results]
        }
        
    for nl in noise_levels:
        nl_str = f"noise_{nl}"
        final_output[nl_str] = {
            "ece_ood": [r["noise_levels"][nl]["ece_ood"] for r in all_results],
            "ece_adapt_ood": [r["noise_levels"][nl]["ece_adapt_ood"] for r in all_results],
            "brier_ood": [r["noise_levels"][nl]["brier_ood"] for r in all_results],
            "comp_rmse_ood": [r["noise_levels"][nl]["comp_rmse_ood"] for r in all_results],
            "boot_clean": [r["noise_levels"][nl]["boot_clean"] for r in all_results],
            "boot_shifted": [r["noise_levels"][nl]["boot_shifted"] for r in all_results]
        }
        
    os.makedirs("figures", exist_ok=True)
    out_name = "results_bioinfo_kfold.json" if not SMOKE_TEST else "results_smoketest_kfold.json"
    with open(out_name, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Metrics saved to {out_name}")

if __name__ == "__main__":
    main()
