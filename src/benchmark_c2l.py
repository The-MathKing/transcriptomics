import os
import gc
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression

from cell2location.models import RegressionModel, Cell2location
import warnings
warnings.filterwarnings('ignore')

from src.benchmark_v4 import (
    generate_pseudo_spots, 
    downsample_counts, 
    apply_noise_shift, 
    get_calibration_stats, 
    optimize_temperature,
    evaluate_bootstrapped_calibrators
)

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

def get_ood_proportions_c2l(adata, inf_dict, epochs=200):
    # Cell2location is not amortized. We must train it on the test spots to infer proportions.
    Cell2location.setup_anndata(adata)
    model = Cell2location(adata, cell_state_df=inf_dict, N_cells_per_location=10, detection_alpha=20)
    model.train(max_epochs=epochs, accelerator='cpu', early_stopping=False)
    adata = model.export_posterior(adata, sample_kwargs={'num_samples': 50, 'batch_size': adata.n_obs})
    
    abundance_key = [k for k in adata.obsm.keys() if 'abundance_w_sf' in k][0]
    df = adata.obsm[abundance_key]
    df.columns = [str(c).split('sf_')[-1].replace('/', '_') for c in df.columns]
    
    props = df[inf_dict.columns].values
    props = props / (np.sum(props, axis=1, keepdims=True) + 1e-9)
    return props

def run_kfold_c2l(adata_sc, fractions, noise_levels, epochs, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cell_type_col = None
    for col in ["cell_subclass", "cluster", "cell_type", "labels"]:
        if col in adata_sc.obs.columns:
            cell_type_col = col
            break
            
    all_results = []
    y_strat = adata_sc.obs[cell_type_col].values
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_strat)), y_strat)):
        print(f"--- Running Cell2Location Fold {fold_idx+1}/{n_splits} ---")
        
        seed = 42 + fold_idx
        scvi.settings.seed = seed
        pl.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        adata_sc_train = adata_sc[train_idx].copy()
        adata_sc_test = adata_sc[test_idx].copy()
        
        all_cell_types = adata_sc_train.obs[cell_type_col].unique()
        
        # 1. Train RegressionModel for signatures
        RegressionModel.setup_anndata(adata=adata_sc_train, labels_key=cell_type_col)
        mod_ref = RegressionModel(adata_sc_train)
        mod_ref.train(max_epochs=epochs, accelerator='cpu', early_stopping=False)
        adata_sc_train = mod_ref.export_posterior(
            adata_sc_train, sample_kwargs={'num_samples': 100, 'batch_size': mod_ref.adata.n_obs}
        )
        inf_dict = adata_sc_train.varm['means_per_cluster_mu_fg'].copy()
        inf_dict.columns = [str(c).replace('means_per_cluster_mu_fg_', '').replace('/', '_') for c in inf_dict.columns]

        n_test_spots = 1000 if not SMOKE_TEST else 100
        adata_st_test = generate_pseudo_spots(adata_sc_test, all_cell_types, n_spots=n_test_spots, cells_per_spot=10, cell_type_col=cell_type_col, seed=seed)
        
        n_train_spots = 2000 if not SMOKE_TEST else 200
        adata_st_train_cal = generate_pseudo_spots(adata_sc_train, all_cell_types, n_spots=n_train_spots, cells_per_spot=10, cell_type_col=cell_type_col, seed=seed)
        
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_train_spots)
        n_train = int(0.7 * n_train_spots)
        train_spot_idx = perm[:n_train]
        cal_spot_idx = perm[n_train:]
        
        # We actually just need calibration split because C2L trains per-spot.
        adata_cal = adata_st_train_cal[cal_spot_idx].copy()
        
        true_props_cal = adata_cal.obsm["proportions"][inf_dict.columns].values
        pred_props_cal = get_ood_proportions_c2l(adata_cal, inf_dict, epochs=epochs)
        
        seed_results = {"fractions": {}, "noise_levels": {}}
        for frac in fractions:
            np.random.seed(seed)
            adata_test_frac = downsample_counts(adata_st_test, fraction=frac)
            true_props_test = adata_test_frac.obsm["proportions"][inf_dict.columns].values
            pred_props_test = get_ood_proportions_c2l(adata_test_frac, inf_dict, epochs=epochs)
            
            adata_cal_frac = downsample_counts(adata_cal, fraction=frac)
            true_props_cal_frac = adata_cal_frac.obsm["proportions"][inf_dict.columns].values
            pred_props_cal_frac = get_ood_proportions_c2l(adata_cal_frac, inf_dict, epochs=epochs)
            
            _, acc_raw, ece_raw, ece_adapt_raw, brier_raw, nll_raw, comp_rmse_raw, hc_raw, _ = get_calibration_stats(true_props_test, pred_props_test)
            boot_clean = evaluate_bootstrapped_calibrators(true_props_cal, pred_props_cal, true_props_test, pred_props_test, n_bootstraps=500)
            boot_shifted = evaluate_bootstrapped_calibrators(true_props_cal_frac, pred_props_cal_frac, true_props_test, pred_props_test, n_bootstraps=500)

            seed_results["fractions"][frac] = {
                "acc": float(np.mean(acc_raw)),
                "ece_ood": ece_raw,
                "nll_ood": nll_raw,
                "ece_adapt_ood": ece_adapt_raw,
                "brier_ood": brier_raw,
                "comp_rmse_ood": comp_rmse_raw,
                "boot_clean": boot_clean,
                "boot_shifted": boot_shifted
            }
            
        for nl in noise_levels:
            np.random.seed(seed)
            adata_test_noise = apply_noise_shift(adata_st_test, noise_lambda=nl, seed=seed)
            true_props_test = adata_test_noise.obsm["proportions"][inf_dict.columns].values
            pred_props_test = get_ood_proportions_c2l(adata_test_noise, inf_dict, epochs=epochs)
            
            adata_cal_noise = apply_noise_shift(adata_cal, noise_lambda=nl, seed=seed)
            true_props_cal_noise = adata_cal_noise.obsm["proportions"][inf_dict.columns].values
            pred_props_cal_noise = get_ood_proportions_c2l(adata_cal_noise, inf_dict, epochs=epochs)
            
            _, acc_raw, ece_raw, ece_adapt_raw, brier_raw, nll_raw, comp_rmse_raw, hc_raw, _ = get_calibration_stats(true_props_test, pred_props_test)
            boot_clean = evaluate_bootstrapped_calibrators(true_props_cal, pred_props_cal, true_props_test, pred_props_test, n_bootstraps=500)
            boot_shifted = evaluate_bootstrapped_calibrators(true_props_cal_noise, pred_props_cal_noise, true_props_test, pred_props_test, n_bootstraps=500)

            seed_results["noise_levels"][nl] = {
                "acc": float(np.mean(acc_raw)),
                "ece_ood": ece_raw,
                "nll_ood": nll_raw,
                "ece_adapt_ood": ece_adapt_raw,
                "brier_ood": brier_raw,
                "comp_rmse_ood": comp_rmse_raw,
                "boot_clean": boot_clean,
                "boot_shifted": boot_shifted
            }
        
        all_results.append(seed_results)
        
        del mod_ref
        del adata_sc_train
        del adata_sc_test
        del adata_st_test
        gc.collect()
        
        if SMOKE_TEST and fold_idx >= 0:
            break
            
    return all_results

def main():
    print("Loading data...")
    adata_sc = sc.read_h5ad("data/processed_sc_reference.h5ad")
    if hasattr(adata_sc.X, "toarray"):
        adata_sc.X = np.round(adata_sc.X.toarray()).astype(np.float32)
    else:
        adata_sc.X = np.round(adata_sc.X).astype(np.float32)
    from scipy.sparse import csr_matrix
    adata_sc.X = csr_matrix(adata_sc.X)

    if SMOKE_TEST:
        epochs = 3
        sc.pp.subsample(adata_sc, n_obs=min(500, adata_sc.n_obs), random_state=0)
        n_splits = 2
    else:
        epochs = 150
        sc.pp.subsample(adata_sc, n_obs=min(3000, adata_sc.n_obs), random_state=0)
        n_splits = 5

    fractions = [1.0, 0.6, 0.2]
    noise_levels = [0.0, 1.0, 2.0]
    all_results = run_kfold_c2l(adata_sc, fractions, noise_levels, epochs, n_splits=n_splits)

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
            "brier_ood": [r["fractions"][frac]["brier_ood"] for r in all_results],
            "boot_clean": [r["fractions"][frac]["boot_clean"] for r in all_results],
            "boot_shifted": [r["fractions"][frac]["boot_shifted"] for r in all_results]
        }
        
    for nl in noise_levels:
        nl_str = f"noise_{nl}"
        final_output[nl_str] = {
            "ece_ood": [r["noise_levels"][nl]["ece_ood"] for r in all_results],
            "brier_ood": [r["noise_levels"][nl]["brier_ood"] for r in all_results],
            "boot_clean": [r["noise_levels"][nl]["boot_clean"] for r in all_results],
            "boot_shifted": [r["noise_levels"][nl]["boot_shifted"] for r in all_results]
        }
        
    os.makedirs("figures", exist_ok=True)
    out_name = "results_c2l_kfold.json" if not SMOKE_TEST else "results_c2l_smoketest.json"
    with open(out_name, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Metrics saved to {out_name}")

if __name__ == "__main__":
    main()
