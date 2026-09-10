import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import warnings
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from netcal.metrics import ECE
import scvi
import cell2location
from benchmark_c2l import generate_pseudo_spots, apply_noise_shift, downsample_counts

warnings.filterwarnings('ignore')

def run_alpha_sweep():
    scvi.settings.seed = 42
    adata_sc = sc.read_h5ad("data/sc_mouse_cortex.h5ad")
    adata_sc.X = np.round(adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else adata_sc.X).astype(int)
    adata_sc = adata_sc[np.random.choice(adata_sc.n_obs, 500, replace=False)].copy()
    
    cell_types = adata_sc.obs["cell_class"].unique()
    
    # Reference model
    scvi.model.CondSCVI.setup_anndata(adata_sc, labels_key="cell_class")
    cell2location.models.RegressionModel.setup_anndata(adata_sc, labels_key="cell_class")
    ref_mod = cell2location.models.RegressionModel(adata_sc)
    ref_mod.train(max_epochs=20, accelerator="cpu")
    
    adata_sc.varm["means_per_cluster"] = ref_mod.samples["post_sample_means"]["w_sf"]
    
    # Pseudo-spots
    adata_ps_cal = generate_pseudo_spots(adata_sc, cell_types, n_spots=200, cells_per_spot=10, cell_type_col="cell_class", seed=42)
    adata_ps_test_clean = generate_pseudo_spots(adata_sc, cell_types, n_spots=200, cells_per_spot=10, cell_type_col="cell_class", seed=100)
    adata_ps_test_shift = downsample_counts(adata_ps_test_clean, 0.2)
    
    t_cal = adata_ps_cal.obsm["proportions"].values
    t_clean = adata_ps_test_clean.obsm["proportions"].values
    
    alphas = [0.2, 2.0, 20.0, 200.0]
    results = {}
    
    ece_metric = ECE(bins=10)
    
    print("Running detection_alpha sensitivity sweep on cell2location...")
    
    for alpha in alphas:
        print(f"\n--- Testing detection_alpha = {alpha} ---")
        
        def get_preds(adata):
            cell2location.models.Cell2location.setup_anndata(adata)
            mod = cell2location.models.Cell2location(
                adata, cell_state_df=adata_sc.varm["means_per_cluster"],
                N_cells_per_location=10, detection_alpha=alpha
            )
            mod.train(max_epochs=20, accelerator="cpu")
            return mod.samples["post_sample_means"]["w_sf"].values
            
        p_cal = get_preds(adata_ps_cal)
        p_clean = get_preds(adata_ps_test_clean)
        p_shift = get_preds(adata_ps_test_shift)
        
        p_cal = p_cal / (p_cal.sum(axis=1, keepdims=True) + 1e-9)
        p_clean = p_clean / (p_clean.sum(axis=1, keepdims=True) + 1e-9)
        p_shift = p_shift / (p_shift.sum(axis=1, keepdims=True) + 1e-9)
        
        conf_cal = np.max(p_cal, axis=1)
        acc_cal = (np.argmax(p_cal, axis=1) == np.argmax(t_cal, axis=1)).astype(int)
        
        conf_clean = np.max(p_clean, axis=1)
        acc_clean = (np.argmax(p_clean, axis=1) == np.argmax(t_clean, axis=1)).astype(int)
        
        conf_shift = np.max(p_shift, axis=1)
        acc_shift = (np.argmax(p_shift, axis=1) == np.argmax(t_clean, axis=1)).astype(int)
        
        # Fit Isotonic
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(conf_cal, acc_cal)
        iso_clean = iso.predict(conf_clean)
        iso_shift = iso.predict(conf_shift)
        
        # Fit TS
        t_vals = np.linspace(0.05, 50.0, 200)
        nlls = []
        for temp in t_vals:
            eps = 1e-7
            logits = np.log(p_cal + eps)
            scaled = logits / temp
            exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            cal_p = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            nll = -np.mean(np.sum(t_cal * np.log(cal_p + eps), axis=1))
            nlls.append(nll)
            
        t_opt = t_vals[np.argmin(nlls)]
        
        # Apply TS
        def apply_ts(p, temp):
            eps = 1e-7
            logits = np.log(p + eps)
            scaled = logits / temp
            exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            return exp_l / np.sum(exp_l, axis=1, keepdims=True)
            
        p_ts_clean = apply_ts(p_clean, t_opt)
        p_ts_shift = apply_ts(p_shift, t_opt)
        
        try:
            auroc_clean = float(roc_auc_score(acc_clean, conf_clean))
            auroc_shift = float(roc_auc_score(acc_shift, conf_shift))
        except:
            auroc_clean, auroc_shift = np.nan, np.nan
            
        res = {
            "optimal_T": float(t_opt),
            "uncal_ece_clean": float(ece_metric.measure(conf_clean, acc_clean)),
            "uncal_ece_shift": float(ece_metric.measure(conf_shift, acc_shift)),
            "ts_ece_clean": float(ece_metric.measure(np.max(p_ts_clean, axis=1), acc_clean)),
            "ts_ece_shift": float(ece_metric.measure(np.max(p_ts_shift, axis=1), acc_shift)),
            "iso_ece_clean": float(ece_metric.measure(iso_clean, acc_clean)),
            "iso_ece_shift": float(ece_metric.measure(iso_shift, acc_shift)),
            "auroc_clean": auroc_clean,
            "auroc_shift": auroc_shift
        }
        results[str(alpha)] = res
        print(f"Alpha={alpha}: T*={t_opt:.2f}, Clean ECE: Uncal={res['uncal_ece_clean']:.3f}, TS={res['ts_ece_clean']:.3f}, Iso={res['iso_ece_clean']:.3f} | Shift ECE: Uncal={res['uncal_ece_shift']:.3f}, TS={res['ts_ece_shift']:.3f}, Iso={res['iso_ece_shift']:.3f}")
        
    with open("results_c2l_alpha_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to results_c2l_alpha_sensitivity.json")

if __name__ == "__main__":
    run_alpha_sweep()
