import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from netcal.metrics import ECE
import scvi
import cell2location
from benchmark_c2l import generate_pseudo_spots, apply_noise_shift, downsample_counts
import warnings
warnings.filterwarnings('ignore')

def main():
    scvi.settings.seed = 42
    adata_sc = sc.read_h5ad("data/sc_mouse_cortex.h5ad")
    adata_sc.X = np.round(adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else adata_sc.X).astype(int)
    adata_sc = adata_sc[np.random.choice(adata_sc.n_obs, 500, replace=False)].copy()
    
    cell_types = adata_sc.obs["cell_class"].unique()
    
    # Quick Reference
    scvi.model.CondSCVI.setup_anndata(adata_sc, labels_key="cell_class")
    cell2location.models.RegressionModel.setup_anndata(adata_sc, labels_key="cell_class")
    ref_mod = cell2location.models.RegressionModel(adata_sc)
    ref_mod.train(max_epochs=20, accelerator="cpu")
    
    adata_sc.varm["means_per_cluster"] = ref_mod.samples["post_sample_means"]["w_sf"]
    
    # Pseudo spots
    adata_ps_cal = generate_pseudo_spots(adata_sc, cell_types, n_spots=200, cells_per_spot=10, cell_type_col="cell_class")
    adata_ps_test_clean = generate_pseudo_spots(adata_sc, cell_types, n_spots=200, cells_per_spot=10, cell_type_col="cell_class", seed=100)
    adata_ps_test_shift = downsample_counts(adata_ps_test_clean, 0.2)
    
    def get_preds(adata):
        cell2location.models.Cell2location.setup_anndata(adata)
        mod = cell2location.models.Cell2location(
            adata, cell_state_df=adata_sc.varm["means_per_cluster"],
            N_cells_per_location=10, detection_alpha=20
        )
        mod.train(max_epochs=20, accelerator="cpu")
        return mod.samples["post_sample_means"]["w_sf"].values
        
    p_cal = get_preds(adata_ps_cal)
    p_clean = get_preds(adata_ps_test_clean)
    p_shift = get_preds(adata_ps_test_shift)
    
    p_cal = p_cal / p_cal.sum(axis=1, keepdims=True)
    p_clean = p_clean / p_clean.sum(axis=1, keepdims=True)
    p_shift = p_shift / p_shift.sum(axis=1, keepdims=True)
    
    t_cal = adata_ps_cal.obsm["proportions"].values
    t_clean = adata_ps_test_clean.obsm["proportions"].values
    
    conf_cal = np.max(p_cal, axis=1)
    acc_cal = (np.argmax(p_cal, axis=1) == np.argmax(t_cal, axis=1)).astype(int)
    
    conf_clean = np.max(p_clean, axis=1)
    acc_clean = (np.argmax(p_clean, axis=1) == np.argmax(t_clean, axis=1)).astype(int)
    
    conf_shift = np.max(p_shift, axis=1)
    acc_shift = (np.argmax(p_shift, axis=1) == np.argmax(t_clean, axis=1)).astype(int)
    
    # AUROC
    try:
        auroc_clean = roc_auc_score(acc_clean, conf_clean)
        auroc_shift = roc_auc_score(acc_shift, conf_shift)
    except:
        auroc_clean, auroc_shift = np.nan, np.nan
        
    print(f"AUROC Clean: {auroc_clean:.4f}")
    print(f"AUROC Shift: {auroc_shift:.4f}")
    
    # Fit Isotonic
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(conf_cal, acc_cal)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    x = np.linspace(0.2, 1.0, 100)
    plt.plot(x, iso.predict(x), 'b-', label="Isotonic Map")
    plt.scatter(conf_cal, acc_cal, color='gray', alpha=0.1)
    plt.xlabel("Uncalibrated Confidence")
    plt.ylabel("Calibrated Confidence")
    plt.title("cell2location Isotonic Fit")
    
    plt.subplot(1, 2, 2)
    t_vals = np.linspace(0.5, 3.0, 50)
    eces = []
    for temp in t_vals:
        eps = 1e-7
        logits = np.log(p_cal + eps)
        scaled = logits / temp
        exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        cal_p = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        eces.append(ECE(bins=10).measure(np.max(cal_p, axis=1), acc_cal))
        
    plt.plot(t_vals, eces, 'r-')
    plt.xlabel("Temperature T")
    plt.ylabel("ECE (Calibration Split)")
    plt.title("Temperature Scaling Search")
    
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/c2l_mechanism.png")
    print("Saved figures/c2l_mechanism.png")

if __name__ == "__main__":
    main()
