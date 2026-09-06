import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class BetaCalibration:
    def __init__(self):
        self.lr = LogisticRegression(C=999, solver='lbfgs')
    def fit(self, conf_cal, acc_cal):
        eps = 1e-7
        log_odds = np.log(conf_cal + eps) - np.log(1 - conf_cal + eps)
        try:
            self.lr.fit(log_odds.reshape(-1, 1), acc_cal)
            self.fitted = True
        except ValueError:
            self.fitted = False
            
    def predict(self, conf):
        if not getattr(self, 'fitted', False):
            return conf
        eps = 1e-7
        log_odds = np.log(conf + eps) - np.log(1 - conf + eps)
        return self.lr.predict_proba(log_odds.reshape(-1, 1))[:, 1]
import torch
import pytorch_lightning as pl
import os

def apply_temperature_scaling(pred_p, temp=1.0):
    eps = 1e-7
    logits = np.log(pred_p + eps)
    scaled_logits = logits / temp
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    cal_pred_p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return cal_pred_p

def main():
    print("Starting Visium->Visium Mouse Cortex Inference...")
    seed = 42
    scvi.settings.seed = seed
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    
    # 1. Load data
    sc_path = "data/processed_sc_reference.h5ad"
    st_path = "data/processed_visium.h5ad"
    
    adata_sc = sc.read_h5ad(sc_path)
    adata_st = sc.read_h5ad(st_path)
    
    # Preprocessing (ensure common genes)
    common_genes = np.intersect1d(adata_sc.var_names, adata_st.var_names)
    adata_sc = adata_sc[:, common_genes].copy()
    adata_st = adata_st[:, common_genes].copy()
    
    # 2. Train CondSCVI
    print("Training CondSCVI...")
    cell_type_col = None
    for col in ["cell_subclass", "cluster", "cell_type", "labels"]:
        if col in adata_sc.obs.columns:
            cell_type_col = col
            break
            
    scvi.model.CondSCVI.setup_anndata(adata_sc, labels_key=cell_type_col)
    sc_model = scvi.model.CondSCVI(adata_sc, weight_obs=False)
    sc_model.train(max_epochs=200, accelerator='cpu', early_stopping=True, train_size=0.9)
    
    # 3. Train DestVI
    print("Training DestVI...")
    scvi.model.DestVI.setup_anndata(adata_st)
    st_model = scvi.model.DestVI.from_rna_model(adata_st, sc_model)
    st_model.train(max_epochs=200, accelerator='cpu', early_stopping=True, train_size=0.9)
    
    # 4. Extract Proportions (Uncalibrated)
    print("Extracting Proportions...")
    pred_props_ood = st_model.get_proportions().values
    conf_ood = np.max(pred_props_ood, axis=1)
    
    # 5. Apply Temperature Scaling
    best_t = 2.0 
    pred_props_cal = apply_temperature_scaling(pred_props_ood, temp=best_t)
    conf_ts = np.max(pred_props_cal, axis=1)

    # 5.b Train Isotonic and Beta on a pseudo-calibration set (simplified here by applying to TS output)
    # Note: For real tissue we lack ground truth, so we apply the functions fitted from the pseudo-spot script.
    # To simulate this rigorously in the plot, we load the calibration from a fast synthetic draw.
    print("Generating calibration pseudo-spots...")
    from src.benchmark_v4 import generate_pseudo_spots
    all_cell_types = adata_sc.obs["cell_class"].unique()
    adata_cal = generate_pseudo_spots(adata_sc, all_cell_types, n_spots=500, cells_per_spot=10, cell_type_col="cell_class", seed=42)
    
    scvi.model.DestVI.setup_anndata(adata_cal)
    st_model_cal = scvi.model.DestVI.from_rna_model(adata_cal, sc_model)
    st_model_cal.train(max_epochs=10, accelerator='cpu')
    
    true_props_cal = adata_cal.obsm["proportions"].values
    pred_props_cal_syn = st_model_cal.get_proportions().values
    conf_syn = np.max(pred_props_cal_syn, axis=1)
    acc_syn = (np.argmax(pred_props_cal_syn, axis=1) == np.argmax(true_props_cal, axis=1)).astype(int)
    
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(conf_syn, acc_syn)
    conf_iso = iso.predict(conf_ood)
    
    beta = BetaCalibration()
    beta.fit(conf_syn, acc_syn)
    conf_beta = beta.predict(conf_ood)
    
    # 6. Plotting the Confidence Distribution
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(conf_ood, bins=50, alpha=0.5, label='Uncalibrated')
    plt.hist(conf_ts, bins=50, alpha=0.5, label=f'Temperature Scaled (T={best_t})')
    plt.xlabel("Max Predicted Cell Type Proportion (Confidence)")
    plt.ylabel("Number of Spots")
    plt.title("Calibration Effect on Real Visium Mouse Cortex Data")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("figures/real_cortex_confidence_dist.png", dpi=300)
    print("Saved figure to figures/real_cortex_confidence_dist.png")

if __name__ == "__main__":
    main()
