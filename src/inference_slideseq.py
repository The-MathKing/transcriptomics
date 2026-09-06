import scanpy as sc
import scvi
import numpy as np
import pandas as pd
from netcal.metrics import ECE
import torch
import pytorch_lightning as pl
import os
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

def apply_temperature_scaling(pred_p, temp=1.0):
    eps = 1e-7
    logits = np.log(pred_p + eps)
    scaled_logits = logits / temp
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    cal_pred_p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return cal_pred_p

def main():
    print("Starting Slide-seqV2 inference with matched Hippocampus reference...")
    seed = 42
    scvi.settings.seed = seed
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    
    # 1. Load DropViz single-cell reference (Hippocampus)
    print("Loading DropViz Hippocampus SC reference...")
    adata_sc = sc.read_h5ad("data/hippo_reference.h5ad")
    
    # 2. Load spatial data (Slide-seqV2 Hippocampus)
    print("Loading Slide-seqV2 spatial data...")
    adata_st = sc.read_h5ad("data/processed_slideseqv2.h5ad")
    
    # 3. Cell Type Alignment (Same-Region Synonym Mapping)
    # Map from DropViz 'celltype' to Slide-seqV2 GT columns
    hippo_map = {
        'Dentate Principal cells': 'DentatePyramids',
        'CA1 Principal cells': 'CA1_CA2_CA3_Subiculum',
        'CA2 Principal cells': 'CA1_CA2_CA3_Subiculum',
        'CA3 Principal cells': 'CA1_CA2_CA3_Subiculum',
        'Subiculum': 'CA1_CA2_CA3_Subiculum',
        'Entorhinal cortex': 'Subiculum_Entorhinal',
        'Interneuron': 'Interneurons',
        'Astrocyte': 'Astrocytes',
        'Oligodendrocyte': 'Oligodendrocytes',
        'Polydendrocyte_1': 'Polydendrocytes',
        'Polydendrocyte_2': 'Polydendrocytes',
        'Endothelial stalk': 'Endothelial_Stalk',
        'Endothelial tip': 'Endothelial_Tip',
        'Mural': 'Mural',
        'Microglia': 'Microglia',
        'Ependymal': 'Ependymal',
        'Neurogenesis (SGZ)': 'Neurogenesis'
    }
    
    # Filter reference and apply mapping
    adata_sc.obs['Mapped_Cell_Type'] = adata_sc.obs['celltype'].map(hippo_map)
    adata_sc = adata_sc[~adata_sc.obs['Mapped_Cell_Type'].isna()].copy()
    
    # Map Spatial GT
    gt_df = adata_st.obsm['deconvolution_results'].copy()
    
    # Combine the two Subiculum_Entorhinal clusters into one to match the 'Entorhinal cortex' from DropViz
    if 'Subiculum_Entorhinal_cl2' in gt_df.columns and 'Subiculum_Entorhinal_cl3' in gt_df.columns:
        gt_df['Subiculum_Entorhinal'] = gt_df['Subiculum_Entorhinal_cl2'] + gt_df['Subiculum_Entorhinal_cl3']
    
    eval_cell_types = sorted(list(set(hippo_map.values())))
    
    # Extract only the mapped columns
    gt_mapped = gt_df[eval_cell_types].copy()
    
    # Re-normalize spatial GT to sum to 1 over the mapped cell types
    gt_mapped = gt_mapped.div(gt_mapped.sum(axis=1), axis=0)
    gt_mapped = gt_mapped.fillna(0) # For spots with 0% mapped cells
    
    # Keep only spots that had at least some of the mapped cell types
    valid_spots = gt_mapped.sum(axis=1) > 0.5 # require majority of spot to be mapped types
    adata_st = adata_st[valid_spots].copy()
    gt_mapped = gt_mapped.loc[valid_spots]
    adata_st.obsm['proportions_mapped'] = gt_mapped
    
    # Make sure common genes (handle case mismatch between DropViz and Slide-seq)
    adata_sc.var_names = adata_sc.var_names.str.upper()
    adata_st.var_names = adata_st.var_names.str.upper()
    
    # Optional: some genes might have become duplicates after upper(), let's make them unique
    adata_sc.var_names_make_unique()
    adata_st.var_names_make_unique()
    
    common_genes = np.intersect1d(adata_sc.var_names, adata_st.var_names)
    adata_sc = adata_sc[:, common_genes].copy()
    adata_st = adata_st[:, common_genes].copy()
    
    print(f"Using {len(common_genes)} common genes across {len(eval_cell_types)} cell classes.")
    
    # 4. Train CondSCVI
    print("Training CondSCVI...")
    scvi.model.CondSCVI.setup_anndata(adata_sc, labels_key='Mapped_Cell_Type')
    sc_model = scvi.model.CondSCVI(adata_sc, weight_obs=False)
    sc_model.train(max_epochs=100, accelerator='cpu', early_stopping=True, train_size=0.9)
    
    # 5. Train DestVI
    print("Training DestVI on Slide-seqV2...")
    scvi.model.DestVI.setup_anndata(adata_st)
    st_model = scvi.model.DestVI.from_rna_model(adata_st, sc_model)
    st_model.train(max_epochs=100, accelerator='cpu', early_stopping=True, train_size=0.9)
    
    # 6. Evaluation
    pred_props = st_model.get_proportions()[eval_cell_types].values
    pred_props = pred_props / pred_props.sum(axis=1, keepdims=True)
    conf_ood = np.max(pred_props, axis=1)

    best_t = 1.8
    pred_props_cal = apply_temperature_scaling(pred_props, temp=best_t)
    conf_ts = np.max(pred_props_cal, axis=1)
    
    print("Generating calibration pseudo-spots...")
    from src.benchmark_v4 import generate_pseudo_spots
    all_cell_types = adata_sc.obs["cluster"].unique()
    adata_cal = generate_pseudo_spots(adata_sc, all_cell_types, n_spots=500, cells_per_spot=10, cell_type_col="cluster", seed=42)
    
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
    plt.hist(conf_iso, bins=50, alpha=0.5, label='Isotonic Regression')
    plt.hist(conf_beta, bins=50, alpha=0.5, label='Beta Calibration')
    plt.xlabel("Max Predicted Cell Type Proportion (Confidence)")
    plt.ylabel("Number of Spots")
    plt.title("Calibration Effect on Real Slide-seqV2 Mouse Hippocampus Data")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("figures/slideseq_hippocampus_calibration.png", dpi=300)
    print("Saved figure to figures/slideseq_hippocampus_calibration.png")

if __name__ == "__main__":
    main()
