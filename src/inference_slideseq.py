import scanpy as sc
import scvi
import numpy as np
import pandas as pd
from netcal.metrics import ECE
import torch
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt

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
    pred_props_df = st_model.get_proportions()
    
    true_props = gt_mapped[eval_cell_types].values
    pred_props = pred_props_df[eval_cell_types].values
    
    pred_props = pred_props / pred_props.sum(axis=1, keepdims=True)
    
    conf_raw = np.max(pred_props, axis=1)
    acc_raw = (np.argmax(pred_props, axis=1) == np.argmax(true_props, axis=1)).astype(int)
    
    ece_raw = ECE(bins=10).measure(conf_raw, acc_raw)
    print(f"OOD ECE (Uncalibrated): {ece_raw:.4f}")
    
    # Temperature Scaling
    best_t = 2.0
    pred_props_cal = apply_temperature_scaling(pred_props, temp=best_t)
    conf_cal = np.max(pred_props_cal, axis=1)
    acc_cal = (np.argmax(pred_props_cal, axis=1) == np.argmax(true_props, axis=1)).astype(int)
    ece_cal = ECE(bins=10).measure(conf_cal, acc_cal)
    print(f"OOD ECE (Temp Scaled T={best_t}): {ece_cal:.4f}")
    
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(conf_raw, bins=50, alpha=0.6, label='Uncalibrated')
    plt.hist(conf_cal, bins=50, alpha=0.6, label=f'Temperature Scaled (T={best_t})')
    plt.xlabel("Max Predicted Cell Type Proportion (Confidence)")
    plt.ylabel("Number of Spots")
    plt.title("Slide-seqV2 Calibration (Matched Hippocampus Ref)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("figures/slideseq_hippocampus_calibration.png", dpi=300)
    print("Saved figure to figures/slideseq_hippocampus_calibration.png")

if __name__ == "__main__":
    main()
