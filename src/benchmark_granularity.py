import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import scvi
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

from src.benchmark_v4 import generate_pseudo_spots, get_ood_proportions, get_calibration_stats

def run_granularity_test():
    adata_sc = sc.read_h5ad("data/sc_mouse_cortex.h5ad")
    
    results = {}
    
    # We will test cell_class (coarse) and cell_subclass (fine)
    for cell_type_col in ['cell_class', 'cell_subclass']:
        print(f"\n=== Testing granularity: {cell_type_col} ===")
        
        # We just need 1 fold for this demonstration
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        y_strat = adata_sc.obs[cell_type_col].values
        train_idx, test_idx = next(skf.split(np.zeros(len(y_strat)), y_strat))
        
        seed = 42
        scvi.settings.seed = seed
        pl.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        adata_sc_train = adata_sc[train_idx].copy()
        adata_sc_test = adata_sc[test_idx].copy()
        
        all_cell_types = adata_sc_train.obs[cell_type_col].unique()
        
        scvi.model.CondSCVI.setup_anndata(adata_sc_train, labels_key=cell_type_col)
        sc_model = scvi.model.CondSCVI(adata_sc_train, weight_obs=False)
        sc_model.train(max_epochs=100, accelerator='cpu', early_stopping=True, train_size=0.9)
        
        adata_st_test = generate_pseudo_spots(adata_sc_test, all_cell_types, n_spots=500, cells_per_spot=10, cell_type_col=cell_type_col, seed=seed)
        adata_st_train_cal = generate_pseudo_spots(adata_sc_train, all_cell_types, n_spots=1000, cells_per_spot=10, cell_type_col=cell_type_col, seed=seed)
        
        rng = np.random.RandomState(seed)
        perm = rng.permutation(1000)
        train_spot_idx = perm[:700]
        
        adata_train = adata_st_train_cal[train_spot_idx].copy()
        
        scvi.model.DestVI.setup_anndata(adata_train)
        st_model = scvi.model.DestVI.from_rna_model(adata_train, sc_model)
        st_model.train(max_epochs=100, accelerator='cpu', early_stopping=True, train_size=0.9)
        
        # Test baseline clean performance
        true_props_test = adata_st_test.obsm["proportions"].values
        pred_props_test = get_ood_proportions(st_model, adata_st_test)
        
        _, acc, ece, _, _, _, _, _ = get_calibration_stats(true_props_test, pred_props_test)
        
        print(f"Granularity {cell_type_col} ({len(all_cell_types)} classes): Baseline ECE = {np.mean(ece):.4f}")
        results[cell_type_col] = float(np.mean(ece))
        
    print("\n=== Granularity Experiment Results ===")
    for k, v in results.items():
        print(f"{k}: ECE = {v:.4f}")

if __name__ == "__main__":
    run_granularity_test()
