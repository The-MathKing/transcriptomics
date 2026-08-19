import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import os

def main():
    print("Loading datasets...")
    adata_sc = sc.read_h5ad("data/sc_mouse_cortex.h5ad")
    adata_visium = sc.read_h5ad("data/visium_mouse_brain.h5ad")
    adata_slideseq = sc.read_h5ad("data/slideseqv2_mouse_hippocampus.h5ad")
    
    print("Preprocessing datasets...")
    # 1. Filter out genes expressed in very few cells/spots
    sc.pp.filter_genes(adata_sc, min_cells=10)
    sc.pp.filter_genes(adata_visium, min_cells=10)
    sc.pp.filter_genes(adata_slideseq, min_cells=10)
    
    # 2. Intersect genes across all three datasets
    common_genes = set(adata_sc.var_names) & set(adata_visium.var_names) & set(adata_slideseq.var_names)
    common_genes = list(common_genes)
    print(f"Number of common genes after intersection: {len(common_genes)}")
    
    adata_sc = adata_sc[:, common_genes].copy()
    adata_visium = adata_visium[:, common_genes].copy()
    adata_slideseq = adata_slideseq[:, common_genes].copy()
    
    # DestVI uses raw counts. Ensure X contains raw counts.
    # We will save them processed.
    
    print("Generating pseudo-spots from scRNA-seq for ground truth ECE evaluation...")
    # Generate 2000 pseudo-spots by sampling 10 cells per spot
    n_spots = 2000
    cells_per_spot = 10
    
    # Cell types column in sc_mouse_cortex is typically 'cluster' or 'cell_subclass'
    # Let's inspect the obs columns to find the cell type annotation
    cell_type_col = None
    for col in ["cell_subclass", "cluster", "cell_type", "labels"]:
        if col in adata_sc.obs.columns:
            cell_type_col = col
            break
            
    if cell_type_col is None:
        # Fallback if specific column isn't found
        cell_type_col = adata_sc.obs.columns[0]
        
    print(f"Using '{cell_type_col}' for cell type annotations.")
    
    cell_types = adata_sc.obs[cell_type_col].unique()
    pseudo_counts = np.zeros((n_spots, len(common_genes)))
    pseudo_props = np.zeros((n_spots, len(cell_types)))
    
    type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    np.random.seed(42)
    for i in range(n_spots):
        # sample cells
        sampled_indices = np.random.choice(adata_sc.n_obs, size=cells_per_spot, replace=True)
        sampled_cells = adata_sc[sampled_indices]
        
        # aggregate counts
        if hasattr(sampled_cells.X, "toarray"):
            pseudo_counts[i, :] = sampled_cells.X.sum(axis=0).A1
        else:
            pseudo_counts[i, :] = sampled_cells.X.sum(axis=0)
            
        # compute proportions
        types = sampled_cells.obs[cell_type_col].values
        for ct in types:
            pseudo_props[i, type_to_idx[ct]] += 1
    
    pseudo_props = pseudo_props / cells_per_spot
    
    adata_pseudo = ad.AnnData(X=pseudo_counts)
    adata_pseudo.var_names = common_genes
    adata_pseudo.obs_names = [f"pseudo_{i}" for i in range(n_spots)]
    
    # Save proportions to obsm
    prop_df = pd.DataFrame(pseudo_props, index=adata_pseudo.obs_names, columns=cell_types)
    adata_pseudo.obsm["proportions"] = prop_df
    
    print("Saving preprocessed datasets...")
    adata_sc.write("data/processed_sc_reference.h5ad")
    adata_visium.write("data/processed_visium.h5ad")
    adata_slideseq.write("data/processed_slideseqv2.h5ad")
    adata_pseudo.write("data/processed_pseudospots.h5ad")
    
    print("Preprocessing Phase Complete!")

if __name__ == "__main__":
    main()
