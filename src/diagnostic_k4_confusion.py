# src/diagnostic_k4_confusion.py
"""Compute 4×4 confusion matrix for k=4 coarse-grained deconvolution.
Generates a heatmap figure and saves the raw matrix.
"""
import os
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from netcal.metrics import ECE
from sklearn.isotonic import IsotonicRegression
import torch
import pytorch_lightning as pl
import scvi

# reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
pl.seed_everything(SEED, workers=True)

# Load single-cell reference
adata_sc = sc.read_h5ad("data/processed_sc_reference.h5ad")

# Determine coarse cell type column (expects 4 classes)
cell_type_col = None
for col in ["cell_class", "cell_subclass", "cluster", "cell_type", "labels"]:
    if col in adata_sc.obs.columns:
        cell_type_col = col
        break
if cell_type_col is None:
    raise ValueError("No cell type column found in reference data")

# Split into train/test for reproducibility (single fold)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
train_idx, test_idx = next(skf.split(np.zeros(adata_sc.n_obs), adata_sc.obs[cell_type_col].values))
adata_sc_train = adata_sc[train_idx].copy()
adata_sc_test = adata_sc[test_idx].copy()

# Helper to generate pseudo spots (copied from benchmark_class)

def generate_pseudo_spots(adata_sc, all_cell_types, n_spots=2000, cells_per_spot=10, seed=SEED):
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
    sanitized = [str(ct).replace("/", "_") for ct in all_cell_types]
    prop_df = pd.DataFrame(pseudo_props, index=adata_pseudo.obs_names, columns=sanitized)
    adata_pseudo.obsm["proportions"] = prop_df
    return adata_pseudo

# Train CondSCVI on single-cell training pool
scvi.model.CondSCVI.setup_anndata(adata_sc_train, labels_key=cell_type_col)
sc_model = scvi.model.CondSCVI(adata_sc_train, weight_obs=False)
sc_model.train(max_epochs=200, accelerator='cpu', early_stopping=True, train_size=0.9)

# Generate pseudo spots for training and test
n_train_spots = 2000
n_test_spots = 1000
all_cell_types = adata_sc_train.obs[cell_type_col].unique()
adata_train = generate_pseudo_spots(adata_sc_train, all_cell_types, n_spots=n_train_spots, seed=SEED)
adata_test = generate_pseudo_spots(adata_sc_test, all_cell_types, n_spots=n_test_spots, seed=SEED)

# Train DestVI on training pseudo spots
scvi.model.DestVI.setup_anndata(adata_train)
st_model = scvi.model.DestVI.from_rna_model(adata_train, sc_model)
st_model.train(max_epochs=200, accelerator='cpu', early_stopping=True, train_size=0.9)

# Helper to get predicted proportions from DestVI
def get_ood_proportions(model, adata):
    if "_indices" not in adata.obs:
        adata.obs["_indices"] = np.arange(adata.n_obs)
    model.transfer_fields(adata)
    old = model.adata
    model.adata = adata
    try:
        props_df = model.get_proportions()
        props = props_df[adata.obsm["proportions"].columns].values
    finally:
        model.adata = old
    return props

# Predictions on test set
true_props_test = adata_test.obsm["proportions"].values
pred_props_test = get_ood_proportions(st_model, adata_test)

# True and predicted dominant class indices
y_true = np.argmax(true_props_test, axis=1)
y_pred = np.argmax(pred_props_test, axis=1)

# Compute confusion matrix (4x4)
labels = list(range(len(all_cell_types)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
# Save raw matrix
os.makedirs("figures", exist_ok=True)
np.save("figures/confusion_k4.npy", cm)

# Hungarian algorithm for optimal permutation
row_ind, col_ind = linear_sum_assignment(-cm)  # maximize sum
perm_accuracy = cm[row_ind, col_ind].sum() / cm.sum()
print(f"Hungarian permuted accuracy: {perm_accuracy:.4f}")

# Plot heatmap with permutation highlighted
plt.figure(figsize=(6,5))
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
for r, c in zip(row_ind, col_ind):
    ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='red', lw=2))
plt.title("4×4 Confusion Matrix (k=4)\nHungarian permuted accuracy: {:.1%}".format(perm_accuracy))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("figures/confusion_k4.png")
print("Figure saved to figures/confusion_k4.png")
