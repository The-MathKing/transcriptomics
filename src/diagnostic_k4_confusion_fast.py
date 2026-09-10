# src/diagnostic_k4_confusion_fast.py
"""Fast diagnostic script to compute a 4×4 confusion matrix for the coarse‑grained (k=4) deconvolution task.
Trains CondSCVI and DestVI for a reduced number of epochs (30) to finish quickly, then
produces the confusion matrix, applies the Hungarian algorithm, and saves a heat‑map figure.
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
import torch
import pytorch_lightning as pl
import json
import pathlib
import scvi
from collections import OrderedDict
from scvi.model._destvi import _get_loaded_data, _SETUP_ARGS_KEY
from scvi.model import DestVI

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
pl.seed_everything(SEED, workers=True)
scvi.settings.dl_num_workers = 0

# Load single‑cell reference
adata_sc = sc.read_h5ad("data/processed_sc_reference.h5ad")
if adata_sc.n_obs > 5000:
    adata_sc = sc.pp.subsample(adata_sc, n_obs=5000, random_state=SEED, copy=True)

# Identify the cell‑type column (expects 4 coarse classes)
cell_type_col = None
for col in ["cell_class", "cell_subclass", "cluster", "cell_type", "labels"]:
    if col in adata_sc.obs.columns:
        cell_type_col = col
        break
if cell_type_col is None:
    raise ValueError("No cell‑type column found in the reference data")

# Split into train / test (single fold) for reproducibility
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
train_idx, test_idx = next(skf.split(np.zeros(adata_sc.n_obs), adata_sc.obs[cell_type_col].values))
adata_sc_train = adata_sc[train_idx].copy()
adata_sc_test = adata_sc[test_idx].copy()

# Helper to generate pseudo‑spots (copied from benchmark_class)
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

# Train CondSCVI on the single‑cell training pool (few epochs for speed)
scvi.model.CondSCVI.setup_anndata(adata_sc_train, labels_key=cell_type_col)
sc_model = scvi.model.CondSCVI(adata_sc_train, weight_obs=False)
sc_model.train(max_epochs=30, accelerator='cpu', early_stopping=False, train_size=0.9)

# Generate pseudo‑spots for training and testing
n_train_spots = 2000
n_test_spots = 1000
all_cell_types = adata_sc_train.obs[cell_type_col].unique()
adata_train = generate_pseudo_spots(adata_sc_train, all_cell_types, n_spots=n_train_spots, seed=SEED)
adata_test = generate_pseudo_spots(adata_sc_test, all_cell_types, n_spots=n_test_spots, seed=SEED)

# Train DestVI on the training pseudo‑spots (few epochs)
scvi.model.DestVI.setup_anndata(adata_train)
def build_destvi_from_sc_model(sc_model, st_adata, vamp_prior_p=15, **module_kwargs):
    """Manually construct a DestVI model from a pre‑trained CondSCVI model.
    Mirrors the internal logic of DestVI.from_rna_model but avoids the
    problematic handling of `module_kwargs['prior']` in the current scvi‑tools
    version.
    """
    # Load internal data from the CondSCVI model
    attr_dict, var_names, load_state_dict, _ = _get_loaded_data(sc_model)
    registry = attr_dict.pop("registry_")

    # Extract decoder state dictionaries
    decoder_state_dict = OrderedDict(
        (i[8:], load_state_dict[i].float())
        for i in load_state_dict
        if i.split(".")[0] == "decoder"
    )
    px_decoder_state_dict = OrderedDict(
        (i[11:], load_state_dict[i].float())
        for i in load_state_dict
        if i.split(".")[0] == "px_decoder"
    )
    px_r = load_state_dict["px_r"]
    per_ct_bias = load_state_dict["per_ct_bias"]
    mapping = registry["field_registries"]["labels"]["state_registry"]["categorical_mapping"]

    # Dropout rate for the decoder
    dropout_decoder = attr_dict["init_params_"]["non_kwargs"]["dropout_rate"]

    # VAMP prior – request explicitly
    if vamp_prior_p is None:
        mean_vprior = var_vprior = mp_vprior = None
    else:
        mean_vprior, var_vprior, mp_vprior = sc_model.get_vamp_prior(
            sc_model.adata, p=vamp_prior_p
        ).values()
        # Ensure torch tensors for buffers
        if isinstance(mean_vprior, np.ndarray):
            mean_vprior = torch.from_numpy(mean_vprior).float()
        if isinstance(var_vprior, np.ndarray):
            var_vprior = torch.from_numpy(var_vprior).float()
        if isinstance(mp_vprior, np.ndarray):
            mp_vprior = torch.from_numpy(mp_vprior).float()

    # Register the spatial AnnData with the same registry used for the scRNA model
    DestVI.setup_anndata(
        st_adata,
        source_registry=registry,
        extend_categories=True,
        **registry[_SETUP_ARGS_KEY],
    )

    # Instantiate DestVI with all extracted components
    return DestVI(
        st_adata,
        mapping,
        decoder_state_dict,
        px_decoder_state_dict,
        px_r,
        per_ct_bias,
        sc_model.module.n_hidden,
        sc_model.module.n_latent,
        sc_model.module.n_layers,
        mean_vprior=mean_vprior,
        var_vprior=var_vprior,
        mp_vprior=mp_vprior,
        dropout_decoder=dropout_decoder,
        **module_kwargs,
    )

# Train DestVI on the training pseudo‑spots (few epochs)
scvi.model.DestVI.setup_anndata(adata_train)
st_model = build_destvi_from_sc_model(sc_model, adata_train, vamp_prior_p=15)


st_model.train(max_epochs=30, accelerator='cpu', early_stopping=False, train_size=0.9)

# Helper to retrieve predicted proportions from DestVI
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

# Predictions on the test set
true_props_test = adata_test.obsm["proportions"].values
pred_props_test = get_ood_proportions(st_model, adata_test)

# Dominant class indices
y_true = np.argmax(true_props_test, axis=1)
y_pred = np.argmax(pred_props_test, axis=1)

# 4×4 confusion matrix
labels = list(range(len(all_cell_types)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
os.makedirs("figures", exist_ok=True)
np.save("figures/confusion_k4.npy", cm)

# Hungarian optimal assignment
row_ind, col_ind = linear_sum_assignment(-cm)  # maximize total matches
perm_acc = cm[row_ind, col_ind].sum() / cm.sum()
print(f"Hungarian permuted accuracy: {perm_acc:.4f}")

# Confidence statistics
conf_test = np.max(pred_props_test, axis=1)
mean_conf = conf_test.mean()
print(f"Mean confidence (test): {mean_conf:.4f}")

# Expected Calibration Error (ECE) using custom implementation
def compute_ece(confidences, correctness, num_bins=10):
    bins = np.linspace(0, 1, num_bins + 1)
    bin_idxs = np.digitize(confidences, bins) - 1
    ece = 0.0
    for b in range(num_bins):
        mask = bin_idxs == b
        if np.any(mask):
            acc = correctness[mask].mean()
            avg_conf = confidences[mask].mean()
            ece += np.abs(avg_conf - acc) * mask.mean()
    return ece

correctness = (y_pred == y_true).astype(int)
ece_val = compute_ece(conf_test, correctness, num_bins=10)
print(f"Uncalibrated ECE: {ece_val:.4f}")

# Brier score
brier = np.mean(np.sum((pred_props_test - np.eye(len(all_cell_types))[y_true])**2, axis=1))
print(f"Brier score (test): {brier:.4f}")
# Export metrics to JSON
metrics = {
    "accuracy": float(perm_acc),
    "mean_confidence": float(mean_conf),
    "ece_uncalibrated": float(ece_val) if 'ece_val' in locals() else None,
    "brier_score": float(brier)
}
out_path = pathlib.Path("results/k4_metrics.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {out_path}")

# Plot side-by-side heatmaps (raw and Hungarian-aligned)
perm_cm = cm[row_ind][:, col_ind]
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[0])
axs[0].set_title("Raw 4×4 Confusion")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("True")
sns.heatmap(perm_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[1])
axs[1].set_title("Hungarian-aligned")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("True")
plt.tight_layout()
plt.savefig("figures/confusion_k4_prepost.png")
print("Figure saved to figures/confusion_k4_prepost.png")

# Temperature Scaling (single-parameter) fit on training data
try:
    from netcal.scaling import TemperatureScaling
    # Get training predictions
    pred_props_train = get_ood_proportions(st_model, adata_train)
    y_true_train = np.argmax(adata_train.obsm["proportions"].values, axis=1)
    ts = TemperatureScaling()
    ts.fit(pred_props_train, y_true_train)
    pred_ts = ts.predict(pred_props_test)
    conf_ts = np.max(pred_ts, axis=1)
    ece_ts = ece_metric.measure(conf_ts, (np.argmax(pred_ts, axis=1) == y_true).astype(int))
    print(f"Temperature Scaling ECE: {ece_ts:.4f}")
except Exception as e:
    print(f"Temperature Scaling failed: {e}")

# Isotonic Regression on confidence vs correctness (binary)
try:
    from sklearn.isotonic import IsotonicRegression
    correct_train = (np.argmax(pred_props_train, axis=1) == y_true_train).astype(int)
    conf_train = np.max(pred_props_train, axis=1)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(conf_train, correct_train)
    conf_iso = iso.predict(conf_test)
    ece_iso = ece_metric.measure(conf_iso, (y_pred == y_true).astype(int))
    print(f"Isotonic Regression ECE: {ece_iso:.4f}")
except Exception as e:
    print(f"Isotonic Regression failed: {e}")

# Plot heatmap, highlight the optimal matching cells in red
plt.figure(figsize=(6,5))
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
for r, c in zip(row_ind, col_ind):
    ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='red', lw=2))
plt.title(f"4×4 Confusion Matrix (k=4)\nHungarian permuted accuracy: {perm_acc:.1%}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("figures/confusion_k4.png")
print("Figure saved to figures/confusion_k4.png")
