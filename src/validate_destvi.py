import scvi
import numpy as np
import pandas as pd
import anndata

print("Testing DestVI setup...")

# 1. Create synthetic single-cell reference data
adata_sc = anndata.AnnData(X=np.random.poisson(2.0, size=(400, 100)).astype(np.float32))
adata_sc.obs["cell_type"] = np.random.choice(["A", "B", "C"], size=400).astype(str)
adata_sc.var_names = [f"gene_{i}" for i in range(100)]
adata_sc.obs_names = [f"cell_{i}" for i in range(400)]

# 2. Create synthetic spatial data
adata_st = anndata.AnnData(X=np.random.poisson(2.0, size=(100, 100)).astype(np.float32))
adata_st.var_names = adata_sc.var_names
adata_st.obs_names = [f"spot_{i}" for i in range(100)]

# 3. Train CondSCVI on reference
print("Setting up CondSCVI...")
scvi.model.CondSCVI.setup_anndata(adata_sc, labels_key="cell_type")
sc_model = scvi.model.CondSCVI(adata_sc, weight_obs=False)
print("Training CondSCVI (10 epochs for quick test)...")
sc_model.train(max_epochs=10)

# 4. Train DestVI on spatial data
print("Setting up DestVI...")
scvi.model.DestVI.setup_anndata(adata_st)
st_model = scvi.model.DestVI.from_rna_model(adata_st, sc_model)
print("Training DestVI (10 epochs for quick test)...")
st_model.train(max_epochs=10)

# 5. Extract uncertainty (posterior variance of proportions)
print("Extracting proportions and uncertainty...")
# We can sample from the posterior to get mean and variance
# In DestVI, get_proportions returns the mean by default.
# We can sample by using the underlying module, but DestVI's get_proportions
# can be run multiple times if we enable dropout, or we can use get_scale_for_ct
# Let's check if we can get variance easily.
try:
    props, props_var = st_model.get_proportions(return_variance=True)
    print("Posterior proportions shape:", props.shape)
    print("Posterior variance shape:", props_var.shape)
    print("Uncertainty extraction successful!")
except TypeError:
    print("DestVI.get_proportions does not support return_variance=True directly.")
    print("We will implement Monte Carlo sampling from the posterior...")
    # Manual Monte Carlo sampling of proportions
    # For a robust benchmark, we need to extract variance.
    # The latent representation of proportions is st_model.module.get_proportions()
    print("Fallback: extracting uncertainty via MC samples (TODO if needed for full run)")

print("DestVI validation script complete.")
