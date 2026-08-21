import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netcal.metrics import ECE
from sklearn.isotonic import IsotonicRegression
from scipy.stats import ttest_rel
import os
import json
import torch
import pytorch_lightning as pl

def downsample_counts(adata, fraction=0.2):
    # Simulate lower capture rate via binomial dropout
    new_adata = adata.copy()
    if hasattr(new_adata.X, "toarray"):
        counts = new_adata.X.toarray()
    else:
        counts = new_adata.X.copy()
    
    counts = counts.astype(int)
    downsampled = np.random.binomial(counts, fraction)
    new_adata.X = downsampled.astype(np.float32)
    return new_adata

def get_calibration_stats(true_p, pred_p, temp=1.0, iso_reg=None):
    eps = 1e-7
    logits = np.log(pred_p + eps)
    scaled_logits = logits / temp
    
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    cal_pred_p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    conf = np.max(cal_pred_p, axis=1)
    acc = (np.argmax(cal_pred_p, axis=1) == np.argmax(true_p, axis=1)).astype(int)
    
    if iso_reg is not None:
        conf = iso_reg.predict(conf)
        conf = np.clip(conf, 0, 1)
        
    ece = ECE(bins=10).measure(conf, acc)
    return conf, acc, ece, cal_pred_p

def optimize_temperature(true_p, pred_p):
    best_t = 1.0
    best_ece = float('inf')
    for t in np.linspace(0.5, 3.0, 50):
        _, _, ece, _ = get_calibration_stats(true_p, pred_p, temp=t)
        if ece < best_ece:
            best_ece = ece
            best_t = t
    return best_t, best_ece

def run_seed_pipeline(adata_sc, adata_st, seed, fractions):
    scvi.settings.seed = seed
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(seed)
    
    cell_type_col = None
    for col in ["cell_subclass", "cluster", "cell_type", "labels"]:
        if col in adata_sc.obs.columns:
            cell_type_col = col
            break
            
    scvi.model.CondSCVI.setup_anndata(adata_sc, labels_key=cell_type_col)
    sc_model = scvi.model.CondSCVI(adata_sc, weight_obs=False)
    sc_model.train(max_epochs=25, accelerator='cpu')
    
    scvi.model.DestVI.setup_anndata(adata_st)
    st_model_id = scvi.model.DestVI.from_rna_model(adata_st, sc_model)
    st_model_id.train(max_epochs=25, accelerator='cpu')
    
    true_props_id = adata_st.obsm["proportions"].values
    pred_props_id = st_model_id.get_proportions().values
    _, _, ece_id, _ = get_calibration_stats(true_props_id, pred_props_id)
    
    seed_results = {"ece_id": ece_id, "fractions": {}}
    
    for frac in fractions:
        print(f"    Running fraction {frac}...")
        scvi.settings.seed = seed
        pl.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        adata_ood = downsample_counts(adata_st, fraction=frac)
        scvi.model.DestVI.setup_anndata(adata_ood)
        st_model_ood = scvi.model.DestVI.from_rna_model(adata_ood, sc_model)
        st_model_ood.train(max_epochs=25, accelerator='cpu')
        
        true_props_ood = adata_ood.obsm["proportions"].values
        pred_props_ood = st_model_ood.get_proportions().values
        
        n_cal = len(true_props_ood) // 2
        true_props_cal = true_props_ood[:n_cal]
        pred_props_cal = pred_props_ood[:n_cal]
        true_props_test = true_props_ood[n_cal:]
        pred_props_test = pred_props_ood[n_cal:]
        
        _, _, ece_ood_test, _ = get_calibration_stats(true_props_test, pred_props_test)
        
        # Temp Scaling
        best_t, _ = optimize_temperature(true_props_cal, pred_props_cal)
        _, _, ece_temp_test, _ = get_calibration_stats(true_props_test, pred_props_test, temp=best_t)
        
        # Isotonic Regression
        conf_cal = np.max(pred_props_cal, axis=1)
        acc_cal = (np.argmax(pred_props_cal, axis=1) == np.argmax(true_props_cal, axis=1)).astype(int)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(conf_cal, acc_cal)
        _, _, ece_iso_test, _ = get_calibration_stats(true_props_test, pred_props_test, iso_reg=iso)
        
        seed_results["fractions"][frac] = {
            "ece_ood": ece_ood_test,
            "ece_temp": ece_temp_test,
            "ece_iso": ece_iso_test,
            "best_t": best_t
        }
        
    return seed_results

def main():
    print("Loading data...")
    adata_sc = sc.read_h5ad("data/processed_sc_reference.h5ad")
    adata_st = sc.read_h5ad("data/processed_pseudospots.h5ad")
    
    seeds = [42, 123, 2026, 777, 999, 1001, 2002, 3003, 4004, 5005]
    fractions = [0.8, 0.6, 0.4, 0.2]
    
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"--- Running Replicate {i+1}/{len(seeds)} (Seed {seed}) ---")
        res = run_seed_pipeline(adata_sc, adata_st, seed, fractions)
        all_results.append(res)
    
    # Aggregate results
    final_output = {
        "seeds": seeds,
        "fractions": fractions,
        "ece_id": {"raw": [r["ece_id"] for r in all_results]}
    }
    final_output["ece_id"]["mean"] = float(np.mean(final_output["ece_id"]["raw"]))
    final_output["ece_id"]["std"] = float(np.std(final_output["ece_id"]["raw"]))
    
    print(f"\nFinal Results across {len(seeds)} replicates:")
    print(f"ID ECE: {final_output['ece_id']['mean']:.3f} +/- {final_output['ece_id']['std']:.3f}")
    
    for frac in fractions:
        f_str = str(frac)
        final_output[f_str] = {
            "ece_ood": [r["fractions"][frac]["ece_ood"] for r in all_results],
            "ece_temp": [r["fractions"][frac]["ece_temp"] for r in all_results],
            "ece_iso": [r["fractions"][frac]["ece_iso"] for r in all_results]
        }
        
        mean_ood = float(np.mean(final_output[f_str]["ece_ood"]))
        std_ood = float(np.std(final_output[f_str]["ece_ood"]))
        mean_temp = float(np.mean(final_output[f_str]["ece_temp"]))
        std_temp = float(np.std(final_output[f_str]["ece_temp"]))
        mean_iso = float(np.mean(final_output[f_str]["ece_iso"]))
        std_iso = float(np.std(final_output[f_str]["ece_iso"]))
        
        # Paired t-tests
        t_stat, p_val = ttest_rel(final_output["ece_id"]["raw"], final_output[f_str]["ece_ood"])
        t_stat_iso, p_val_iso = ttest_rel(final_output[f_str]["ece_temp"], final_output[f_str]["ece_iso"])
        
        final_output[f_str].update({
            "mean_ood": mean_ood, "std_ood": std_ood,
            "mean_temp": mean_temp, "std_temp": std_temp,
            "mean_iso": mean_iso, "std_iso": std_iso,
            "ttest_vs_id": {"t_stat": float(t_stat), "p_val": float(p_val)},
            "ttest_temp_vs_iso": {"t_stat": float(t_stat_iso), "p_val": float(p_val_iso)}
        })
        
        print(f"\nFraction {frac}:")
        print(f"  OOD ECE: {mean_ood:.3f} +/- {std_ood:.3f} (p={p_val:.4f} vs ID)")
        print(f"  Temp Scaling ECE: {mean_temp:.3f} +/- {std_temp:.3f}")
        print(f"  Isotonic ECE: {mean_iso:.3f} +/- {std_iso:.3f} (p={p_val_iso:.4e} vs Temp)")
        
    os.makedirs("figures", exist_ok=True)
    with open("results.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("Metrics saved to results.json")
    
    print("Generating dose-response figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.array(fractions)
    
    # Sort by decreasing fraction (increasing dropout severity)
    # fraction=1.0 is ID
    x_plot = np.array([1.0] + sorted(fractions, reverse=True))
    
    y_ood = [final_output["ece_id"]["mean"]] + [final_output[str(f)]["mean_ood"] for f in sorted(fractions, reverse=True)]
    err_ood = [final_output["ece_id"]["std"]] + [final_output[str(f)]["std_ood"] for f in sorted(fractions, reverse=True)]
    
    y_temp = [final_output["ece_id"]["mean"]] + [final_output[str(f)]["mean_temp"] for f in sorted(fractions, reverse=True)]
    err_temp = [final_output["ece_id"]["std"]] + [final_output[str(f)]["std_temp"] for f in sorted(fractions, reverse=True)]
    
    y_iso = [final_output["ece_id"]["mean"]] + [final_output[str(f)]["mean_iso"] for f in sorted(fractions, reverse=True)]
    err_iso = [final_output["ece_id"]["std"]] + [final_output[str(f)]["std_iso"] for f in sorted(fractions, reverse=True)]
    
    ax.errorbar(x_plot, y_ood, yerr=err_ood, label='Uncalibrated', marker='o', capsize=5)
    ax.errorbar(x_plot, y_temp, yerr=err_temp, label='Temperature Scaling', marker='s', capsize=5)
    ax.errorbar(x_plot, y_iso, yerr=err_iso, label='Isotonic Regression', marker='^', capsize=5)
    
    ax.set_xlabel('Capture Efficiency Fraction (1.0 = ID)')
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title(f'Calibration Degradation under Dose-Response Shift (n={len(seeds)})')
    ax.invert_xaxis() # 1.0 -> 0.2
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("figures/ece_degradation.png", dpi=300)
    print("Saved dose-response figure to figures/ece_degradation.png")

if __name__ == "__main__":
    main()
