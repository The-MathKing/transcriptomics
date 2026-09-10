import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from netcal.metrics import ECE

def beta_calibration(p, y):
    eps = 1e-7
    p_safe = np.clip(p, eps, 1-eps)
    logit_p = np.log(p_safe / (1 - p_safe))
    
    def loss(a):
        logit_cal = a * logit_p
        p_cal = 1 / (1 + np.exp(-logit_cal))
        p_cal = np.clip(p_cal, eps, 1-eps)
        return -np.sum(y * np.log(p_cal) + (1-y) * np.log(1-p_cal))
    
    res = minimize(loss, [1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
    a_opt = res.x[0]
    logit_cal = a_opt * logit_p
    p_cal = 1 / (1 + np.exp(-logit_cal))
    return np.clip(p_cal, eps, 1-eps), a_opt

def temperature_scaling(p, y):
    eps = 1e-7
    p_safe = np.clip(p, eps, 1-eps)
    logit_p = np.log(p_safe / (1 - p_safe))
    
    def loss(t):
        logit_cal = logit_p / t
        p_cal = 1 / (1 + np.exp(-logit_cal))
        p_cal = np.clip(p_cal, eps, 1-eps)
        return -np.sum(y * np.log(p_cal) + (1-y) * np.log(1-p_cal))
    
    res = minimize(loss, [1.0], bounds=[(0.05, 50.0)], method='L-BFGS-B')
    t_opt = res.x[0]
    logit_cal = logit_p / t_opt
    p_cal = 1 / (1 + np.exp(-logit_cal))
    return np.clip(p_cal, eps, 1-eps), t_opt

# Define miscalibration functions
def g_power(s):
    return s**1.5

def g_quadratic(s):
    return s**2

def g_sigmoid(s):
    return 1 / (1 + np.exp(-6 * (s - 0.5)))

def sample_marginal(family_name, n, rng):
    if family_name == "Beta_2_5_to_5_2":
        return rng.beta(a=2, b=5, size=n), rng.beta(a=5, b=2, size=n)
    elif family_name == "Beta_1_3_to_3_1":
        return rng.beta(a=1, b=3, size=n), rng.beta(a=3, b=1, size=n)
    elif family_name == "TruncNormal_0.3_to_0.7":
        s_clean = np.clip(rng.normal(loc=0.3, scale=0.12, size=n), 0.01, 0.99)
        s_shift = np.clip(rng.normal(loc=0.7, scale=0.12, size=n), 0.01, 0.99)
        return s_clean, s_shift
    elif family_name == "Uniform_to_Beta_4_2":
        s_clean = rng.uniform(0.01, 0.99, size=n)
        s_shift = rng.beta(a=4, b=2, size=n)
        return s_clean, s_shift
    else:
        raise ValueError(f"Unknown family: {family_name}")

def run_comprehensive_density_sweep(n_reps=10, n_samples=5000):
    families = [
        ("Beta_2_5_to_5_2", r"$\text{Beta}(2,5) \to \text{Beta}(5,2)$"),
        ("Beta_1_3_to_3_1", r"$\text{Beta}(1,3) \to \text{Beta}(3,1)$"),
        ("TruncNormal_0.3_to_0.7", r"$\mathcal{TN}(0.3, 0.12) \to \mathcal{TN}(0.7, 0.12)$"),
        ("Uniform_to_Beta_4_2", r"$\mathcal{U}(0,1) \to \text{Beta}(4,2)$")
    ]
    
    g_funcs = [
        ("Power ($s^{1.5}$)", g_power),
        ("Quadratic ($s^2$)", g_quadratic),
        ("Sigmoidal", g_sigmoid)
    ]
    
    results = {}
    ece_metric = ECE(bins=15)
    
    print(f"Running comprehensive density sweep (n_reps={n_reps}, n_samples={n_samples})...")
    
    for fam_key, fam_label in families:
        results[fam_key] = {"label": fam_label, "experiments": {}}
        for g_name, g_fn in g_funcs:
            uncal_clean_list, uncal_shift_list = [], []
            iso_clean_list, iso_shift_list = [], []
            ts_clean_list, ts_shift_list = [], []
            param_clean_list, param_shift_list = [], []
            
            for rep in range(n_reps):
                rng = np.random.RandomState(42 + rep)
                S_cal, S_test = sample_marginal(fam_key, n_samples, rng)
                
                Y_cal = rng.binomial(1, np.clip(g_fn(S_cal), 0, 1))
                Y_test = rng.binomial(1, np.clip(g_fn(S_test), 0, 1))
                
                # Fit calibrators on clean
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(S_cal, Y_cal)
                
                _, a_opt = beta_calibration(S_cal, Y_cal)
                _, t_opt = temperature_scaling(S_cal, Y_cal)
                
                # Clean evaluations
                iso_clean = iso.predict(S_cal)
                ts_clean = 1 / (1 + np.exp(-(np.log(np.clip(S_cal, 1e-7, 1-1e-7)/(1-np.clip(S_cal, 1e-7, 1-1e-7))) / t_opt)))
                param_clean = 1 / (1 + np.exp(-(a_opt * np.log(np.clip(S_cal, 1e-7, 1-1e-7)/(1-np.clip(S_cal, 1e-7, 1-1e-7))))))
                
                uncal_clean_list.append(ece_metric.measure(S_cal, Y_cal))
                iso_clean_list.append(ece_metric.measure(iso_clean, Y_cal))
                ts_clean_list.append(ece_metric.measure(ts_clean, Y_cal))
                param_clean_list.append(ece_metric.measure(param_clean, Y_cal))
                
                # Shift evaluations
                iso_shift = iso.predict(S_test)
                ts_shift = 1 / (1 + np.exp(-(np.log(np.clip(S_test, 1e-7, 1-1e-7)/(1-np.clip(S_test, 1e-7, 1-1e-7))) / t_opt)))
                param_shift = 1 / (1 + np.exp(-(a_opt * np.log(np.clip(S_test, 1e-7, 1-1e-7)/(1-np.clip(S_test, 1e-7, 1-1e-7))))))
                
                uncal_shift_list.append(ece_metric.measure(S_test, Y_test))
                iso_shift_list.append(ece_metric.measure(iso_shift, Y_test))
                ts_shift_list.append(ece_metric.measure(ts_shift, Y_test))
                param_shift_list.append(ece_metric.measure(param_shift, Y_test))
                
            results[fam_key]["experiments"][g_name] = {
                "uncal_clean": [float(np.mean(uncal_clean_list)), float(np.std(uncal_clean_list)/np.sqrt(n_reps))],
                "uncal_shift": [float(np.mean(uncal_shift_list)), float(np.std(uncal_shift_list)/np.sqrt(n_reps))],
                "iso_clean": [float(np.mean(iso_clean_list)), float(np.std(iso_clean_list)/np.sqrt(n_reps))],
                "iso_shift": [float(np.mean(iso_shift_list)), float(np.std(iso_shift_list)/np.sqrt(n_reps))],
                "ts_clean": [float(np.mean(ts_clean_list)), float(np.std(ts_clean_list)/np.sqrt(n_reps))],
                "ts_shift": [float(np.mean(ts_shift_list)), float(np.std(ts_shift_list)/np.sqrt(n_reps))],
                "param_clean": [float(np.mean(param_clean_list)), float(np.std(param_clean_list)/np.sqrt(n_reps))],
                "param_shift": [float(np.mean(param_shift_list)), float(np.std(param_shift_list)/np.sqrt(n_reps))],
            }
            
            print(f"[{fam_key} | {g_name}] Iso Clean: {np.mean(iso_clean_list):.4f} -> Shift: {np.mean(iso_shift_list):.4f} | TS Clean: {np.mean(ts_clean_list):.4f} -> Shift: {np.mean(ts_shift_list):.4f}")

    # Save to JSON
    with open("results_synthetic_density_comprehensive.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to results_synthetic_density_comprehensive.json")

    # Generate multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    
    for idx, (fam_key, fam_label) in enumerate(families):
        ax = axes[idx]
        exp = results[fam_key]["experiments"]
        
        g_names = list(exp.keys())
        x = np.arange(len(g_names))
        width = 0.2
        
        iso_clean = [exp[g]["iso_clean"][0] for g in g_names]
        iso_shift = [exp[g]["iso_shift"][0] for g in g_names]
        ts_clean = [exp[g]["ts_clean"][0] for g in g_names]
        ts_shift = [exp[g]["ts_shift"][0] for g in g_names]
        
        ax.bar(x - 1.5*width, iso_clean, width, label="Isotonic (Clean)", color="#e74c3c", alpha=0.6)
        ax.bar(x - 0.5*width, iso_shift, width, label="Isotonic (Shifted)", color="#c0392b")
        ax.bar(x + 0.5*width, ts_clean, width, label="TS (Clean)", color="#3498db", alpha=0.6)
        ax.bar(x + 1.5*width, ts_shift, width, label="TS (Shifted)", color="#2980b9")
        
        ax.set_title(fam_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(g_names, fontsize=10)
        ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)
            
    plt.suptitle("Synthetic Marginal Density Invariance: Non-Parametric vs. Parametric Calibration", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/density_mechanism_expanded.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved figure to figures/density_mechanism_expanded.png")

if __name__ == "__main__":
    run_comprehensive_density_sweep(n_reps=10, n_samples=5000)
