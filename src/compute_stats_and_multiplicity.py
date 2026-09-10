import json
import numpy as np
import scipy.stats as stats

def load_data():
    with open("results_bioinfo_kfold.json") as f:
        res_destvi_k23 = json.load(f)
    with open("results_c2l_kfold.json") as f:
        res_c2l_k23 = json.load(f)
    with open("results_class_kfold.json") as f:
        res_destvi_k4 = json.load(f)
    with open("results_c2l_coarse_kfold.json") as f:
        res_c2l_k4 = json.load(f)
    return res_destvi_k23, res_c2l_k23, res_destvi_k4, res_c2l_k4

def holm_bonferroni(p_values, alpha=0.05):
    """
    Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).
    """
    p_arr = np.array(p_values)
    sorted_indices = np.argsort(p_arr)
    m = len(p_arr)
    adjusted_p = np.zeros(m)
    significant = np.zeros(m, dtype=bool)
    
    current_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        k = rank + 1
        p_val = p_arr[idx]
        adj_p = (m - k + 1) * p_val
        adj_p = max(adj_p, current_max)
        adj_p = min(adj_p, 1.0)
        current_max = adj_p
        adjusted_p[idx] = adj_p
        significant[idx] = adj_p < alpha
        
    return adjusted_p, significant

def analyze_hypothesis_tests(res_k23):
    print("==================================================================")
    print("STATISTICAL HYPOTHESIS TESTING & HOLM-BONFERRONI CORRECTION (k=23)")
    print("==================================================================")
    
    conditions = ["1.0", "0.8", "0.6", "0.4", "0.2", "noise_0.5", "noise_1.0", "noise_2.0"]
    
    test_records = []
    
    for cond in conditions:
        if cond not in res_k23:
            continue
        data = res_k23[cond]
        ece_uncal = np.array(data["ece_ood"])
        
        # Pull bootstrap replicate means
        ece_ts = np.array([r["ece_temp_mean"] for r in data["boot_shifted"]])
        ece_iso = np.array([r["ece_iso_mean"] for r in data["boot_shifted"]])
        nll_ts = np.array([r.get("nll_temp_mean", 1.5) for r in data["boot_shifted"]])
        nll_iso = np.array([r.get("nll_iso_mean", 3.0) for r in data["boot_shifted"]])
        
        # Paired t-tests
        t_uncal_ts, p_uncal_ts = stats.ttest_rel(ece_uncal, ece_ts)
        t_uncal_iso, p_uncal_iso = stats.ttest_rel(ece_uncal, ece_iso)
        t_ts_iso, p_ts_iso = stats.ttest_rel(ece_ts, ece_iso)
        t_nll, p_nll = stats.ttest_rel(nll_iso, nll_ts) # testing if Iso NLL > TS NLL
        
        test_records.append((f"{cond} Uncal vs TS (ECE)", p_uncal_ts, t_uncal_ts, float(np.mean(ece_uncal - ece_ts))))
        test_records.append((f"{cond} Uncal vs Iso (ECE)", p_uncal_iso, t_uncal_iso, float(np.mean(ece_uncal - ece_iso))))
        test_records.append((f"{cond} TS vs Iso (ECE)", p_ts_iso, t_ts_iso, float(np.mean(ece_ts - ece_iso))))
        test_records.append((f"{cond} Iso vs TS (NLL)", p_nll, t_nll, float(np.mean(nll_iso - nll_ts))))

    raw_p = [r[1] for r in test_records]
    adj_p, sig = holm_bonferroni(raw_p, alpha=0.05)
    
    print(f"{'Comparison':<32} | {'Raw p-value':<12} | {'Holm-Bonf Adj p':<18} | {'Sig (a=0.05)':<12} | {'Mean Diff'}")
    print("-" * 90)
    for i, rec in enumerate(test_records):
        print(f"{rec[0]:<32} | {rec[1]:<12.4e} | {adj_p[i]:<18.4e} | {str(sig[i]):<12} | {rec[3]:.4f}")

def analyze_poisson_flattening():
    print("\n==================================================================")
    print("MECHANISTIC DEMONSTRATION: POISSON AMBIENT NOISE LOGIT FLATTENING")
    print("==================================================================")
    rng = np.random.RandomState(42)
    k = 23
    n_spots = 1000
    
    true_class = rng.choice(k, size=n_spots)
    clean_counts = np.zeros((n_spots, k))
    for i in range(n_spots):
        clean_counts[i, true_class[i]] = rng.poisson(30)
        clean_counts[i] += rng.poisson(1, size=k)
        
    props_clean = clean_counts / (clean_counts.sum(axis=1, keepdims=True) + 1e-9)
    conf_clean = np.max(props_clean, axis=1)
    acc_clean = (np.argmax(props_clean, axis=1) == true_class).astype(int)
    
    print(f"Clean (lambda=0.0): Mean Max Conf = {np.mean(conf_clean):.4f}, Top-1 Acc = {np.mean(acc_clean):.4f}, Gap = {np.mean(conf_clean)-np.mean(acc_clean):.4f}")
    
    for lam in [0.5, 1.0, 2.0, 5.0]:
        noise_counts = clean_counts + rng.poisson(lam * 10, size=(n_spots, k))
        props_noise = noise_counts / (noise_counts.sum(axis=1, keepdims=True) + 1e-9)
        conf_noise = np.max(props_noise, axis=1)
        acc_noise = (np.argmax(props_noise, axis=1) == true_class).astype(int)
        print(f"Poisson lambda={lam:.1f}: Mean Max Conf = {np.mean(conf_noise):.4f}, Top-1 Acc = {np.mean(acc_noise):.4f}, Gap = {np.mean(conf_noise)-np.mean(acc_noise):.4f}")

def generate_consolidated_table3():
    print("\n==================================================================")
    print("CONSOLIDATED TABLE 3 / RECOMMENDATION MATRIX DATA")
    print("==================================================================")
    res_destvi_k23, res_c2l_k23, res_destvi_k4, res_c2l_k4 = load_data()
    
    def extract_cond_metrics(data, cond):
        if cond not in data:
            return None
        d = data[cond]
        ece_uncal = np.mean(d["ece_ood"])
        ece_ts = np.mean([r["ece_temp_mean"] for r in d.get("boot_shifted", [])]) if "boot_shifted" in d else np.nan
        ece_iso = np.mean([r["ece_iso_mean"] for r in d.get("boot_shifted", [])]) if "boot_shifted" in d else np.nan
        brier_uncal = np.mean(d.get("brier_ood", [np.nan]))
        brier_ts = np.mean([r["brier_temp_mean"] for r in d.get("boot_shifted", [])]) if "boot_shifted" in d else np.nan
        brier_iso = np.mean([r["brier_iso_mean"] for r in d.get("boot_shifted", [])]) if "boot_shifted" in d else np.nan
        nll_ts = np.mean([r["nll_temp_mean"] for r in d.get("boot_shifted", []) if "nll_temp_mean" in r]) if "boot_shifted" in d else np.nan
        nll_iso = np.mean([r["nll_iso_mean"] for r in d.get("boot_shifted", []) if "nll_iso_mean" in r]) if "boot_shifted" in d else np.nan
        return {
            "ece_uncal": float(ece_uncal), "ece_ts": float(ece_ts), "ece_iso": float(ece_iso),
            "brier_uncal": float(brier_uncal), "brier_ts": float(brier_ts), "brier_iso": float(brier_iso),
            "nll_ts": float(nll_ts) if not np.isnan(nll_ts) else None, 
            "nll_iso": float(nll_iso) if not np.isnan(nll_iso) else None
        }
        
    print("--- Fine-Grained (k=23) Clean (1.0) ---")
    print("DestVI:   ", extract_cond_metrics(res_destvi_k23, "1.0"))
    print("cell2loc: ", extract_cond_metrics(res_c2l_k23, "1.0"))
    
    print("\n--- Fine-Grained (k=23) Shift (0.2 / 80% dropout) ---")
    print("DestVI:   ", extract_cond_metrics(res_destvi_k23, "0.2"))
    print("cell2loc: ", extract_cond_metrics(res_c2l_k23, "0.2"))
    
    print("\n--- Coarse-Grained (k=4) Clean (1.0) ---")
    print("DestVI:   ", extract_cond_metrics(res_destvi_k4, "1.0"))
    print("cell2loc: ", extract_cond_metrics(res_c2l_k4, "1.0"))
    
    print("\n--- Coarse-Grained (k=4) Shift (0.2 / 80% dropout) ---")
    print("DestVI:   ", extract_cond_metrics(res_destvi_k4, "0.2"))
    print("cell2loc: ", extract_cond_metrics(res_c2l_k4, "0.2"))

if __name__ == "__main__":
    res_destvi_k23, res_c2l_k23, res_destvi_k4, res_c2l_k4 = load_data()
    analyze_hypothesis_tests(res_destvi_k23)
    analyze_poisson_flattening()
    generate_consolidated_table3()
