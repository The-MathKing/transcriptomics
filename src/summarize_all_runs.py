import json
import numpy as np

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

def get_stats(data, cond):
    if cond not in data:
        return {}
    d = data[cond]
    ece_uncal = d.get("ece_ood", [])
    brier_uncal = d.get("brier_ood", [])
    
    ece_ts = [r["ece_temp_mean"] for r in d.get("boot_shifted", []) if "ece_temp_mean" in r]
    ece_iso = [r["ece_iso_mean"] for r in d.get("boot_shifted", []) if "ece_iso_mean" in r]
    brier_ts = [r["brier_temp_mean"] for r in d.get("boot_shifted", []) if "brier_temp_mean" in r]
    brier_iso = [r["brier_iso_mean"] for r in d.get("boot_shifted", []) if "brier_iso_mean" in r]
    nll_ts = [r["nll_temp_mean"] for r in d.get("boot_shifted", []) if "nll_temp_mean" in r]
    nll_iso = [r["nll_iso_mean"] for r in d.get("boot_shifted", []) if "nll_iso_mean" in r]
    
    def fmt(arr):
        if not arr: return "N/A"
        mean = np.mean(arr)
        sem = np.std(arr) / np.sqrt(len(arr))
        return f"{mean:.3f} +/- {sem:.3f}"
        
    return {
        "ece_uncal": fmt(ece_uncal),
        "ece_ts": fmt(ece_ts),
        "ece_iso": fmt(ece_iso),
        "brier_uncal": fmt(brier_uncal),
        "brier_ts": fmt(brier_ts),
        "brier_iso": fmt(brier_iso),
        "nll_ts": fmt(nll_ts),
        "nll_iso": fmt(nll_iso),
    }

def print_summary():
    res_destvi_k23, res_c2l_k23, res_destvi_k4, res_c2l_k4 = load_data()
    print("=" * 80)
    print("TABLE OF EXPERIMENTAL RESULTS ACROSS MODELS AND GRANULARITIES (n=5 folds)")
    print("=" * 80)
    
    for g_name, d_destvi, d_c2l in [("Fine (k=23)", res_destvi_k23, res_c2l_k23), ("Coarse (k=4)", res_destvi_k4, res_c2l_k4)]:
        print(f"\n--- {g_name} ---")
        for cond in ["1.0", "0.2", "noise_2.0"]:
            cond_label = "Clean (0% Dropout)" if cond == "1.0" else ("80% Dropout Shift" if cond == "0.2" else "Poisson 2.0x Shift")
            print(f"\n[{cond_label}]")
            s_destvi = get_stats(d_destvi, cond)
            s_c2l = get_stats(d_c2l, cond)
            print(f"DestVI   | Uncal ECE: {s_destvi['ece_uncal']:<16} | TS ECE: {s_destvi['ece_ts']:<16} | Iso ECE: {s_destvi['ece_iso']:<16}")
            print(f"cell2loc | Uncal ECE: {s_c2l['ece_uncal']:<16} | TS ECE: {s_c2l['ece_ts']:<16} | Iso ECE: {s_c2l['ece_iso']:<16}")

if __name__ == "__main__":
    print_summary()
