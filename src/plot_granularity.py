import json
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    try:
        with open("results_bioinfo_kfold.json", "r") as f:
            subclass_res = json.load(f)
    except Exception as e:
        print("Could not load subclass results:", e)
        return
        
    try:
        with open("results_class_kfold.json", "r") as f:
            class_res = json.load(f)
    except Exception as e:
        print("Could not load class results:", e)
        return

    fractions = class_res.get('fractions', [1.0, 0.8, 0.6, 0.4, 0.2])
    n_splits = class_res.get('n_splits', 5)
    
    sub_iso, sub_iso_s = [], []
    sub_temp, sub_temp_s = [], []
    cls_iso, cls_iso_s = [], []
    cls_temp, cls_temp_s = [], []
    
    for f in fractions:
        f_str = str(f)
        if f_str in subclass_res:
            s_iso = subclass_res[f_str]['ece_iso']
            s_temp = subclass_res[f_str]['ece_temp']
            sub_iso.append(np.mean(s_iso))
            sub_iso_s.append(np.std(s_iso)/np.sqrt(5))
            sub_temp.append(np.mean(s_temp))
            sub_temp_s.append(np.std(s_temp)/np.sqrt(5))
        else:
            sub_iso.append(np.nan)
            sub_iso_s.append(np.nan)
            sub_temp.append(np.nan)
            sub_temp_s.append(np.nan)
            
        if f_str in class_res:
            fold_iso = [fold['ece_iso_mean'] for fold in class_res[f_str]['boot_shifted']]
            fold_temp = [fold['ece_temp_mean'] for fold in class_res[f_str]['boot_shifted']]
            cls_iso.append(np.mean(fold_iso))
            cls_iso_s.append(np.std(fold_iso)/np.sqrt(5))
            cls_temp.append(np.mean(fold_temp))
            cls_temp_s.append(np.std(fold_temp)/np.sqrt(5))
        else:
            cls_iso.append(np.nan)
            cls_iso_s.append(np.nan)
            cls_temp.append(np.nan)
            cls_temp_s.append(np.nan)
            
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_plot = (1.0 - np.array(fractions)) * 100
    
    ax.errorbar(x_plot, sub_iso, yerr=sub_iso_s, marker='o', label='Subclass (23) - IsoReg', color='blue', linestyle='-')
    ax.errorbar(x_plot, sub_temp, yerr=sub_temp_s, marker='^', label='Subclass (23) - TempScale', color='blue', linestyle='--')
    
    ax.errorbar(x_plot, cls_iso, yerr=cls_iso_s, marker='o', label='Class (4) - IsoReg', color='green', linestyle='-')
    ax.errorbar(x_plot, cls_temp, yerr=cls_temp_s, marker='^', label='Class (4) - TempScale', color='green', linestyle='--')
    
    ax.set_xlabel('Capture Efficiency Dropout (%)', fontsize=12)
    ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
    ax.set_title('Calibration Sensitivity to Taxonomic Granularity', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/granularity_ece.png", dpi=300)
    print("Saved granularity comparison to figures/granularity_ece.png")
    
if __name__ == "__main__":
    main()
