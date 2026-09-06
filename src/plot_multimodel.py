import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    try:
        with open('results_bioinfo_kfold.json', 'r') as f:
            destvi_results = json.load(f)
    except FileNotFoundError:
        print("results_bioinfo_kfold.json not found")
        return
        
    try:
        with open('results_c2l_kfold.json', 'r') as f:
            c2l_results = json.load(f)
    except FileNotFoundError:
        print("results_c2l_kfold.json not found")
        return
        
    fractions = c2l_results.get('fractions', [1.0, 0.6, 0.2])
    
    n_splits_destvi = destvi_results.get('n_splits', 5)
    n_splits_c2l = c2l_results.get('n_splits', 5) # Updated to 5!
    
    destvi_ece_m, destvi_ece_s = [], []
    destvi_ts_m, destvi_ts_s = [], []
    destvi_iso_m, destvi_iso_s = [], []
    
    c2l_ece_m, c2l_ece_s = [], []
    c2l_ts_m, c2l_ts_s = [], []
    c2l_iso_m, c2l_iso_s = [], []
    
    for frac in fractions:
        f_str = str(frac)
        
        # DestVI
        if f_str in destvi_results:
            d_ece = destvi_results[f_str]['ece_ood']
            destvi_ece_m.append(np.mean(d_ece))
            destvi_ece_s.append(np.std(d_ece) / np.sqrt(n_splits_destvi))
            
            d_ts = destvi_results[f_str]['ece_temp']
            destvi_ts_m.append(np.mean(d_ts))
            destvi_ts_s.append(np.std(d_ts) / np.sqrt(n_splits_destvi))
            
            d_iso = destvi_results[f_str]['ece_iso']
            destvi_iso_m.append(np.mean(d_iso))
            destvi_iso_s.append(np.std(d_iso) / np.sqrt(n_splits_destvi))
            
        # Cell2Location
        if f_str in c2l_results:
            c_ece = c2l_results[f_str]['ece_ood']
            c2l_ece_m.append(np.mean(c_ece))
            c2l_ece_s.append(np.std(c_ece) / np.sqrt(n_splits_c2l))
            
            c_ts = [fold['ece_temp_mean'] for fold in c2l_results[f_str]['boot_shifted']]
            c2l_ts_m.append(np.mean(c_ts))
            c2l_ts_s.append(np.std(c_ts) / np.sqrt(n_splits_c2l))
            
            c_iso = [fold['ece_iso_mean'] for fold in c2l_results[f_str]['boot_shifted']]
            c2l_iso_m.append(np.mean(c_iso))
            c2l_iso_s.append(np.std(c_iso) / np.sqrt(n_splits_c2l))
            
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x_plot = np.array(fractions)
    
    # DestVI Plot
    axes[0].errorbar(x_plot, destvi_ece_m, yerr=destvi_ece_s, label='Uncalibrated', marker='o', color='gray', linestyle='--')
    axes[0].errorbar(x_plot, destvi_ts_m, yerr=destvi_ts_s, label='Temperature Scaling', marker='s', color='#1f77b4')
    axes[0].errorbar(x_plot, destvi_iso_m, yerr=destvi_iso_s, label='Isotonic Regression', marker='D', color='#2ca02c')
    axes[0].set_title('DestVI (Amortized)')
    axes[0].set_xlabel('Capture Efficiency Fraction (1.0 = Clean)')
    axes[0].set_ylabel('Expected Calibration Error (ECE)')
    axes[0].invert_xaxis()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Cell2Location Plot
    axes[1].errorbar(x_plot, c2l_ece_m, yerr=c2l_ece_s, label='Uncalibrated', marker='o', color='gray', linestyle='--')
    axes[1].errorbar(x_plot, c2l_ts_m, yerr=c2l_ts_s, label='Temperature Scaling', marker='s', color='#1f77b4')
    axes[1].errorbar(x_plot, c2l_iso_m, yerr=c2l_iso_s, label='Isotonic Regression', marker='D', color='#2ca02c')
    axes[1].set_title('cell2location (Per-Spot)')
    axes[1].set_xlabel('Capture Efficiency Fraction (1.0 = Clean)')
    axes[1].invert_xaxis()
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/multimodel_ece.png', dpi=300)
    print("Saved multi-model comparison to figures/multimodel_ece.png")
        
if __name__ == "__main__":
    main()
