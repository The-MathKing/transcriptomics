import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    with open('results_bioinfo_kfold.json', 'r') as f:
        results = json.load(f)
        
    fractions = results['fractions']
    n_splits = results['n_splits']
    
    ece_ood_mean = []
    ece_ood_std = []
    ece_temp_mean = []
    ece_temp_std = []
    ece_iso_mean = []
    ece_iso_std = []
    
    # We'll plot against fractions
    for frac in fractions:
        f_str = str(frac)
        data = results[f_str]
        
        # Calculate mean and std error across folds (n=5)
        ece_ood_mean.append(np.mean(data['ece_ood']))
        ece_ood_std.append(np.std(data['ece_ood']) / np.sqrt(n_splits))
        
        ece_temp_mean.append(np.mean(data['ece_temp']))
        ece_temp_std.append(np.std(data['ece_temp']) / np.sqrt(n_splits))
        
        ece_iso_mean.append(np.mean(data['ece_iso']))
        ece_iso_std.append(np.std(data['ece_iso']) / np.sqrt(n_splits))
        
    fig, ax = plt.subplots(figsize=(8, 5))
    x_plot = np.array(fractions)
    
    ax.errorbar(x_plot, ece_ood_mean, yerr=ece_ood_std, label='Uncalibrated', marker='o', capsize=5, color='#1f77b4')
    ax.errorbar(x_plot, ece_temp_mean, yerr=ece_temp_std, label='Temperature Scaling', marker='s', capsize=5, color='#ff7f0e')
    ax.errorbar(x_plot, ece_iso_mean, yerr=ece_iso_std, label='Isotonic Regression', marker='^', capsize=5, color='#2ca02c')
    
    ax.set_xlabel('Capture Efficiency Fraction (1.0 = In-Distribution)')
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title(f'Calibration Under Deployment-Time Shift (frozen model, n={n_splits})')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/ece_degradation.png', dpi=300)
    print("Saved dose-response figure to figures/ece_degradation.png")

if __name__ == "__main__":
    main()
