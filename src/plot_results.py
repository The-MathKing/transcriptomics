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
        
        c_ts = [fold['ece_temp_mean'] for fold in data['boot_clean']]
        ece_temp_mean.append(np.mean(c_ts))
        ece_temp_std.append(np.std(c_ts) / np.sqrt(n_splits))
        
        c_iso = [fold['ece_iso_mean'] for fold in data['boot_clean']]
        ece_iso_mean.append(np.mean(c_iso))
        ece_iso_std.append(np.std(c_iso) / np.sqrt(n_splits))
        
    fig, ax = plt.subplots(figsize=(8, 5))
    x_plot = np.array(fractions)
    
    # Plot means as lines
    ax.plot(x_plot, ece_ood_mean, label='Uncalibrated', marker='o', color='#1f77b4', markersize=6)
    ax.plot(x_plot, ece_temp_mean, label='Temperature Scaling', marker='s', color='#ff7f0e', markersize=6)
    ax.plot(x_plot, ece_iso_mean, label='Isotonic Regression', marker='^', color='#2ca02c', markersize=6)
    
    # Scatter raw fold-level values
    for frac in fractions:
        f_str = str(frac)
        data = results[f_str]
        n_pts = len(data['ece_ood'])
        
        # Add jitter
        jitter = np.random.normal(0, 0.01, n_pts)
        
        ax.scatter([frac]*n_pts + jitter, data['ece_ood'], color='#1f77b4', alpha=0.3, s=20)
        c_ts = [fold['ece_temp_mean'] for fold in data['boot_clean']]
        ax.scatter([frac]*len(c_ts) + jitter, c_ts, color='#ff7f0e', alpha=0.3, s=20)
        c_iso = [fold['ece_iso_mean'] for fold in data['boot_clean']]
        ax.scatter([frac]*len(c_iso) + jitter, c_iso, color='#2ca02c', alpha=0.3, s=20)
    
    ax.set_xlabel('Capture Efficiency Fraction (1.0 = In-Distribution)')
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title(f'Calibration Under Deployment-Time Shift (frozen model, n={n_splits} folds)')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/ece_degradation.png', dpi=300)
    print("Saved dose-response figure to figures/ece_degradation.png")

if __name__ == "__main__":
    main()
