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
    
    print("--- Table 1 Values Generated from results_bioinfo_kfold.json ---")
    print(f"{'Shift (Dropout)':<16} | {'Uncalibrated':<15} | {'Temp Scaled':<15} | {'Isotonic Reg':<15}")
    print("-" * 70)
    
    for frac in fractions:
        f_str = str(frac)
        data = results[f_str]
        
        # Calculate mean across folds (n=5)
        m_ood = np.mean(data['ece_ood'])
        s_ood = np.std(data['ece_ood']) / np.sqrt(n_splits)
        ece_ood_mean.append(m_ood)
        ece_ood_std.append(s_ood)
        
        # For evaluation under deployment shift, we must use the calibrator fit on clean data
        # evaluated on shifted test data. In the pipeline, this is boot_clean.
        m_ts = np.mean([fold['ece_temp_mean'] for fold in data['boot_clean']])
        s_ts = np.std([fold['ece_temp_mean'] for fold in data['boot_clean']]) / np.sqrt(n_splits)
        ece_temp_mean.append(m_ts)
        ece_temp_std.append(s_ts)
        
        m_iso = np.mean([fold['ece_iso_mean'] for fold in data['boot_clean']])
        s_iso = np.std([fold['ece_iso_mean'] for fold in data['boot_clean']]) / np.sqrt(n_splits)
        ece_iso_mean.append(m_iso)
        ece_iso_std.append(s_iso)
        
        m_brier_ood = np.mean(data['brier_ood'])
        m_brier_ts = np.mean([fold.get('brier_temp_mean', 0) for fold in data['boot_clean']])
        s_brier_ts = np.std([fold.get('brier_temp_mean', 0) for fold in data['boot_clean']]) / np.sqrt(n_splits)
        m_brier_iso = np.mean([fold.get('brier_iso_mean', 0) for fold in data['boot_clean']])
        s_brier_iso = np.std([fold.get('brier_iso_mean', 0) for fold in data['boot_clean']]) / np.sqrt(n_splits)
        
        m_adapt_ood = np.mean(data['ece_adapt_ood'])
        m_adapt_ts = np.mean([fold.get('ece_adapt_temp_mean', 0) for fold in data['boot_clean']])
        s_adapt_ts = np.std([fold.get('ece_adapt_temp_mean', 0) for fold in data['boot_clean']]) / np.sqrt(n_splits)
        m_adapt_iso = np.mean([fold.get('ece_adapt_iso_mean', 0) for fold in data['boot_clean']])
        s_adapt_iso = np.std([fold.get('ece_adapt_iso_mean', 0) for fold in data['boot_clean']]) / np.sqrt(n_splits)
        
        dropout_pct = int(round((1 - frac) * 100))
        print(f"{dropout_pct}% Dropout ({frac:.1f}) | {m_ood:.3f}          | {m_ts:.3f} ± {s_ts:.3f} | {m_iso:.3f} ± {s_iso:.3f}")
        print(f"Brier             | {m_brier_ood:.3f}          | {m_brier_ts:.3f} ± {s_brier_ts:.3f} | {m_brier_iso:.3f} ± {s_brier_iso:.3f}")
        print(f"Adaptive ECE      | {m_adapt_ood:.3f}          | {m_adapt_ts:.3f} ± {s_adapt_ts:.3f} | {m_adapt_iso:.3f} ± {s_adapt_iso:.3f}")
        
    print("\nRegenerating Figure 1 (ece_degradation.png)...")
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
    
    ax.set_xlabel('Capture Efficiency Fraction (1.0 = Clean Reference)')
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title(f'Calibration Under Deployment Shift (frozen model, n={n_splits} holdout replicates)')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/ece_degradation.png', dpi=300)
    print("Saved dose-response figure to figures/ece_degradation.png")

if __name__ == "__main__":
    main()
