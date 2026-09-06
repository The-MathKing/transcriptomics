import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import json
import os
from scipy.optimize import minimize
from netcal.metrics import ECE

def temperature_scaling(logits, temp):
    scaled_logits = logits / temp
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def nll_loss(temp, logits, labels):
    p = temperature_scaling(logits, temp)
    # cross entropy
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    loss = -np.sum(labels * np.log(p)) / len(labels)
    return loss

def optimize_temp(logits, y_onehot):
    res = minimize(nll_loss, x0=[1.5], args=(logits, y_onehot), bounds=[(0.1, 5.0)])
    return res.x[0]

def generate_data(n_samples, alpha, seed=42):
    np.random.seed(seed)
    n_classes = len(alpha)
    
    # 1. Generate true probabilities (proportions)
    true_p = np.random.dirichlet(alpha, size=n_samples)
    
    # 2. Simulate overconfident predictions via non-linear distortion (not temperature scaling)
    # E.g., raise to power < 1 to flatten, or power > 1 to sharpen.
    # To create overconfidence: square the probabilities and re-normalize, 
    # making the dominant class even more dominant.
    pred_p = true_p ** 2.0 
    pred_p = pred_p / np.sum(pred_p, axis=1, keepdims=True)
    
    # Get logits for TS to fit on
    eps = 1e-9
    pred_logits = np.log(pred_p + eps)
    
    # 3. Generate ground truth labels by sampling from true_p
    y = np.array([np.random.choice(n_classes, p=p) for p in true_p])
    y_onehot = np.eye(n_classes)[y]
    
    conf = np.max(pred_p, axis=1)
    acc = (np.argmax(pred_p, axis=1) == y).astype(int)
    
    return pred_logits, pred_p, conf, acc, y_onehot

def main():
    n_classes = 5
    alpha_clean = np.ones(n_classes) * 1.5 
    alpha_shifted = np.ones(n_classes) * 0.1 
    
    ece_clean_raw_list, ece_clean_iso_list, ece_clean_t_list = [], [], []
    ece_shift_raw_list, ece_shift_iso_list, ece_shift_t_list = [], [], []
    
    n_seeds = 10
    for seed in range(n_seeds):
        # 1. Train/Calibration clean draw
        logits_cal, p_cal, conf_cal, acc_cal, y_cal = generate_data(10000, alpha_clean, seed=seed)
        
        # 2. Held-out clean draw (for testing clean performance)
        logits_clean_test, p_clean_test, conf_clean_test, acc_clean_test, _ = generate_data(10000, alpha_clean, seed=seed+100)
        
        # 3. Held-out shifted draw
        logits_test, p_test, conf_test, acc_test, y_test = generate_data(10000, alpha_shifted, seed=seed+200)
        
        # Fit Isotonic
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(conf_cal, acc_cal)
        
        # Fit Temp Scaling
        best_t = optimize_temp(logits_cal, y_cal)
        
        # Evaluate on Held-out Clean
        ece_clean_raw_list.append(ECE(bins=15).measure(conf_clean_test, acc_clean_test))
        conf_clean_iso = iso.predict(conf_clean_test)
        ece_clean_iso_list.append(ECE(bins=15).measure(conf_clean_iso, acc_clean_test))
        
        p_clean_t = temperature_scaling(logits_clean_test, best_t)
        conf_clean_t = np.max(p_clean_t, axis=1)
        ece_clean_t_list.append(ECE(bins=15).measure(conf_clean_t, acc_clean_test))
        
        # Evaluate on Shifted
        ece_shift_raw_list.append(ECE(bins=15).measure(conf_test, acc_test))
        conf_shift_iso = iso.predict(conf_test)
        ece_shift_iso_list.append(ECE(bins=15).measure(conf_shift_iso, acc_test))
        
        p_test_t = temperature_scaling(logits_test, best_t)
        conf_shift_t = np.max(p_test_t, axis=1)
        ece_shift_t_list.append(ECE(bins=15).measure(conf_shift_t, acc_test))
    
    print("--- Synthetic Ground Truth Check (10 seeds) ---")
    print(f"Clean ECE -> Raw: {np.mean(ece_clean_raw_list):.4f}±{np.std(ece_clean_raw_list):.4f} | Iso: {np.mean(ece_clean_iso_list):.4f}±{np.std(ece_clean_iso_list):.4f} | Temp: {np.mean(ece_clean_t_list):.4f}±{np.std(ece_clean_t_list):.4f}")
    print(f"Shift ECE -> Raw: {np.mean(ece_shift_raw_list):.4f}±{np.std(ece_shift_raw_list):.4f} | Iso: {np.mean(ece_shift_iso_list):.4f}±{np.std(ece_shift_iso_list):.4f} | Temp: {np.mean(ece_shift_t_list):.4f}±{np.std(ece_shift_t_list):.4f}")
    
    # Plot calibration curves (from the last seed)
    plt.figure(figsize=(15, 5))
    x_plot = np.linspace(0.2, 1.0, 100)
    plt.subplot(1, 3, 1)
    plt.title("Isotonic Mapping")
    plt.plot(x_plot, iso.predict(x_plot), label='Isotonic Fit (Clean)', color='blue')
    plt.xlabel('Uncalibrated Confidence')
    plt.ylabel('Calibrated Confidence')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(conf_cal, bins=20, alpha=0.5, label='Clean Density', density=True)
    plt.hist(conf_test, bins=20, alpha=0.5, label='Shifted Density', density=True)
    plt.title("Marginal Density Shift")
    plt.legend()
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/synthetic_ground_truth_check.png")
    print("Saved to figures/synthetic_ground_truth_check.png")

if __name__ == "__main__":
    main()
