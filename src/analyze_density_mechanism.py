import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from netcal.metrics import ECE
import os

def beta_calibration(p, y):
    # A simple single-parameter parametric map for demonstration
    from scipy.optimize import minimize
    def loss(a):
        # Beta[a=b] calibration: logit(p') = a * logit(p)
        eps = 1e-7
        p_safe = np.clip(p, eps, 1-eps)
        logit_p = np.log(p_safe / (1 - p_safe))
        logit_cal = a * logit_p
        p_cal = 1 / (1 + np.exp(-logit_cal))
        p_cal = np.clip(p_cal, eps, 1-eps)
        return -np.sum(y * np.log(p_cal) + (1-y) * np.log(1-p_cal))
    
    res = minimize(loss, [1.0], bounds=[(0.01, 10)])
    a_opt = res.x[0]
    
    eps = 1e-7
    p_safe = np.clip(p, eps, 1-eps)
    logit_p = np.log(p_safe / (1 - p_safe))
    logit_cal = a_opt * logit_p
    return 1 / (1 + np.exp(-logit_cal)), a_opt

def g(s):
    # Fixed non-temperature miscalibration map
    return s**1.5

def run_experiment(seed=42):
    rng = np.random.RandomState(seed)
    
    # 1. Clean Calibration Split
    # Draw scores S ~ Beta(a1, b1)
    S_cal = rng.beta(a=2, b=5, size=5000)
    # Correctness is Bernoulli(g(S)), so P(Y=1|S) = g(S)
    Y_cal = rng.binomial(1, g(S_cal))
    
    # Fit Isotonic Regression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(S_cal, Y_cal)
    
    # Fit Parametric map (Beta[a=b])
    _, a_opt = beta_calibration(S_cal, Y_cal)
    
    # 2. Evaluate on Shifted Target Split
    # Shift only the marginal P(S), not P(Y|S)
    S_test = rng.beta(a=5, b=2, size=5000)
    Y_test = rng.binomial(1, g(S_test))
    
    # Apply fitted maps
    iso_preds = iso.predict(S_test)
    
    eps = 1e-7
    S_test_safe = np.clip(S_test, eps, 1-eps)
    logit_p = np.log(S_test_safe / (1 - S_test_safe))
    param_preds = 1 / (1 + np.exp(-a_opt * logit_p))
    
    # 3. Calculate ECE
    ece = ECE(bins=15)
    
    # The arrays passed to ECE must be shaped correctly
    ece_uncal = ece.measure(S_test, Y_test)
    ece_iso = ece.measure(iso_preds, Y_test)
    ece_param = ece.measure(param_preds, Y_test)
    
    print(f"Shifted Target ECE:")
    print(f"Uncalibrated: {ece_uncal:.4f}")
    print(f"Isotonic:     {ece_iso:.4f}")
    print(f"Parametric:   {ece_param:.4f}")
    
    # Plotting to illustrate
    plt.figure(figsize=(10, 4))
    
    s_grid = np.linspace(0, 1, 1000)
    plt.plot(s_grid, g(s_grid), 'k--', label="True P(Y|S) = g(S)")
    plt.plot(s_grid, iso.predict(s_grid), 'r-', label="Fitted Isotonic")
    
    logit_s = np.log(np.clip(s_grid, eps, 1-eps) / (1 - np.clip(s_grid, eps, 1-eps)))
    param_grid = 1 / (1 + np.exp(-a_opt * logit_s))
    plt.plot(s_grid, param_grid, 'b-', label="Fitted Parametric")
    
    plt.hist(S_cal, bins=30, density=True, alpha=0.3, color='gray', label="Clean P(S)")
    plt.hist(S_test, bins=30, density=True, alpha=0.3, color='orange', label="Shifted P(S)")
    
    plt.xlabel("Reliability Score S")
    plt.ylabel("P(Y|S) / Density")
    plt.legend()
    plt.title("Invariant P(Y|S) with Marginal Density Shift")
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/density_mechanism.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_experiment()
