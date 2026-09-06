import numpy as np
from sklearn.isotonic import IsotonicRegression

# Re-simulate or load the data exactly as in plot_knots.py to check bounds
np.random.seed(42)
N = 2000

# K=23
conf_k23 = np.random.beta(1, 5, N)
acc_k23 = conf_k23 * 0.8 + np.random.normal(0, 0.05, N)
iso_k23 = IsotonicRegression(out_of_bounds='clip')
iso_k23.fit(conf_k23, acc_k23)

print("K=23")
print(f"Confidence min: {conf_k23.min():.4f}, max: {conf_k23.max():.4f}")
print(f"1/k floor: {1/23:.4f}")
print(f"X_thresholds_ min: {iso_k23.X_thresholds_.min():.4f}, max: {iso_k23.X_thresholds_.max():.4f}")

# K=4
conf_k4 = np.random.beta(5, 2, N)
acc_k4 = conf_k4 * 0.9 + np.random.normal(0, 0.02, N)
iso_k4 = IsotonicRegression(out_of_bounds='clip')
iso_k4.fit(conf_k4, acc_k4)

print("\nK=4")
print(f"Confidence min: {conf_k4.min():.4f}, max: {conf_k4.max():.4f}")
print(f"1/k floor: {1/4:.4f}")
print(f"X_thresholds_ min: {iso_k4.X_thresholds_.min():.4f}, max: {iso_k4.X_thresholds_.max():.4f}")

# Check shifted predictions
conf_shifted_k23 = np.random.beta(1, 8, N)  # Shifted towards lower confidence
outside_support_k23 = np.sum((conf_shifted_k23 < iso_k23.X_thresholds_.min()) | (conf_shifted_k23 > iso_k23.X_thresholds_.max()))
print(f"\nK=23 Shifted outside support: {outside_support_k23} / {N} ({outside_support_k23/N*100:.2f}%)")

conf_shifted_k4 = np.random.beta(2, 5, N) 
outside_support_k4 = np.sum((conf_shifted_k4 < iso_k4.X_thresholds_.min()) | (conf_shifted_k4 > iso_k4.X_thresholds_.max()))
print(f"K=4 Shifted outside support: {outside_support_k4} / {N} ({outside_support_k4/N*100:.2f}%)")
