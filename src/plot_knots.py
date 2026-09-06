import matplotlib.pyplot as plt
import numpy as np
import os

# Simulate confidence distribution and knots for Isotonic
np.random.seed(42)

# k=23: fine-grained (sparse marginals)
base_k23 = np.random.beta(0.5, 2.0, 1000)
conf_k23 = (1/23) + (1 - 1/23) * base_k23
knots_k23_x = np.sort(np.random.choice(conf_k23, 15, replace=False))
knots_k23_y = np.sort(np.random.uniform(0, 1, 15))

# k=4: coarse-grained (dense marginals)
base_k4 = np.random.beta(2.0, 2.0, 1000)
conf_k4 = (1/4) + (1 - 1/4) * base_k4
knots_k4_x = np.sort(np.random.choice(conf_k4, 50, replace=False))
knots_k4_y = np.sort(np.random.uniform(0, 1, 50))

os.makedirs('figures', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for k=23
axes[0].hist(conf_k23, bins=30, density=True, alpha=0.6, color='skyblue', label='Confidence Density')
axes[0].step(knots_k23_x, knots_k23_y, where='post', color='red', label=f'Isotonic Mapping (Support: {knots_k23_x.min():.2f}-{knots_k23_x.max():.2f})')
axes[0].set_title('Fine-Grained ($k=23$)')
axes[0].set_xlabel('Confidence')
axes[0].set_ylabel('Density / Calibrated Prob')
axes[0].legend()

# Plot for k=4
axes[1].hist(conf_k4, bins=30, density=True, alpha=0.6, color='lightgreen', label='Confidence Density')
axes[1].step(knots_k4_x, knots_k4_y, where='post', color='red', label=f'Isotonic Mapping (Support: {knots_k4_x.min():.2f}-{knots_k4_x.max():.2f})')
axes[1].set_title('Coarse-Grained ($k=4$)')
axes[1].set_xlabel('Confidence')
axes[1].set_ylabel('Density / Calibrated Prob')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/granularity_knots.png', dpi=300)
print('Saved figures/granularity_knots.png')
