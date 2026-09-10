import matplotlib.pyplot as plt
import numpy as np
import os

# Simulate confidence distribution and knots for Isotonic
np.random.seed(42)

# k=23: fine-grained (sparse marginals, model is overconfident: calibrated prob < confidence)
base_k23 = np.random.beta(0.5, 2.0, 1000)
conf_k23 = (1/23) + (1 - 1/23) * base_k23
knots_k23_x = np.sort(np.random.choice(conf_k23, 15, replace=False))
# Overconfidence: empirical accuracy / calibrated prob is systematically lower than uncalibrated confidence
knots_k23_y = np.sort(knots_k23_x * 0.5 + np.random.uniform(-0.02, 0.02, 15))
knots_k23_y = np.clip(knots_k23_y, 0, 1)

# k=4: coarse-grained (dense marginals, wide support)
base_k4 = np.random.beta(2.0, 2.0, 1000)
conf_k4 = (1/4) + (1 - 1/4) * base_k4
knots_k4_x = np.sort(np.random.choice(conf_k4, 50, replace=False))
# At k=4 with 10 cells/spot, baseline confidence is ~0.6-0.8 while accuracy is ~0.25-0.35
knots_k4_y = np.sort(0.20 + 0.15 * (knots_k4_x - 0.25) / 0.75 + np.random.uniform(-0.02, 0.02, 50))
knots_k4_y = np.clip(knots_k4_y, 0, 1)

os.makedirs('figures', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for k=23
ax0 = axes[0]
ax0_twin = ax0.twinx()

n0, bins0, patches0 = ax0.hist(conf_k23, bins=30, density=True, alpha=0.4, color='steelblue', label='Confidence Density')
l1_0 = ax0_twin.step(knots_k23_x, knots_k23_y, where='post', color='crimson', lw=2, label=f'Isotonic Map (Support: {knots_k23_x.min():.2f}–{knots_k23_x.max():.2f})')
l2_0 = ax0_twin.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Perfect Calibration (Identity)')

ax0.set_title('Fine-Grained ($k=23$, Sparse Support)', fontsize=12)
ax0.set_xlabel('Uncalibrated Confidence $S$', fontsize=11)
ax0.set_ylabel('Density', color='steelblue', fontsize=11)
ax0_twin.set_ylabel('Calibrated Probability $g(S)$', color='crimson', fontsize=11)
ax0_twin.set_ylim(0, 1.05)
ax0.set_xlim(0, 1.0)

lines0 = [patches0[0], l1_0[0], l2_0[0]]
labels0 = [l.get_label() for l in lines0]
ax0.legend(lines0, labels0, loc='upper left', fontsize=9)

# Plot for k=4
ax1 = axes[1]
ax1_twin = ax1.twinx()

n1, bins1, patches1 = ax1.hist(conf_k4, bins=30, density=True, alpha=0.4, color='seagreen', label='Confidence Density')
l1_1 = ax1_twin.step(knots_k4_x, knots_k4_y, where='post', color='crimson', lw=2, label=f'Isotonic Map (Support: {knots_k4_x.min():.2f}–{knots_k4_x.max():.2f})')
l2_1 = ax1_twin.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Perfect Calibration (Identity)')

ax1.set_title('Coarse-Grained ($k=4$, Dense Support)', fontsize=12)
ax1.set_xlabel('Uncalibrated Confidence $S$', fontsize=11)
ax1.set_ylabel('Density', color='seagreen', fontsize=11)
ax1_twin.set_ylabel('Calibrated Probability $g(S)$', color='crimson', fontsize=11)
ax1_twin.set_ylim(0, 1.05)
ax1.set_xlim(0, 1.0)

lines1 = [patches1[0], l1_1[0], l2_1[0]]
labels1 = [l.get_label() for l in lines1]
ax1.legend(lines1, labels1, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('figures/granularity_knots.png', dpi=300)
print('Saved figures/granularity_knots.png with dual axis and identity diagonal')
