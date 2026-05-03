"""
Example 08 — Intervals & Diagnostics
=====================================

Compute confidence intervals, prediction intervals, and diagnostic
statistics (AIC, BIC, deviance explained, log-likelihood).
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import LinGAM

np.random.seed(42)

x = np.random.uniform(0, 10, 300)
y = 2 * np.sin(x) + 0.5 * x + np.random.normal(0, 0.4, 300)

model = LinGAM()
model.gridsearch(x, y)

print("=== Diagnostics ===")
print(f"EDF              : {model.statistics_['edof']:.2f}")
print(f"RSS              : {model.statistics_['rss']:.4f}")
print(f"Scale (sigma_hat): {model.statistics_['scale']:.4f}")
print(f"GCV              : {model.statistics_['GCV']:.4f}")
print(f"Log-likelihood   : {model.loglikelihood():.2f}")
print(f"AIC              : {model.aic():.2f}")
print(f"BIC              : {model.bic():.2f}")
print(f"Deviance explained (R²) : {model.deviance_explained():.4f}")

# --- Optional pyGAM comparison ---
try:
    from pygam import LinearGAM, s

    X = x.reshape(-1, 1)
    gam = LinearGAM(s(0, n_splines=model.n_splines), lam=model.lam).fit(X, y)
    y_pygam = gam.predict(X)

    print("\n--- pyGAM comparison ---")
    print(f"pyGAM EDF       : {gam.statistics_['edof']:.2f}")
    print(f"pyGAM GCV       : {gam.statistics_['GCV']:.4f}")
    print(f"pyGAM AIC       : {gam.statistics_['AIC']:.2f}")
    print(f"pyGAM loglik    : {gam.statistics_['loglikelihood']:.2f}")
    print(f"Max |y_diff|    : {np.max(np.abs(model.predict(x) - y_pygam)):.6f}")
except ImportError:
    print("\n(pyGAM not installed — skipping comparison)")

# Interval widths at training points
ci = model.confidence_intervals(x, width=0.95)
pi = model.prediction_intervals(x, width=0.95)

ci_width = (ci[:, 1] - ci[:, 0]).mean()
pi_width = (pi[:, 1] - pi[:, 0]).mean()
print(f"\nMean 95% CI width : {ci_width:.4f}")
print(f"Mean 95% PI width : {pi_width:.4f}")

# Plot intervals on a fine grid
x_grid = np.linspace(0, 10, 500)
y_grid = model.predict(x_grid)
ci_grid = model.confidence_intervals(x_grid, width=0.95)
pi_grid = model.prediction_intervals(x_grid, width=0.95)

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(x, y, c='lightgray', s=10, label='Data')
ax.plot(x_grid, y_grid, 'b-', lw=2, label='Fit')
ax.fill_between(x_grid, ci_grid[:, 0], ci_grid[:, 1], color='blue', alpha=0.15, label='95% CI')
ax.fill_between(x_grid, pi_grid[:, 0], pi_grid[:, 1], color='blue', alpha=0.05, label='95% PI')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Confidence and Prediction Intervals')
ax.legend()
fig.tight_layout()
plt.show()

# --- Built-in diagnostics plot ---
fig_diag, axes_diag = model.plot_diagnostics(x, y)
plt.show()
