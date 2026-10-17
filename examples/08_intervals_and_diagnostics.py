"""
Example 08 — Intervals & Diagnostics
=====================================

Compute confidence intervals, prediction intervals, and diagnostic
statistics (AIC, BIC, deviance explained, log-likelihood).
"""

import numpy as np
import matplotlib.pyplot as plt
from magli import LinGAM

np.random.seed(42)

x = np.random.uniform(0, 10, 300)
y = 2 * np.sin(x) + 0.5 * x + np.random.normal(0, 0.4, 300)

model = LinGAM(backend="eilers")
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

# --- Backend comparison: Eilers-Marx vs Wood (side-by-side) ---
# Same data and basis size, each backend's idiomatic tuning (GCV grid search
# vs REML). Compares fit quality plus the mean widths of the 95% confidence
# and prediction intervals both backends provide.
wood_model = LinGAM(k=model.n_splines, bs="cr", backend="wood")
wood_model.fit(x, y)
ci_wood = wood_model.confidence_intervals(x, width=0.95)
pi_wood = wood_model.prediction_intervals(x, width=0.95)
print("\n--- Backend comparison: Eilers-Marx (B-splines + GCV) vs Wood (cubic + REML) ---")
print(f"{'Metric':<18}{'Eilers-Marx':>12}{'Wood':>12}")
print(f"{'EDF':<18}{model.statistics_['edof']:>12.2f}{wood_model.statistics_['edof']:>12.2f}")
print(f"{'RSS':<18}{model.statistics_['rss']:>12.4f}{wood_model.statistics_['rss']:>12.4f}")
print(f"{'R^2':<18}{model.deviance_explained():>12.4f}{wood_model.deviance_explained():>12.4f}")
print(f"{'Mean CI width':<18}{ci_width:>12.4f}{((ci_wood[:, 1] - ci_wood[:, 0]).mean()):>12.4f}")
print(f"{'Mean PI width':<18}{pi_width:>12.4f}{((pi_wood[:, 1] - pi_wood[:, 0]).mean()):>12.4f}")

# --- Same-plot overlay: both fits with their 95% confidence bands ---
ci_wood_grid = wood_model.confidence_intervals(x_grid, width=0.95)
fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4))
ax_cmp.scatter(x, y, c='lightgray', s=10, label='Data')
ax_cmp.plot(x_grid, y_grid, 'b-', lw=2, label='Eilers-Marx backend')
ax_cmp.fill_between(x_grid, ci_grid[:, 0], ci_grid[:, 1], color='blue', alpha=0.15)
ax_cmp.plot(x_grid, wood_model.predict(x_grid), 'r-.', lw=2, label='Wood backend')
ax_cmp.fill_between(x_grid, ci_wood_grid[:, 0], ci_wood_grid[:, 1], color='red', alpha=0.15)
ax_cmp.set_xlabel('x')
ax_cmp.set_ylabel('y')
ax_cmp.set_title('Backend Overlay: fits with 95% CIs')
ax_cmp.legend()
fig_cmp.tight_layout()
plt.show()
