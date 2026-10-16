"""
Example 01 — Basic LinGAM
=========================

Single smooth term with automatic GCV grid search.
Demonstrates fit, predict, confidence intervals, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from gamkit import LinGAM

np.random.seed(42)

# Synthetic smooth data
x = np.linspace(0, 1, 300)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))

# Fit with automatic grid search
model = LinGAM(backend="eilers")
model.gridsearch(x, y)

print("=== Basic LinGAM ===")
print(f"Best n_splines : {model.n_splines}")
print(f"Best lam       : {model.lam:.4f}")
print(f"EDF            : {model.statistics_['edof']:.2f}")
print(f"GCV            : {model.statistics_['GCV']:.4f}")
print(f"RSS            : {model.statistics_['rss']:.4f}")
print(f"Scale          : {model.statistics_['scale']:.4f}")

# Predictions
x_grid = np.linspace(-0.1, 1.1, 500)
y_pred = model.predict(x_grid)
ci = model.confidence_intervals(x_grid, width=0.95)
pi = model.prediction_intervals(x_grid, width=0.95)

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x, y, c='lightgray', s=10, label='Data')
ax.plot(x_grid, y_pred, 'b-', lw=2, label='Fit')
ax.fill_between(x_grid, ci[:, 0], ci[:, 1], color='blue', alpha=0.15, label='95% CI')
ax.fill_between(x_grid, pi[:, 0], pi[:, 1], color='blue', alpha=0.05, label='95% PI')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('LinGAM — Basic Fit with Intervals')
ax.legend(loc='upper right')
fig.tight_layout()
plt.show()

# Basis decomposition plot
fig2, axes = model.plot_decomposition(x, y)
plt.show()

# --- Built-in diagnostics plot ---
model.plot_diagnostics(x, y)
plt.show()

# --- Backend comparison: Eilers-Marx vs Wood (side-by-side) ---
# Same data, each backend's idiomatic tuning: Eilers-Marx uses a discrete GCV
# grid search (above), Wood estimates the smoothing parameter automatically
# via REML. Both should recover the same sine curve with similar EDF/RSS.
wood_model = LinGAM(k=model.n_splines, bs="cr", backend="wood")
wood_model.fit(x, y)
print("\n--- Backend comparison: Eilers-Marx (B-splines + GCV) vs Wood (cubic + REML) ---")
print(f"{'Metric':<22}{'Eilers-Marx':>12}{'Wood':>12}")
print(f"{'sp (lam)':<22}{model.lam:>12.4f}{wood_model.lam:>12.4f}")
print(f"{'EDF':<22}{model.statistics_['edof']:>12.2f}{wood_model.statistics_['edof']:>12.2f}")
print(f"{'RSS':<22}{model.statistics_['rss']:>12.4f}{wood_model.statistics_['rss']:>12.4f}")
print(f"{'R^2':<22}{model.deviance_explained():>12.4f}{wood_model.deviance_explained():>12.4f}")
print("(Eilers-Marx penalises B-spline second differences; Wood uses cubic regression")
print(" splines with a sum-to-zero constraint. See 13_eilers_vs_wood_comparison.py")
print(" for the full head-to-head with plots.)")

# --- Same-plot overlay: both backends on one axes ---
fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
ax_cmp.scatter(x, y, c='lightgray', s=10, label='Data')
ax_cmp.plot(x_grid, y_pred, 'b-', lw=2, label='Eilers-Marx backend')
ax_cmp.plot(x_grid, wood_model.predict(x_grid), 'r-.', lw=2, label='Wood backend')
ax_cmp.set_xlabel('x')
ax_cmp.set_ylabel('y')
ax_cmp.set_title('Backend Overlay: Eilers-Marx vs Wood')
ax_cmp.legend(loc='upper right')
fig_cmp.tight_layout()
plt.show()
