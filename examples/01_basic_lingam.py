"""
Example 01 — Basic LinGAM
=========================

Single smooth term with automatic GCV grid search.
Demonstrates fit, predict, confidence intervals, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import LinGAM

np.random.seed(42)

# Synthetic smooth data
x = np.linspace(0, 1, 300)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))

# Fit with automatic grid search
model = LinGAM()
model.gridsearch(x, y)

print("=== Basic LinGAM ===")
print(f"Best n_splines : {model.n_splines}")
print(f"Best lam       : {model.lam:.4f}")
print(f"EDF            : {model.statistics_['edof']:.2f}")
print(f"GCV            : {model.statistics_['GCV']:.4f}")
print(f"RSS            : {model.statistics_['rss']:.4f}")
print(f"Scale          : {model.statistics_['scale']:.4f}")

# --- Optional pyGAM comparison ---
try:
    from pygam import LinearGAM, s

    X = x.reshape(-1, 1)
    gam = LinearGAM(s(0, n_splines=model.n_splines), lam=model.lam).fit(X, y)
    y_pygam = gam.predict(X)

    print("\n--- pyGAM comparison ---")
    print(f"pyGAM EDF   : {gam.statistics_['edof']:.2f}")
    print(f"pyGAM GCV   : {gam.statistics_['GCV']:.4f}")
    print(f"pyGAM scale : {gam.statistics_['scale']:.4f}")
    print(f"Max |y_diff|: {np.max(np.abs(model.predict(x) - y_pygam)):.6f}")
except ImportError:
    print("\n(pyGAM not installed — skipping comparison)")

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

# --- Diagnostics: observed vs fitted & residuals vs fitted ---
y_fit = model.predict(x)
residuals = y - y_fit

fig_diag, axes_diag = plt.subplots(1, 2, figsize=(10, 4))
axes_diag[0].scatter(y, y_fit, c='steelblue', edgecolors='k', s=20, alpha=0.6)
ymin, ymax = y.min(), y.max()
axes_diag[0].plot([ymin, ymax], [ymin, ymax], 'r--', lw=1.5, label='1:1 line')
axes_diag[0].set_xlabel('Observed y')
axes_diag[0].set_ylabel('Fitted y')
axes_diag[0].set_title('Observed vs Fitted')
axes_diag[0].legend()

axes_diag[1].scatter(y_fit, residuals, c='steelblue', edgecolors='k', s=20, alpha=0.6)
axes_diag[1].axhline(0, color='r', linestyle='--', lw=1.5)
axes_diag[1].set_xlabel('Fitted y')
axes_diag[1].set_ylabel('Residuals')
axes_diag[1].set_title('Residuals vs Fitted')
fig_diag.tight_layout()
plt.show()
