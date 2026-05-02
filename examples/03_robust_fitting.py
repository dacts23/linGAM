"""
Example 03 — Robust Fitting with Outliers
=========================================

Compare a standard least-squares fit vs. Huber IRLS robust fit
when the data contains extreme outliers.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import LinGAM

np.random.seed(42)

x = np.random.uniform(0, 10, 250)
y = np.sin(x) + np.random.normal(0, 0.2, len(x))

# Inject extreme outliers
y[20] += 9
y[80] -= 8
y[150] += 7

# Standard fit
model_std = LinGAM(n_splines=14)
model_std.fit(x, y)

# Robust fit
model_rob = LinGAM(n_splines=14)
model_rob.fit(x, y, robust=True)

print("=== Robust Fitting ===")
print(f"Standard RSS : {model_std.statistics_['rss']:.4f}")
print(f"Robust RSS   : {model_rob.statistics_['rss']:.4f}")
print(f"Standard GCV : {model_std.statistics_['GCV']:.4f}")
print(f"Robust GCV   : {model_rob.statistics_['GCV']:.4f}")

# --- Optional pyGAM comparison ---
try:
    from pygam import LinearGAM, s

    X = x.reshape(-1, 1)
    gam = LinearGAM(s(0, n_splines=14), lam=1.0).fit(X, y)
    y_pygam = gam.predict(X)

    print("\n--- pyGAM comparison (standard fit) ---")
    print(f"pyGAM EDF   : {gam.statistics_['edof']:.2f}")
    print(f"pyGAM GCV   : {gam.statistics_['GCV']:.4f}")
    print(f"Max |y_diff|: {np.max(np.abs(model_std.predict(x) - y_pygam)):.6f}")
except ImportError:
    print("\n(pyGAM not installed — skipping comparison)")

x_grid = np.linspace(0, 10, 500)

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(x, y, c='lightgray', s=15, label='Data (with outliers)')
ax.plot(x_grid, model_std.predict(x_grid), 'r--', lw=2, label='Standard fit')
ax.plot(x_grid, model_rob.predict(x_grid), 'g-', lw=2, label='Robust fit (Huber IRLS)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Standard vs Robust Fit')
ax.legend()
fig.tight_layout()
plt.show()

# --- Diagnostics: observed vs fitted & residuals vs fitted ---
for label, model in [("Standard", model_std), ("Robust", model_rob)]:
    y_fit = model.predict(x)
    residuals = y - y_fit

    fig_diag, axes_diag = plt.subplots(1, 2, figsize=(10, 4))
    axes_diag[0].scatter(y, y_fit, c='steelblue', edgecolors='k', s=20, alpha=0.6)
    ymin, ymax = y.min(), y.max()
    axes_diag[0].plot([ymin, ymax], [ymin, ymax], 'r--', lw=1.5, label='1:1 line')
    axes_diag[0].set_xlabel('Observed y')
    axes_diag[0].set_ylabel('Fitted y')
    axes_diag[0].set_title(f'{label} — Observed vs Fitted')
    axes_diag[0].legend()

    axes_diag[1].scatter(y_fit, residuals, c='steelblue', edgecolors='k', s=20, alpha=0.6)
    axes_diag[1].axhline(0, color='r', linestyle='--', lw=1.5)
    axes_diag[1].set_xlabel('Fitted y')
    axes_diag[1].set_ylabel('Residuals')
    axes_diag[1].set_title(f'{label} — Residuals vs Fitted')
    fig_diag.tight_layout()
    plt.show()
