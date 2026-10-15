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

# mgcv has no robust mode — fit its standard (outlier-sensitive) counterpart
# here so it can join the same axes: it should track the pyGAM standard fit.
mgcv_model = LinGAM(k=14, bs="cr", backend="mgcv")
mgcv_model.fit(x, y)

x_grid = np.linspace(0, 10, 500)

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(x, y, c='lightgray', s=15, label='Data (with outliers)')
ax.plot(x_grid, model_std.predict(x_grid), 'r--', lw=2, label='Standard fit')
ax.plot(x_grid, model_rob.predict(x_grid), 'g-', lw=2, label='Robust fit (Huber IRLS)')
ax.plot(x_grid, mgcv_model.predict(x_grid), 'b-.', lw=2, label='mgcv backend (std fit)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Standard vs Robust Fit')
ax.legend()
fig.tight_layout()
plt.show()

# --- Built-in diagnostics plot ---
for model in (model_std, model_rob):
    model.plot_diagnostics(x, y)
    plt.show()

# --- Backend comparison: pyGAM vs R mgcv (side-by-side) ---
# Huber IRLS robust fitting is a pyGAM-backend feature; the mgcv backend has
# no robust mode, so it can only run the standard (outlier-sensitive) fit.
# It should land near the pyGAM standard fit, far from the robust one.
# (mgcv standard fit already computed above for the overlay plot)
print("\n--- Backend comparison: robust is pyGAM-only; mgcv shown for reference ---")
print(f"{'Metric':<18}{'pyGAM std':>12}{'pyGAM robust':>14}{'mgcv std':>12}")
print(f"{'EDF':<18}{model_std.statistics_['edof']:>12.2f}{model_rob.statistics_['edof']:>14.2f}{mgcv_model.statistics_['edof']:>12.2f}")
print(f"{'RSS':<18}{model_std.statistics_['rss']:>12.4f}{model_rob.statistics_['rss']:>14.4f}{mgcv_model.statistics_['rss']:>12.4f}")
print(f"{'R^2':<18}{model_std.deviance_explained():>12.4f}{model_rob.deviance_explained():>14.4f}{mgcv_model.deviance_explained():>12.4f}")
