"""
Example 02 — Multi-Term Formula Interface
=========================================

GAMCore with a smooth term and a linear term.
Shows how the formula string maps features to model components.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import GAMCore

np.random.seed(42)
n = 400
x = np.column_stack([
    np.random.uniform(0, 10, n),   # feature 0 — smooth
    np.random.uniform(0, 5, n),    # feature 1 — linear
])
y = (
    np.sin(x[:, 0])
    + 0.4 * x[:, 1]
    + np.random.normal(0, 0.25, n)
)

# Fit a 2-term model
model = GAMCore("s(0, n_splines=12) + l(1)")
model.fit(x, y)

print("=== Multi-Term GAMCore ===")
print(f"EDF : {model.statistics_['edof']:.2f}")
print(f"GCV : {model.statistics_['GCV']:.4f}")
print(f"AIC : {model.aic():.2f}")
print(f"BIC : {model.bic():.2f}")
print(f"R²  : {model.deviance_explained():.4f}")

# Grid search to tune both terms
model.gridsearch(x, y)
print(f"\nAfter gridsearch:")
print(f"EDF : {model.statistics_['edof']:.2f}")
print(f"GCV : {model.statistics_['GCV']:.4f}")

# Decomposition plot
fig, axes = model.plot_decomposition(x, y)
plt.show()

# --- Built-in diagnostics plot ---
model.plot_diagnostics(x, y)
plt.show()

# --- Backend comparison: pyGAM vs R mgcv (side-by-side) ---
# Same formula parses for both backends (n_splines/k and lam/sp are aliases).
# pyGAM tunes via discrete GCV grid search (above); mgcv estimates smoothing
# automatically via REML with a single fit.
mgcv_model = GAMCore("s(0, n_splines=12) + l(1)", backend="mgcv")
mgcv_model.fit(x, y)
print("\n--- Backend comparison: pyGAM (GCV grid search) vs R mgcv (REML) ---")
print(f"{'Metric':<18}{'pyGAM':>12}{'mgcv':>12}")
print(f"{'EDF':<18}{model.statistics_['edof']:>12.2f}{mgcv_model.statistics_['edof']:>12.2f}")
print(f"{'RSS':<18}{model.statistics_['rss']:>12.4f}{mgcv_model.statistics_['rss']:>12.4f}")
print(f"{'R^2':<18}{model.deviance_explained():>12.4f}{mgcv_model.deviance_explained():>12.4f}")
print(f"{'AIC':<18}{model.aic():>12.2f}{mgcv_model.aic():>12.2f}")
print(f"{'BIC':<18}{model.bic():>12.2f}{mgcv_model.bic():>12.2f}")
print("(See 13_pygam_vs_mgcv_comparison.py for the full head-to-head with plots.)")

# --- Same-plot overlay: overall prediction sweeping feature 0 (feature 1 at median) ---
x_sweep = np.tile(np.median(x, axis=0), (200, 1))
x_sweep[:, 0] = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)
fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
ax_cmp.scatter(x[:, 0], y, c='lightgray', s=10, label='Data')
ax_cmp.plot(x_sweep[:, 0], model.predict(x_sweep), 'b-', lw=2, label='pyGAM backend')
ax_cmp.plot(x_sweep[:, 0], mgcv_model.predict(x_sweep), 'r-.', lw=2, label='mgcv backend')
ax_cmp.set_xlabel('Feature 0 (feature 1 at median)')
ax_cmp.set_ylabel('y')
ax_cmp.set_title('Backend Overlay: pyGAM vs mgcv')
ax_cmp.legend()
fig_cmp.tight_layout()
plt.show()
