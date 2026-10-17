"""
Example 05  -  2-D Tensor Interaction
====================================

A tensor-product interaction captures a non-additive relationship
between two predictors that univariate splines cannot represent.
"""

import numpy as np
import matplotlib.pyplot as plt
from magli import GAMCore

np.random.seed(42)
n = 600
x = np.column_stack([
    np.random.uniform(0, 10, n),
    np.random.uniform(0, 5, n),
])
y = (
    np.sin(x[:, 0]) * np.cos(x[:, 1])   # non-additive interaction
    + 0.5 * x[:, 0]
    + np.random.normal(0, 0.3, n)
)

# Additive model (no interaction)
model_add = GAMCore("s(0) + s(1)", backend="eilers")
model_add.fit(x, y)

# Tensor interaction model
model_te = GAMCore("s(0) + s(1) + te(0, 1, n_splines=6)", backend="eilers")
model_te.fit(x, y)
model_te.gridsearch(x, y, search_terms=["s"],lam_grids=[np.logspace(-3, 3, 7),np.logspace(-3, 3, 7)],n_splines_grids=[np.arange(4, 10), np.arange(4, 10)])

print("=== Tensor Interaction ===")
print(f"Additive  GCV : {model_add.statistics_['GCV']:.4f}")
print(f"Additive  R²  : {model_add.deviance_explained():.4f}")
print(f"Tensor    GCV : {model_te.statistics_['GCV']:.4f}")
print(f"Tensor    R²  : {model_te.deviance_explained():.4f}")

# Partial dependence of the interaction term
pdep_te = model_te.partial_dependence(2, x)
print(f"Interaction partial dependence shape: {pdep_te.shape}")

# Plot decomposition
fig, axes = model_te.plot_decomposition(x, y)
plt.show()

# --- Built-in diagnostics plot ---
model_te.plot_diagnostics(x, y)
plt.show()

# --- Backend comparison: Eilers-Marx vs Wood (side-by-side) ---
# Same tensor formula on both backends: Eilers-Marx uses B-splines with the GCV-tuned
# penalties from the grid search above; Wood builds a tensor product of cubic
# regression splines with one REML-estimated smoothing parameter per margin.
wood_te = GAMCore("s(0) + s(1) + te(0, 1, n_splines=6)", backend="wood")
wood_te.fit(x, y)
print("\n--- Backend comparison: tensor interaction on both backends ---")
print(f"{'Metric':<18}{'Eilers-Marx':>12}{'Wood':>12}")
print(f"{'EDF':<18}{model_te.statistics_['edof']:>12.2f}{wood_te.statistics_['edof']:>12.2f}")
print(f"{'GCV':<18}{model_te.statistics_['GCV']:>12.4f}{wood_te.statistics_['GCV']:>12.4f}")
print(f"{'R^2':<18}{model_te.deviance_explained():>12.4f}{wood_te.deviance_explained():>12.4f}")
print("(See 13_eilers_vs_wood_comparison.py for the full head-to-head with plots.)")
