"""
Example 05 — 2-D Tensor Interaction
====================================

A tensor-product interaction captures a non-additive relationship
between two predictors that univariate splines cannot represent.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import GAMCore

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
model_add = GAMCore("s(0) + s(1)")
model_add.fit(x, y)

# Tensor interaction model
model_te = GAMCore("s(0) + s(1) + te(0, 1, n_splines=6)")
model_te.fit(x, y)
model_te.gridsearch(x, y)

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

# --- Diagnostics: observed vs fitted & residuals vs fitted ---
y_fit = model_te.predict(x)
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
