"""
Example 09 — Plotting & Partial Dependence
===========================================

Visualise term contributions with partial_dependence and
decomposition plots for multi-term models.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import GAMCore

np.random.seed(42)
n = 400
x = np.column_stack([
    np.random.uniform(0, 10, n),
    np.random.uniform(0, 5, n),
])
y = (
    np.sin(x[:, 0])
    + 0.3 * x[:, 1]
    + np.random.normal(0, 0.2, n)
)

model = GAMCore("s(0, n_splines=14) + l(1)")
model.fit(x, y)

# Partial dependence: contribution of each term evaluated at the data
pdep_0 = model.partial_dependence(0, x)   # s(0)
pdep_1 = model.partial_dependence(1, x)   # l(1)

print("=== Partial Dependence ===")
print(f"s(0) mean contribution : {pdep_0.mean():.4f}")
print(f"s(0) std  contribution : {pdep_0.std():.4f}")
print(f"l(1) mean contribution : {pdep_1.mean():.4f}")
print(f"l(1) std  contribution : {pdep_1.std():.4f}")

# Decomposition plot with extrapolation shading
fig, axes = model.plot_decomposition(x, y, n_grid=200, extrap_frac=0.15)
plt.show()

# Manual partial dependence curve for term 0
x_grid_0 = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)
x_dummy = np.column_stack([
    x_grid_0,
    np.full_like(x_grid_0, x[:, 1].mean()),
])
pdep_curve = model.partial_dependence(0, x_dummy)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(x_grid_0, pdep_curve, 'b-', lw=2)
ax2.set_xlabel('Feature 0')
ax2.set_ylabel('Partial dependence')
ax2.set_title('Partial Dependence of s(0)')
fig2.tight_layout()
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
