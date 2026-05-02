"""
Example 07 — Custom Grid Search
================================

Restrict grid search to specific terms and supply custom candidate grids.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import GAMCore

np.random.seed(42)
n = 400
x = np.column_stack([
    np.random.uniform(0, 10, n),
    np.random.uniform(0, 5, n),
    np.random.uniform(0, 3, n),
])
y = (
    np.sin(x[:, 0])
    + 0.3 * x[:, 1]
    + 0.2 * x[:, 2]**2
    + np.random.normal(0, 0.2, n)
)

model = GAMCore("s(0) + s(1) + l(2)")
model.fit(x, y)

print("=== Before grid search ===")
for i, term in enumerate(model._terms):
    print(f"  Term {i}: {term}")

# 1. Search only the first term
print("\n--- Search term 0 only ---")
model.gridsearch(x, y, search_terms=[0])
print(f"GCV : {model.statistics_['GCV']:.4f}")

# 2. Search only spline terms by type
print("\n--- Search all 's' terms ---")
model.gridsearch(x, y, search_terms=['s'])
print(f"GCV : {model.statistics_['GCV']:.4f}")

# 3. Custom grids — coarse lambda, fine n_splines
print("\n--- Custom grids ---")
model.gridsearch(
    x, y,
    lam_grids=[
        np.logspace(-2, 1, 4),   # s(0) — 4 lambdas
        np.logspace(-2, 1, 4),   # s(1)
        np.array([0.0]),          # l(2) — fixed
    ],
    n_splines_grids=[
        np.arange(6, 16, 2),     # s(0) — 5 values
        np.arange(6, 16, 2),     # s(1)
    ],
)
print(f"GCV : {model.statistics_['GCV']:.4f}")
print(f"EDF : {model.statistics_['edof']:.2f}")

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
