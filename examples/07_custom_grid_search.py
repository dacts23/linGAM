"""
Example 07 — Custom Grid Search
================================

Restrict grid search to specific terms and supply custom candidate grids.
"""

import numpy as np
import matplotlib.pyplot as plt
from gamkit import GAMCore

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

model = GAMCore("s(0) + s(1) + l(2)", backend="eilers")
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

# --- Built-in diagnostics plot ---
model.plot_diagnostics(x, y)
plt.show()

# --- Backend comparison: Eilers-Marx vs Wood (side-by-side) ---
# Tuning philosophies differ: Eilers-Marx searches discrete lam/n_splines grids with
# GCV (as above, including selective search_terms); Wood estimates smoothing
# continuously via REML in a single fit, and its gridsearch() only varies the
# basis dimension k. Same final complexity metrics on the same data:
wood_model = GAMCore("s(0) + s(1) + l(2)", backend="wood")
wood_model.fit(x, y)
wood_edf_before_ksearch = wood_model.statistics_['edof']
print("\n--- Backend comparison: Eilers-Marx (discrete GCV grids) vs Wood (REML) ---")
print(f"{'Metric':<18}{'Eilers-Marx':>12}{'Wood':>12}")
print(f"{'EDF':<18}{model.statistics_['edof']:>12.2f}{wood_edf_before_ksearch:>12.2f}")
print(f"{'GCV':<18}{model.statistics_['GCV']:>12.4f}{wood_model.statistics_['GCV']:>12.4f}")
print(f"{'R^2':<18}{model.deviance_explained():>12.4f}{wood_model.deviance_explained():>12.4f}")
wood_model.gridsearch(x, y, n_splines_grids=[np.arange(6, 16, 2), np.arange(6, 16, 2)])
print(f"{'Wood EDF after k-search':<18}{'':>12}{wood_model.statistics_['edof']:>12.2f}")

# --- Same-plot overlay: overall prediction sweeping feature 0 (others at median) ---
x_sweep = np.tile(np.median(x, axis=0), (200, 1))
x_sweep[:, 0] = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)
fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
ax_cmp.scatter(x[:, 0], y, c='lightgray', s=10, label='Data')
ax_cmp.plot(x_sweep[:, 0], model.predict(x_sweep), 'b-', lw=2, label='Eilers-Marx backend')
ax_cmp.plot(x_sweep[:, 0], wood_model.predict(x_sweep), 'r-.', lw=2, label='Wood backend')
ax_cmp.set_xlabel('Feature 0 (others at median)')
ax_cmp.set_ylabel('y')
ax_cmp.set_title('Backend Overlay: Eilers-Marx vs Wood')
ax_cmp.legend()
fig_cmp.tight_layout()
plt.show()
