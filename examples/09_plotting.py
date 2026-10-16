"""
Example 09 — Plotting & Partial Dependence
===========================================

Visualise term contributions with partial_dependence and
decomposition plots for multi-term models.
"""

import numpy as np
import matplotlib.pyplot as plt
from gamkit import GAMCore

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

model = GAMCore("s(0, n_splines=14) + l(1)", backend="eilers")
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

# --- Built-in diagnostics plot ---
fig_diag, axes_diag = model.plot_diagnostics(x, y)
plt.show()

# --- Backend comparison: Eilers-Marx vs Wood (side-by-side) ---
# Same s() + l() formula on both backends (REML needs no grid search).
# Compares fit quality plus agreement of the term contributions: the mean
# absolute difference between the partial-dependence curves should be small
# next to the terms' own variation printed above.
wood_model = GAMCore("s(0, n_splines=14) + l(1)", backend="wood")
wood_model.fit(x, y)
print("\n--- Backend comparison: Eilers-Marx (GCV grid search) vs Wood (REML) ---")
print(f"{'Metric':<18}{'Eilers-Marx':>12}{'Wood':>12}")
print(f"{'EDF':<18}{model.statistics_['edof']:>12.2f}{wood_model.statistics_['edof']:>12.2f}")
print(f"{'RSS':<18}{model.statistics_['rss']:>12.4f}{wood_model.statistics_['rss']:>12.4f}")
print(f"{'R^2':<18}{model.deviance_explained():>12.4f}{wood_model.deviance_explained():>12.4f}")
for _ti in range(model.n_terms):
    _dp_em = model.partial_dependence(_ti, x)
    _dp_wd = wood_model.partial_dependence(_ti, x)
    # Center first: backends split intercept/centering differently, so only
    # the shape (not the level) is comparable across backends.
    _dp_em_c = _dp_em - _dp_em.mean()
    _dp_wd_c = _dp_wd - _dp_wd.mean()
    print(f"{'Centered |pdep_em-pdep_wd| term ' + str(_ti):<18}{np.mean(np.abs(_dp_em_c - _dp_wd_c)):>12.4f}")

# --- Same-plot overlay: centered term-0 partial dependence from both backends ---
pdep_wd_curve = wood_model.partial_dependence(0, x_dummy)
fig_cmp, ax_cmp = plt.subplots(figsize=(7, 4))
ax_cmp.plot(x_grid_0, pdep_curve - pdep_curve.mean(), 'b-', lw=2, label='Eilers-Marx backend')
ax_cmp.plot(x_grid_0, pdep_wd_curve - pdep_wd_curve.mean(), 'r-.', lw=2, label='Wood backend')
ax_cmp.set_xlabel('Feature 0')
ax_cmp.set_ylabel('Centered partial dependence')
ax_cmp.set_title('Backend Overlay: s(0) shape')
ax_cmp.legend()
fig_cmp.tight_layout()
plt.show()
