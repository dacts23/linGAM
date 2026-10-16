"""
Example 11 — Wood (2011) Backend
================================

Demonstrates gamkit's native Wood (2011) backend:
1. Single-term GAM with automatic REML smoothing parameter estimation (no lambda grid search required).
2. Formatted summary table with approximate F-tests and p-values (Wood 2013).
3. Using multiple basis types ('tp', 'cr', 'ps', 'cc').
4. Multi-term GAM with tensor product interaction te(0, 1) and parametric linear term l(2).
5. Seamless interoperability between Eilers-Marx and Wood backends.
"""

import numpy as np
import matplotlib.pyplot as plt
from gamkit import LinGAM, GAMCore

np.random.seed(42)

# ── 1. Single-term LinGAM with automatic REML ─────────────────────────────────
print("=" * 60)
print("1. Single-term LinGAM (REML auto-estimated smoothing parameter)")
print("=" * 60)

n = 200
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.15, n)

# In Wood backend, lambda (sp) is estimated automatically during fit via REML
model = LinGAM(k=10, bs="cr", backend="wood")
model.fit(x, y)

# Print full summary table
print(model.summary())

# Predict and compute Bayesian confidence and prediction intervals
x_grid = np.linspace(0, 1, 300)
y_pred = model.predict(x_grid)
ci = model.confidence_intervals(x_grid, width=0.95)
pi = model.prediction_intervals(x_grid, width=0.95)

print(f"Optimal estimated lambda (sp): {model.lam:.4f}")
print(f"Effective degrees of freedom:  {model.statistics_['edof']:.2f}")
print(f"Adjusted R-squared:            {model.statistics_['r_sq_adj']:.4f}")
print(f"Deviance explained:            {model.statistics_['deviance_explained'] * 100:.1f}%")

# ── 2. Comparing different Wood bases ─────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Comparison across Wood basis types ('tp', 'cr', 'ps', 'cc')")
print("=" * 60)

bases = ["tp", "cr", "ps", "cc"]
for b in bases:
    m = LinGAM(k=10, bs=b, backend="wood")
    m.fit(x, y)
    print(f"Basis bs='{b:2s}':  EDF = {m.statistics_['edof']:5.2f}  |  REML = {m.statistics_['score']:8.3f}  |  sp = {m.lam:8.4f}")

# ── 3. Multi-term GAM with Tensor Product and Linear Term ────────────────────
print("\n" + "=" * 60)
print("3. Multi-term formula with tensor interaction te(0, 1) + l(2)")
print("=" * 60)

x0 = np.random.uniform(0, 1, n)
x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(0, 1, n)
X = np.column_stack([x0, x1, x2])
y_multi = np.sin(np.pi * x0) * np.cos(np.pi * x1) + 2.5 * x2 + np.random.normal(0, 0.1, n)

gam = GAMCore("te(0, 1, bs='cr', k=4) + l(2)", backend="wood")
gam.fit(X, y_multi)

print(gam.summary())

print("\nFinished Example 11 successfully!")

# ── Backend comparison: Wood vs Eilers-Marx (side-by-side) ────────────────────
# The fits above are Wood (REML auto-tuning, summary table). The same data
# with the Eilers-Marx backend (B-splines + GCV grid search) for comparison.
print("\n" + "=" * 60)
print("Backend comparison: Wood (above) vs Eilers-Marx (below, same data)")
print("=" * 60)

em_model = LinGAM(backend="eilers")
em_model.gridsearch(x, y)
print("\n--- Single-term: Eilers-Marx (B-splines + GCV grid search) ---")
print(f"{'Metric':<22}{'Eilers-Marx':>12}{'Wood':>12}")
print(f"{'sp (lam)':<22}{em_model.lam:>12.4f}{model.lam:>12.4f}")
print(f"{'EDF':<22}{em_model.statistics_['edof']:>12.2f}{model.statistics_['edof']:>12.2f}")
print(f"{'RSS':<22}{em_model.statistics_['rss']:>12.4f}{model.statistics_['rss']:>12.4f}")
print(f"{'R^2':<22}{em_model.deviance_explained():>12.4f}{model.deviance_explained():>12.4f}")

em_gam = GAMCore("te(0, 1, n_splines=4) + l(2)", backend="eilers")
em_gam.gridsearch(X, y_multi)
print("\n--- Multi-term te(0, 1) + l(2): Eilers-Marx vs Wood ---")
print(f"{'Metric':<22}{'Eilers-Marx':>12}{'Wood':>12}")
print(f"{'EDF':<22}{em_gam.statistics_['edof']:>12.2f}{gam.statistics_['edof']:>12.2f}")
print(f"{'RSS':<22}{em_gam.statistics_['rss']:>12.4f}{gam.statistics_['rss']:>12.4f}")
print(f"{'R^2':<22}{em_gam.deviance_explained():>12.4f}{gam.deviance_explained():>12.4f}")
print("(See 13_eilers_vs_wood_comparison.py for the full head-to-head with plots.)")

# --- Same-plot overlay: single-term fits from both backends ---
fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
ax_cmp.scatter(x, y, c='lightgray', s=10, label='Data')
ax_cmp.plot(x_grid, em_model.predict(x_grid), 'b-', lw=2, label='Eilers-Marx backend')
ax_cmp.plot(x_grid, model.predict(x_grid), 'r-.', lw=2, label='Wood backend')
ax_cmp.set_xlabel('x')
ax_cmp.set_ylabel('y')
ax_cmp.set_title('Backend Overlay: Eilers-Marx vs Wood')
ax_cmp.legend()
fig_cmp.tight_layout()
plt.show()
