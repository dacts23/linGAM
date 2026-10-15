"""
Example 11 — R mgcv Backend
============================

Demonstrates linGAM's native R mgcv backend:
1. Single-term GAM with automatic REML smoothing parameter estimation (no lambda grid search required).
2. Formatted R-style summary table with approximate F-tests and p-values (Wood 2013).
3. Using multiple basis types ('tp', 'cr', 'ps', 'cc').
4. Multi-term GAM with tensor product interaction te(0, 1) and parametric linear term l(2).
5. Seamless interoperability between pyGAM and mgcv backends.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import LinGAM, GAMCore

np.random.seed(42)

# ── 1. Single-term LinGAM with automatic REML ─────────────────────────────────
print("=" * 60)
print("1. Single-term LinGAM (REML auto-estimated smoothing parameter)")
print("=" * 60)

n = 200
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.15, n)

# In mgcv, lambda (sp) is estimated automatically during fit via REML
model = LinGAM(k=10, bs="cr", backend="mgcv")
model.fit(x, y)

# Print full R-style summary table
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

# ── 2. Comparing different mgcv bases ─────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Comparison across mgcv basis types ('tp', 'cr', 'ps', 'cc')")
print("=" * 60)

bases = ["tp", "cr", "ps", "cc"]
for b in bases:
    m = LinGAM(k=10, bs=b, backend="mgcv")
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

gam = GAMCore("te(0, 1, bs='cr', k=4) + l(2)", backend="mgcv")
gam.fit(X, y_multi)

print(gam.summary())

print("\nFinished Example 11 successfully!")
