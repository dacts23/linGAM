"""
Example 15 — Eilers-Marx Backend: y vs x Basis Decomposition & Visualizations
=============================================================================

Comprehensive demonstration of the Eilers-Marx (1996) P-spline backend
on a 1D non-linear relationship (y vs x):
1. Simulating data from Simon Wood's classic GAM test function with noise.
2. Fitting an Eilers-Marx LinGAM with automatic GCV grid search.
3. Bayesian confidence and prediction interval visualization.
4. Built-in and custom B-spline basis decomposition plots (unweighted and weighted).
5. Cumulative partial sums showing how localized bell-shaped B-splines assemble the curve.
6. Partition of unity: demonstrating that sum_j B_j(x) == 1 everywhere.
7. Penalty effect: second-difference regularization (lambda -> inf shrinks to a straight line).
8. Shape constraints: unconstrained vs mono_inc vs convex vs periodic.
9. Direct basis comparison: Eilers-Marx uniform B-splines vs Wood wavy orthogonal modes.
10. Built-in 4-panel diagnostic plots and backend comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from gamkit import LinGAM

np.random.seed(42)

# ── 1. Generate Data: Simon Wood's Classic GAM Benchmark Function ─────────────
# f(x) = 0.2 * x^11 * (10*(1-x))^6 + 10 * (10*x)^3 * (1-x)^10  (Wood 2006, 2017)
def wood_test_function(x: np.ndarray) -> np.ndarray:
    return 0.2 * (x**11) * (10 * (1 - x)) ** 6 + 10 * (10 * x) ** 3 * (1 - x) ** 10

n = 250
x = np.sort(np.random.uniform(0.0, 1.0, n))
y_true = wood_test_function(x)
sigma = 0.4
y = y_true + np.random.normal(0, sigma, n)

# ── 2. Fit Eilers-Marx LinGAM with GCV Grid Search ────────────────────────────
# Uses uniform B-splines with second-difference penalty (Eilers & Marx 1996).
# The smoothing parameter (lam) is selected by GCV grid search.
model = LinGAM(n_splines=10, backend="eilers")
model.gridsearch(x, y)

print("=" * 65)
print("Eilers-Marx LinGAM Model (Simon Wood's y vs x Function)")
print("=" * 65)
print(f"Optimal n_splines:          {model.n_splines}")
print(f"Optimal lambda (lam):       {model.lam:.5f}")
print(f"Effective degrees of freedom: {model.statistics_['edof']:.2f} / {model.n_splines}")
print(f"Generalized CV (GCV):       {model.statistics_['GCV']:.5f}")
print(f"Residual Sum of Squares:    {model.statistics_['rss']:.4f}")
print(f"Deviance explained (R^2):   {model.deviance_explained() * 100:.2f}%")
print(f"Estimated scale (sigma^2):  {model.statistics_['scale']:.4f} (true = {sigma**2:.4f})")

# ── 3. Plot 1: Model Fit with Confidence & Prediction Intervals ───────────────
x_grid = np.linspace(0, 1, 400)
y_pred = model.predict(x_grid)
ci = model.confidence_intervals(x_grid, width=0.95)
pi = model.prediction_intervals(x_grid, width=0.95)

fig1, ax1 = plt.subplots(figsize=(8.5, 4.8))
ax1.scatter(x, y, c="gray", alpha=0.5, s=18, edgecolors="none", label="Observations (y vs x)")
ax1.plot(x_grid, wood_test_function(x_grid), "k--", lw=1.8, alpha=0.75, label="True function f(x)")
ax1.plot(x_grid, y_pred, color="#1f77b4", lw=2.2, label=f"Eilers-Marx fit (EDF={model.statistics_['edof']:.2f})")
ax1.fill_between(x_grid, ci[:, 0], ci[:, 1], color="#1f77b4", alpha=0.25, label="95% Bayesian CI")
ax1.fill_between(x_grid, pi[:, 0], pi[:, 1], color="#1f77b4", alpha=0.10, label="95% Prediction Interval")
ax1.set_xlabel("Predictor x", fontsize=11)
ax1.set_ylabel("Response y", fontsize=11)
ax1.set_title("Plot 1: Eilers-Marx LinGAM Fit with Uncertainty Intervals", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right", frameon=True)
ax1.grid(True, linestyle=":", alpha=0.6)
fig1.tight_layout()
plt.show()

# ── 4. Plot 2: Built-in 3-Panel Basis Decomposition ───────────────────────────
# Decomposes the GAM into:
# 1) Evaluated uniform B-spline basis functions B_j(x) with knot locations
# 2) Estimated linear weights / coefficients beta_j and intercept beta_0
# 3) Scaled/weighted B-spline basis functions summing to the overall fit
fig2, axes2 = model.plot_decomposition(x, y, figsize=(15, 4.5))
plt.show()

# ── 5. Plot 3: Deep-Dive Component Decomposition & Cumulative Synthesis ───────
# Extract model matrix columns on the grid:
# y_pred(x) = beta_0 + sum_{j=1}^{k} beta_j * B_j(x)
X_grid = model._build_model_matrix(x_grid.reshape(-1, 1))
k_smooth = model.n_coefs
B_grid = X_grid[:, :k_smooth]
beta_smooth = model.coef_[:k_smooth]
intercept = model.coef_[-1]

cmap = plt.get_cmap("tab10")
colors = [cmap(i % 10) for i in range(k_smooth)]

fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9))

# Panel (a): Unweighted Uniform B-Spline Basis Functions
ax_a = axes3[0, 0]
for j in range(k_smooth):
    ax_a.plot(x_grid, B_grid[:, j], color=colors[j], lw=1.8, label=f"B_{j+1}")
if model.knots_ is not None:
    for i_k, knot in enumerate(model.knots_):
        ax_a.axvline(knot, color="gray", linestyle="--", lw=0.9, alpha=0.6,
                     label="Knots" if i_k == 0 else None)
ax_a.set_xlabel("x")
ax_a.set_ylabel("B_j(x)")
ax_a.set_title("(a) Unweighted B-Spline Basis Functions (Uniform Bumps)", fontweight="bold")
ax_a.grid(True, linestyle=":", alpha=0.5)
ax_a.legend(fontsize=7, ncol=3, loc="upper right")

# Panel (b): Weighted Basis Contributions (beta_j * B_j(x))
ax_b = axes3[0, 1]
for j in range(k_smooth):
    ax_b.plot(x_grid, beta_smooth[j] * B_grid[:, j], color=colors[j], lw=1.8, label=f"β_{j+1}*B_{j+1}")
ax_b.axhline(0, color="k", linestyle="--", lw=0.7, alpha=0.5)
ax_b.set_xlabel("x")
ax_b.set_ylabel("β_j * B_j(x)")
ax_b.set_title("(b) Weighted B-Spline Contributions (Scaled by β)", fontweight="bold")
ax_b.grid(True, linestyle=":", alpha=0.5)
ax_b.legend(fontsize=7, ncol=3, loc="upper right")

# Panel (c): Cumulative Assembly of the Curve
ax_c = axes3[1, 0]
ax_c.scatter(x, y, c="lightgray", alpha=0.4, s=12, label="Data")
ax_c.plot(x_grid, np.full_like(x_grid, intercept), "k:", lw=1.5, label="β_0 (Intercept)")

steps = [2, 5, 8, k_smooth]
step_colors = ["#9467bd", "#ff7f0e", "#2ca02c", "#d62728"]
for s, col in zip(steps, step_colors):
    partial_fit = intercept + B_grid[:, :s] @ beta_smooth[:s]
    ax_c.plot(x_grid, partial_fit, color=col, lw=2.0, label=f"β_0 + sum(j=1..{s})")
ax_c.set_xlabel("x")
ax_c.set_ylabel("Partial reconstruction")
ax_c.set_title("(c) Step-by-Step Curve Reconstruction", fontweight="bold")
ax_c.grid(True, linestyle=":", alpha=0.5)
ax_c.legend(fontsize=8, loc="upper right")

# Panel (d): Partition of Unity & Coefficient Bar Chart
ax_d = axes3[1, 1]
coef_labels = [f"B_{j+1}" for j in range(k_smooth)] + ["Intercept"]
coef_vals = np.append(beta_smooth, intercept)
bar_colors = colors + ["dimgray"]
ax_d.bar(range(len(coef_vals)), coef_vals, color=bar_colors, edgecolor="black", linewidth=0.6)
ax_d.axhline(0, color="k", lw=0.8)
ax_d.set_xticks(range(len(coef_vals)))
ax_d.set_xticklabels(coef_labels, fontsize=8)
ax_d.set_ylabel("Estimated Coefficient")
ax_d.set_title("(d) Estimated Spline Coefficients & Intercept", fontweight="bold")
ax_d.grid(True, linestyle=":", alpha=0.5, axis="y")

fig3.suptitle("Plot 3: Detailed Eilers-Marx B-Spline Decomposition & Synthesis", fontsize=14, fontweight="bold")
fig3.tight_layout()
plt.show()

# ── 6. Plot 4: Effect of Difference Penalty (lam) on B-Splines ────────────────
# Eilers-Marx penalises squared second differences: lam * ||Delta^2 beta||^2
# As lam -> inf, Delta^2 beta -> 0, which restricts the coefficients to an affine line!
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4.5))

lam_cases = [
    (1e-5, "Under-smoothed (lam=1e-5, noisy fluctuations)"),
    (model.lam, f"GCV Optimal (lam={model.lam:.4f}, balanced fit)"),
    (1e4, "Over-smoothed (lam=1e4, linear null space)"),
]

for idx, (lam_val, title) in enumerate(lam_cases):
    m_case = LinGAM(n_splines=10, lam=lam_val, backend="eilers")
    m_case.fit(x, y)
    y_case = m_case.predict(x_grid)
    
    X_case = m_case._build_model_matrix(x_grid.reshape(-1, 1))
    B_case = X_case[:, :m_case.n_coefs]
    b_coef = m_case.coef_[:m_case.n_coefs]
    ic_val = m_case.coef_[-1]
    
    ax = axes4[idx]
    ax.scatter(x, y, c="lightgray", alpha=0.4, s=10)
    for j in range(m_case.n_coefs):
        ax.plot(x_grid, b_coef[j] * B_case[:, j] + ic_val, alpha=0.45, lw=1)
    ax.plot(x_grid, y_case, "k-", lw=2.2, label=f"Fit (EDF={m_case.statistics_['edof']:.2f})")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

fig4.suptitle("Plot 4: Impact of Difference Penalty (lam) on Weighted B-Splines", fontsize=13, fontweight="bold")
fig4.tight_layout()
plt.show()

# ── 7. Plot 5: Shape Constraints (Eilers-Marx Exclusive Feature) ──────────────
# Eilers-Marx natively supports shape constraints via asymmetric P-spline penalties:
# - unconstrained
# - mono_inc: monotonic increasing
# - convex: positive second differences
# - periodic: circular boundary wrap
fig5, axes5 = plt.subplots(1, 4, figsize=(16, 4))
constraints = [
    (None, "Unconstrained", "#1f77b4"),
    ("mono_inc", "Monotonic Increasing", "#2ca02c"),
    ("convex", "Convex", "#ff7f0e"),
    ("periodic", "Periodic", "#9467bd"),
]

for idx, (cons, name, col) in enumerate(constraints):
    m_cons = LinGAM(n_splines=12, constraint=cons, backend="eilers")
    m_cons.fit(x, y)
    ax = axes5[idx]
    ax.scatter(x, y, c="gray", alpha=0.3, s=10)
    ax.plot(x_grid, wood_test_function(x_grid), "k--", lw=1.2, alpha=0.6, label="True f(x)")
    ax.plot(x_grid, m_cons.predict(x_grid), color=col, lw=2, label=f"Fit ({name})")
    ax.set_title(f"{name}\n(EDF={m_cons.statistics_['edof']:.1f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7)

fig5.suptitle("Plot 5: Shape-Constrained Eilers-Marx LinGAM Fits", fontsize=13, fontweight="bold")
fig5.tight_layout()
plt.show()

# ── 8. Plot 6: Visual Basis Comparison — Eilers-Marx vs Wood ──────────────────
# Direct visual comparison addressing why B-splines look uniform while Wood bases look wavy:
# - Eilers-Marx: raw B-splines are localized, uniform positive bumps (sum_j B_j(x) = 1)
# - Wood: QR null-space projection (Z matrix) absorbs sum-to-zero, forcing wavy zero-mean modes
wood_model = LinGAM(k=10, bs="ps", backend="wood")
wood_model.fit(x, y)

X_em = model._build_model_matrix(x_grid.reshape(-1, 1))[:, :model.n_coefs]
X_wd = wood_model._build_model_matrix(x_grid.reshape(-1, 1))[:, :wood_model._term.n_coefs]

fig6, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Top: Eilers-Marx B-spline basis functions
for j in range(X_em.shape[1]):
    ax_top.plot(x_grid, X_em[:, j], lw=1.8)
ax_top.set_ylabel("B_j(x)")
ax_top.set_title("Eilers-Marx Basis: Uniform Non-Negative B-Splines (pyGAM approach, sum_j B_j(x) = 1)",
                 fontweight="bold")
ax_top.grid(True, linestyle=":", alpha=0.5)

# Bottom: Wood's basis functions (sum-to-zero absorbed)
for j in range(X_wd.shape[1]):
    ax_bot.plot(x_grid, X_wd[:, j], lw=1.8)
ax_bot.axhline(0, color="k", linestyle="--", lw=0.8, alpha=0.6)
ax_bot.set_xlabel("Predictor x")
ax_bot.set_ylabel("b_j(x)")
ax_bot.set_title("Wood Basis ('ps'): Identifiability-Constrained Modes (sum b_j(x_i) = 0, wavy oscillations)",
                 fontweight="bold")
ax_bot.grid(True, linestyle=":", alpha=0.5)

fig6.suptitle("Plot 6: Why Wood Looks Wavy vs Eilers-Marx Uniform Bumps", fontsize=13, fontweight="bold")
fig6.tight_layout()
plt.show()

# ── 9. Plot 7: Built-in 4-Panel Model Diagnostics ─────────────────────────────
fig7, axes7 = model.plot_diagnostics(x, y, figsize=(10, 8.5))
plt.show()

# ── 10. Backend Comparison: Eilers-Marx vs Wood ───────────────────────────────
print("\n" + "=" * 65)
print("Backend Comparison: Eilers-Marx (GCV) vs Wood (REML)")
print("=" * 65)
print(f"{'Metric':<24}{'Eilers-Marx':>18}{'Wood':>18}")
print(f"{'Basis shape':<24}{'Uniform B-splines':>18}{'Wavy zero-sum modes':>18}")
print(f"{'Penalty type':<24}{'2nd differences':>18}{'Derivative penalty':>18}")
print(f"{'Optimal sp / lam':<24}{model.lam:>18.4f}{wood_model.lam:>18.4f}")
print(f"{'Effective DF (EDF)':<24}{model.statistics_['edof']:>18.2f}{wood_model.statistics_['edof']:>18.2f}")
print(f"{'Residual SS (RSS)':<24}{model.statistics_['rss']:>18.4f}{wood_model.statistics_['rss']:>18.4f}")
print(f"{'R-squared':<24}{model.deviance_explained():>18.4f}{wood_model.deviance_explained():>18.4f}")

fig8, ax8 = plt.subplots(figsize=(8.5, 4.5))
ax8.scatter(x, y, c="lightgray", alpha=0.5, s=14, label="Data")
ax8.plot(x_grid, wood_test_function(x_grid), "k--", lw=1.5, alpha=0.7, label="True f(x)")
ax8.plot(x_grid, model.predict(x_grid), "b-", lw=2, label=f"Eilers-Marx (EDF={model.statistics_['edof']:.2f})")
ax8.plot(x_grid, wood_model.predict(x_grid), "r-.", lw=2, label=f"Wood (EDF={wood_model.statistics_['edof']:.2f})")
ax8.set_xlabel("x")
ax8.set_ylabel("y")
ax8.set_title("Plot 8: Backend Overlay — Eilers-Marx vs Wood", fontsize=12, fontweight="bold")
ax8.grid(True, linestyle=":", alpha=0.5)
ax8.legend(loc="upper right")
fig8.tight_layout()
plt.show()

print("\nExample 15 finished successfully.")
