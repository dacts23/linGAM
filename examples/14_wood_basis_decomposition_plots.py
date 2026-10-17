"""
Example 14  -  Wood Backend: y vs x Basis Decomposition & Visualizations
=====================================================================

Comprehensive demonstration of Wood's (2011) penalized GAM on a 1D
non-linear relationship (y vs x):
1. Simulating data from Simon Wood's classic GAM test function with noise.
2. Fitting a Wood LinGAM with automatic REML smoothing parameter estimation.
3. Bayesian confidence and prediction interval visualization.
4. Built-in and custom GAM basis decomposition plots (unweighted and weighted).
5. Cumulative partial sums showing how individual basis functions assemble the fit.
6. Penalty effect: comparing under-smoothed, REML optimal, and over-smoothed fits.
7. Comparing Wood basis types ('cr', 'tp', 'ps') on the same data.
8. Built-in 4-panel diagnostic plots.
9. Backend comparison: Wood (REML) vs Eilers-Marx (GCV grid search).
"""

import numpy as np
import matplotlib.pyplot as plt
from magli import LinGAM

np.random.seed(42)

# ── 1. Generate Data: Simon Wood's Classic GAM Benchmark Function ─────────────
# f(x) = 0.2 * x^11 * (10*(1-x))^6 + 10 * (10*x)^3 * (1-x)^10  (Wood 2006, 2017)
# A classic non-linear curve featuring two peaks of distinct widths and heights.
def wood_test_function(x: np.ndarray) -> np.ndarray:
    return 0.2 * (x**11) * (10 * (1 - x)) ** 6 + 10 * (10 * x) ** 3 * (1 - x) ** 10

n = 250
x = np.sort(np.random.uniform(0.0, 1.0, n))
y_true = wood_test_function(x)
sigma = 0.4
y = y_true + np.random.normal(0, sigma, n)

# ── 2. Fit Wood LinGAM with Automatic REML ─────────────────────────────────────
# Uses cubic regression splines (bs="cr") with k=10 basis functions.
# The smoothing parameter (sp / lam) is automatically selected via REML.
model = LinGAM(k=10, bs="cr", backend="wood")
model.fit(x, y)

print("=" * 65)
print("Wood LinGAM Model Summary (Simon Wood's y vs x Function)")
print("=" * 65)
print(model.summary())
print(f"Optimal estimated lambda (sp): {model.lam:.5f}")
print(f"Effective degrees of freedom:  {model.statistics_['edof']:.2f} / {model.n_splines}")
print(f"Deviance explained:            {model.deviance_explained() * 100:.2f}%")
print(f"Adjusted R-squared:            {model.statistics_['r_sq_adj']:.4f}")
print(f"Estimated scale (sigma^2):     {model.statistics_['scale']:.4f} (true = {sigma**2:.4f})")

# ── 3. Plot 1: Model Fit with Confidence & Prediction Intervals ───────────────
x_grid = np.linspace(0, 1, 400)
y_pred = model.predict(x_grid)
ci = model.confidence_intervals(x_grid, width=0.95)
pi = model.prediction_intervals(x_grid, width=0.95)

fig1, ax1 = plt.subplots(figsize=(8.5, 4.8))
ax1.scatter(x, y, c="gray", alpha=0.5, s=18, edgecolors="none", label="Observations (y vs x)")
ax1.plot(x_grid, wood_test_function(x_grid), "k--", lw=1.8, alpha=0.75, label="True function f(x)")
ax1.plot(x_grid, y_pred, color="#1f77b4", lw=2.2, label=f"Wood GAM fit (EDF={model.statistics_['edof']:.2f})")
ax1.fill_between(x_grid, ci[:, 0], ci[:, 1], color="#1f77b4", alpha=0.25, label="95% Bayesian CI")
ax1.fill_between(x_grid, pi[:, 0], pi[:, 1], color="#1f77b4", alpha=0.10, label="95% Prediction Interval")
ax1.set_xlabel("Predictor x", fontsize=11)
ax1.set_ylabel("Response y", fontsize=11)
ax1.set_title("Plot 1: Wood LinGAM Fit with Uncertainty Intervals", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right", frameon=True)
ax1.grid(True, linestyle=":", alpha=0.6)
fig1.tight_layout()
plt.show()

# ── 4. Plot 2: Built-in 3-Panel Basis Decomposition ───────────────────────────
# Decomposes the GAM into:
# 1) Evaluated basis functions b_j(x) (sum-to-zero identifiability constrained)
# 2) Estimated linear weights / coefficients beta_j and intercept beta_0
# 3) Scaled/weighted basis functions summing to the overall fit
fig2, axes2 = model.plot_decomposition(x, y, show_weighted=True, figsize=(15, 4.5))
plt.show()

# ── 5. Plot 3: Deep-Dive Component Decomposition & Cumulative Assembly ────────
# Extract model matrix columns on the grid to inspect individual basis contributions:
# y_pred(x) = beta_0 + sum_{j=1}^{k-1} beta_j * b_j(x)
X_grid = model._build_model_matrix(x_grid.reshape(-1, 1))
k_smooth = model._term.n_coefs
B_grid = X_grid[:, :k_smooth]
beta_smooth = model.coef_[:k_smooth]
intercept = model.coef_[-1]

cmap = plt.get_cmap("tab10")
colors = [cmap(i % 10) for i in range(k_smooth)]

fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9))

# Panel (a): Unweighted Basis Functions
ax_a = axes3[0, 0]
for j in range(k_smooth):
    ax_a.plot(x_grid, B_grid[:, j], color=colors[j], lw=1.6, label=f"b_{j+1}")
ax_a.axhline(0, color="k", linestyle="--", lw=0.7, alpha=0.5)
ax_a.set_xlabel("x")
ax_a.set_ylabel("b_j(x)")
ax_a.set_title("(a) Unweighted Wood Basis Functions ('cr', sum-to-zero)", fontweight="bold")
ax_a.grid(True, linestyle=":", alpha=0.5)
ax_a.legend(fontsize=7, ncol=3, loc="upper right")

# Panel (b): Weighted Basis Functions (beta_j * b_j(x))
ax_b = axes3[0, 1]
for j in range(k_smooth):
    ax_b.plot(x_grid, beta_smooth[j] * B_grid[:, j], color=colors[j], lw=1.8, label=f"β_{j+1}*b_{j+1}")
ax_b.axhline(0, color="k", linestyle="--", lw=0.7, alpha=0.5)
ax_b.set_xlabel("x")
ax_b.set_ylabel("β_j * b_j(x)")
ax_b.set_title("(b) Weighted Basis Contributions (Scaled by β)", fontweight="bold")
ax_b.grid(True, linestyle=":", alpha=0.5)
ax_b.legend(fontsize=7, ncol=3, loc="upper right")

# Panel (c): Cumulative Assembly of the Curve
ax_c = axes3[1, 0]
ax_c.scatter(x, y, c="lightgray", alpha=0.4, s=12, label="Data")
# Baseline intercept
ax_c.plot(x_grid, np.full_like(x_grid, intercept), "k:", lw=1.5, label="β_0 (Intercept)")

steps = [2, 4, 7, k_smooth]
step_colors = ["#9467bd", "#ff7f0e", "#2ca02c", "#d62728"]
for s, col in zip(steps, step_colors):
    partial_fit = intercept + B_grid[:, :s] @ beta_smooth[:s]
    ax_c.plot(x_grid, partial_fit, color=col, lw=2.0, label=f"β_0 + sum(j=1..{s})")
ax_c.set_xlabel("x")
ax_c.set_ylabel("Partial reconstruction")
ax_c.set_title("(c) Step-by-Step Curve Reconstruction", fontweight="bold")
ax_c.grid(True, linestyle=":", alpha=0.5)
ax_c.legend(fontsize=8, loc="upper right")

# Panel (d): Bar Chart of Coefficients
ax_d = axes3[1, 1]
coef_labels = [f"β_{j+1}" for j in range(k_smooth)] + ["β_0\n(Intercept)"]
coef_vals = np.append(beta_smooth, intercept)
bar_colors = colors + ["dimgray"]
ax_d.bar(range(len(coef_vals)), coef_vals, color=bar_colors, edgecolor="black", linewidth=0.6)
ax_d.axhline(0, color="k", lw=0.8)
ax_d.set_xticks(range(len(coef_vals)))
ax_d.set_xticklabels(coef_labels, fontsize=8)
ax_d.set_ylabel("Coefficient Value")
ax_d.set_title("(d) Estimated Spline Coefficients & Intercept", fontweight="bold")
ax_d.grid(True, linestyle=":", alpha=0.5, axis="y")

fig3.suptitle("Plot 3: Detailed Wood GAM Basis Decomposition & Synthesis", fontsize=14, fontweight="bold")
fig3.tight_layout()
plt.show()

# ── 6. Plot 4: Effect of the Smoothing Parameter (sp) on Basis Decomposition ──
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4.5))

sp_cases = [
    (1e-5, "Under-smoothed (sp=1e-5, low penalty, wiggles)"),
    (model.lam, f"REML Optimal (sp={model.lam:.4f}, balanced)"),
    (1e3, "Over-smoothed (sp=1000, high penalty, linear)"),
]

for idx, (sp_val, title) in enumerate(sp_cases):
    m_case = LinGAM(k=10, bs="cr", sp=sp_val, backend="wood")
    m_case.fit(x, y)
    y_case = m_case.predict(x_grid)
    
    # Compute basis components
    X_case = m_case._build_model_matrix(x_grid.reshape(-1, 1))
    B_case = X_case[:, :m_case._term.n_coefs]
    b_coef = m_case.coef_[:m_case._term.n_coefs]
    ic_val = m_case.coef_[-1]
    
    ax = axes4[idx]
    ax.scatter(x, y, c="lightgray", alpha=0.4, s=10)
    # Plot weighted basis lines
    for j in range(m_case._term.n_coefs):
        ax.plot(x_grid, b_coef[j] * B_case[:, j] + ic_val, alpha=0.45, lw=1)
    ax.plot(x_grid, y_case, "k-", lw=2.2, label=f"Fit (EDF={m_case.statistics_['edof']:.2f})")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

fig4.suptitle("Plot 4: Impact of Penalty (sp) on Weighted Basis Functions and Overall Fit", fontsize=13, fontweight="bold")
fig4.tight_layout()
plt.show()

# ── 7. Plot 5: Comparison of Wood Basis Types ('cr', 'tp', 'ps') ─────────────
fig5, axes5 = plt.subplots(2, 3, figsize=(15, 8))
basis_types = [
    ("cr", "Cubic Regression Spline ('cr')"),
    ("tp", "Thin-Plate Spline ('tp')"),
    ("ps", "P-Spline ('ps')"),
]

for col_idx, (bs_type, bs_name) in enumerate(basis_types):
    m_bs = LinGAM(k=10, bs=bs_type, backend="wood")
    m_bs.fit(x, y)
    
    X_bs = m_bs._build_model_matrix(x_grid.reshape(-1, 1))
    n_c = m_bs._term.n_coefs
    B_bs = X_bs[:, :n_c]
    
    # Top row: Unweighted Basis Functions
    ax_top = axes5[0, col_idx]
    for j in range(n_c):
        ax_top.plot(x_grid, B_bs[:, j], lw=1.5)
    ax_top.axhline(0, color="k", linestyle="--", lw=0.6, alpha=0.5)
    ax_top.set_title(f"{bs_name}\nUnweighted Basis Functions", fontsize=10, fontweight="bold")
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("b_j(x)")
    ax_top.grid(True, linestyle=":", alpha=0.5)
    
    # Bottom row: Model Fit & Confidence Band
    ax_bot = axes5[1, col_idx]
    ci_bs = m_bs.confidence_intervals(x_grid, width=0.95)
    ax_bot.scatter(x, y, c="gray", alpha=0.35, s=10)
    ax_bot.plot(x_grid, wood_test_function(x_grid), "k--", lw=1.2, alpha=0.7, label="True f(x)")
    ax_bot.plot(x_grid, m_bs.predict(x_grid), "r-", lw=2, label=f"Fit (EDF={m_bs.statistics_['edof']:.2f})")
    ax_bot.fill_between(x_grid, ci_bs[:, 0], ci_bs[:, 1], color="red", alpha=0.2, label="95% CI")
    ax_bot.set_title(f"Fit (sp={m_bs.lam:.4f}, REML={m_bs.statistics_['score']:.2f})", fontsize=10)
    ax_bot.set_xlabel("x")
    ax_bot.set_ylabel("y")
    ax_bot.grid(True, linestyle=":", alpha=0.5)
    ax_bot.legend(loc="upper right", fontsize=8)

fig5.suptitle("Plot 5: Comparison of Wood Spline Bases on y vs x", fontsize=13, fontweight="bold")
fig5.tight_layout()
plt.show()

# ── 8. Plot 6: Built-in 4-Panel Model Diagnostics ─────────────────────────────
fig6, axes6 = model.plot_diagnostics(x, y, figsize=(10, 8.5))
plt.show()

# ── 9. Backend Comparison: Wood vs Eilers-Marx ────────────────────────────────
# Compare the Wood backend (automatic REML) against the Eilers-Marx backend
# (B-splines + GCV grid search) on the same dataset.
em_model = LinGAM(backend="eilers")
em_model.gridsearch(x, y)

print("\n" + "=" * 65)
print("Backend Comparison: Wood (REML) vs Eilers-Marx (GCV grid search)")
print("=" * 65)
print(f"{'Metric':<24}{'Eilers-Marx':>18}{'Wood':>18}")
print(f"{'Basis type':<24}{'B-splines':>18}{'Cubic Spline (cr)':>18}")
print(f"{'Tuning criterion':<24}{'GCV grid search':>18}{'REML Newton':>18}")
print(f"{'Optimal sp / lam':<24}{em_model.lam:>18.4f}{model.lam:>18.4f}")
print(f"{'Effective DF (EDF)':<24}{em_model.statistics_['edof']:>18.2f}{model.statistics_['edof']:>18.2f}")
print(f"{'Residual SS (RSS)':<24}{em_model.statistics_['rss']:>18.4f}{model.statistics_['rss']:>18.4f}")
print(f"{'R-squared':<24}{em_model.deviance_explained():>18.4f}{model.deviance_explained():>18.4f}")
print(f"{'Estimated sigma':<24}{np.sqrt(em_model.statistics_['scale']):>18.4f}{np.sqrt(model.statistics_['scale']):>18.4f}")

fig7, ax7 = plt.subplots(figsize=(8.5, 4.5))
ax7.scatter(x, y, c="lightgray", alpha=0.5, s=14, label="Data")
ax7.plot(x_grid, wood_test_function(x_grid), "k--", lw=1.5, alpha=0.7, label="True f(x)")
ax7.plot(x_grid, em_model.predict(x_grid), "b-", lw=2, label=f"Eilers-Marx (EDF={em_model.statistics_['edof']:.2f})")
ax7.plot(x_grid, model.predict(x_grid), "r-.", lw=2, label=f"Wood (EDF={model.statistics_['edof']:.2f})")
ax7.set_xlabel("x")
ax7.set_ylabel("y")
ax7.set_title("Plot 7: Backend Overlay  -  Wood vs Eilers-Marx Fits", fontsize=12, fontweight="bold")
ax7.grid(True, linestyle=":", alpha=0.5)
ax7.legend(loc="upper right")
fig7.tight_layout()
plt.show()

print("\nExample 14 finished successfully.")
