"""
Example 13  -  Head-to-Head Comparison: Eilers-Marx vs Wood Philosophies
======================================================================

Demonstrates and contrasts the two GAM philosophies implemented in magli:

1. Univariate (y vs x):
   - Eilers-Marx (1996): B-splines with second-difference penalty and GCV grid search.
   - Wood (2011): Cubic regression splines ('cr') with null-space sum-to-zero constraint
     and automatic REML smoothing parameter optimization.

2. Multivariate (y vs x1, x2, x3):
   - Compares additive decomposition, effective degrees of freedom (EDF),
     RSS, and deviance explained on a multi-feature dataset.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from magli import LinGAM, GAMCore

def run_comparison():
    print("=" * 78)
    print("  magli Comparison: Eilers-Marx Backend vs Wood Backend")
    print("=" * 78)

    np.random.seed(42)

    # =========================================================================
    # PART 1: Univariate (y vs x)
    # =========================================================================
    print("\n" + "-" * 78)
    print("  PART 1: Univariate Fit (y vs x)")
    print("-" * 78)

    n_samples = 150
    x = np.sort(np.random.uniform(0, 1, n_samples))
    y_true = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    noise = np.random.normal(0, 0.2, n_samples)
    y = y_true + noise

    # Grid for smooth curve predictions
    x_grid = np.linspace(0, 1, 300)
    y_grid_true = np.sin(2 * np.pi * x_grid) + 0.5 * np.cos(4 * np.pi * x_grid)

    # 1. Eilers-Marx engine (B-spline + GCV gridsearch)
    t0 = time.perf_counter()
    m_em = LinGAM(n_splines=12, backend="eilers")
    m_em.gridsearch(x, y, lam=np.logspace(-3, 3, 25))
    t_em = (time.perf_counter() - t0) * 1000

    pred_em = m_em.predict(x_grid)
    ci_em = m_em.confidence_intervals(x_grid, width=0.95)

    # 2. Wood engine (Cubic splines + REML auto-optimization)
    t0 = time.perf_counter()
    m_wd = LinGAM(k=12, bs="cr", backend="wood")
    m_wd.fit(x, y)
    t_wd = (time.perf_counter() - t0) * 1000

    pred_wd = m_wd.predict(x_grid)
    ci_wd = m_wd.confidence_intervals(x_grid, width=0.95)

    # Summary table
    print(f"\n{'Metric':<25} {'Eilers-Marx (B-spline + GCV)':<30} {'Wood (Cubic + REML)':<26}")
    print("-" * 78)
    print(f"{'Basis type':<25} {'B-splines (order 3)':<30} {'Cubic reg. spline (cr)':<26}")
    print(f"{'Basis dimension (k)':<25} {m_em.n_splines:<30} {m_wd.n_splines:<26}")
    print(f"{'Smoothing parameter (sp)':<25} {m_em.lam:<30.4f} {m_wd.lam:<26.4f}")
    print(f"{'Effective DF (EDF)':<25} {m_em.statistics_['edof']:<30.2f} {m_wd.statistics_['edof']:<26.2f}")
    print(f"{'Residual SS (RSS)':<25} {m_em.statistics_['rss']:<30.3f} {m_wd.statistics_['rss']:<26.3f}")
    print(f"{'Deviance Explained':<25} {m_em.deviance_explained()*100:<29.1f}% {m_wd.deviance_explained()*100:<25.1f}%")
    print(f"{'Fit time':<25} {t_em:<28.2f}ms {t_wd:<24.2f}ms")

    # =========================================================================
    # PART 2: Multivariate (y vs x1, x2, x3)
    # =========================================================================
    print("\n" + "-" * 78)
    print("  PART 2: Multivariate Fit (y vs x1, x2, x3)")
    print("-" * 78)

    n_multi = 250
    x1 = np.random.uniform(-1, 1, n_multi)
    x2 = np.random.uniform(0, 2, n_multi)
    x3 = np.random.uniform(-2, 2, n_multi)

    # True additive components
    f1 = np.sin(np.pi * x1)
    f2 = 0.5 * (x2 ** 2) - 0.7
    f3 = 1.2 * x3
    y_multi = f1 + f2 + f3 + np.random.normal(0, 0.3, n_multi)
    X = np.column_stack([x1, x2, x3])

    formula = "s(0, k=10) + s(1, k=10) + l(2)"
    print(f"\nModel Formula: {formula}")

    # 1. Eilers-Marx multivariate (B-splines + 2D Cartesian GCV gridsearch: 9x9 = 81 models)
    t0 = time.perf_counter()
    g_em = GAMCore(formula, backend="eilers")
    g_em.gridsearch(X, y_multi, lam=[np.logspace(-2, 2, 9), np.logspace(-2, 2, 9)])
    t_em_multi = (time.perf_counter() - t0) * 1000

    # 2. Wood multivariate (Cubic splines + REML auto-optimization)
    t0 = time.perf_counter()
    g_wd = GAMCore(formula, backend="wood")
    g_wd.fit(X, y_multi)
    t_wd_multi = (time.perf_counter() - t0) * 1000

    lam_em_str = f"[{', '.join(f'{t.lam:.2f}' for t in g_em._terms[:2])}]"
    sp_wd_str = f"[{', '.join(f'{s:.4f}' for s in g_wd.statistics_['sp'][:2])}]"

    print(f"\n{'Multivariate Metric':<25} {'Eilers-Marx (81-grid GCV)':<30} {'Wood (REML)':<26}")
    print("-" * 78)
    print(f"{'Smoothing params':<25} {lam_em_str:<30} {sp_wd_str:<26}")
    print(f"{'Total EDF':<25} {g_em.statistics_['edof']:<30.2f} {g_wd.statistics_['edof']:<26.2f}")
    print(f"{'Residual SS (RSS)':<25} {g_em.statistics_['rss']:<30.3f} {g_wd.statistics_['rss']:<26.3f}")
    print(f"{'Deviance Explained':<25} {g_em.deviance_explained()*100:<29.1f}% {g_wd.deviance_explained()*100:<25.1f}%")
    print(f"{'Optimization time':<25} {t_em_multi:<28.2f}ms {t_wd_multi:<24.2f}ms")

    # =========================================================================
    # Visualizations
    # =========================================================================
    fig = plt.figure(figsize=(15, 10))

    # --- Row 1: Univariate overlay  -  both backends on the same axes ---
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(x, y, color="#7f7f7f", alpha=0.5, s=18, label="Data")
    ax1.plot(x_grid, y_grid_true, "k--", lw=1.5, label="True function")
    ax1.plot(x_grid, pred_em, color="#1f77b4", lw=2, label=f"Eilers-Marx (EDF={m_em.statistics_['edof']:.1f})")
    ax1.fill_between(x_grid, ci_em[:, 0], ci_em[:, 1], color="#1f77b4", alpha=0.15)
    ax1.plot(x_grid, pred_wd, color="#d62728", lw=2, ls="-.", label=f"Wood (EDF={m_wd.statistics_['edof']:.1f})")
    ax1.fill_between(x_grid, ci_wd[:, 0], ci_wd[:, 1], color="#d62728", alpha=0.15)
    ax1.set_title("1. Univariate overlay: Eilers-Marx (B-splines + GCV) vs Wood (Cubic + REML)", fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Row 2: Multivariate Partial Dependence Comparison ---
    # Partial dependence for s(0)
    grid_x1 = np.linspace(-1, 1, 100)
    dummy_X1 = np.zeros((100, 3))
    dummy_X1[:, 0] = grid_x1

    pdep_em_1 = g_em.partial_dependence(0, dummy_X1)
    pdep_wd_1 = g_wd.partial_dependence(0, dummy_X1)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(grid_x1, np.sin(np.pi * grid_x1), "k--", lw=1.5, label="True f1(x1)")
    ax3.plot(grid_x1, pdep_em_1, color="#1f77b4", lw=2, label="Eilers-Marx")
    ax3.plot(grid_x1, pdep_wd_1, color="#d62728", lw=2, ls="-.", label="Wood")
    ax3.set_title("2A. Component s(0): sin(pi * x1)", fontweight="bold")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("Effect")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Partial dependence for s(1)
    grid_x2 = np.linspace(0, 2, 100)
    dummy_X2 = np.zeros((100, 3))
    dummy_X2[:, 1] = grid_x2

    pdep_em_2 = g_em.partial_dependence(1, dummy_X2)
    pdep_wd_2 = g_wd.partial_dependence(1, dummy_X2)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(grid_x2, 0.5 * (grid_x2 ** 2) - np.mean(0.5 * (grid_x2 ** 2)), "k--", lw=1.5, label="True centered f2(x2)")
    ax4.plot(grid_x2, pdep_em_2, color="#1f77b4", lw=2, label="Eilers-Marx")
    ax4.plot(grid_x2, pdep_wd_2, color="#d62728", lw=2, ls="-.", label="Wood")
    ax4.set_title("2B. Component s(1): quadratic x2", fontweight="bold")
    ax4.set_xlabel("x2")
    ax4.set_ylabel("Effect")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle("magli Engine Comparison: Eilers-Marx vs Wood", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    run_comparison()
