"""Demo for linGAM — concrete compressive strength GAM."""

import numpy as np
import matplotlib.pyplot as plt

from lingam import GAMCore
from lingam._terms import _FactorTerm, _LinearTerm, _SplineTerm, _TensorTerm


def _concrete_demo():
    print("=" * 72)
    print("  GAMCore — Concrete Compressive Strength GAM")
    print("=" * 72)

    np.random.seed(42)
    n = 200

    age = np.random.uniform(1, 90, n)
    wc = np.random.uniform(0.30, 0.60, n)
    cement_labels = np.random.choice(
        ['Ordinary', 'Rapid-Hardening', 'Sulfate-Resistant'], n,
        p=[0.5, 0.3, 0.2],
    )
    cement_map = {'Ordinary': 0, 'Rapid-Hardening': 1, 'Sulfate-Resistant': 2}
    cement = np.array([cement_map[c] for c in cement_labels], dtype=float)
    temp = np.random.uniform(15, 35, n)

    cement_bias = {'Ordinary': 0.0, 'Rapid-Hardening': 8.0, 'Sulfate-Resistant': 5.0}
    strength_true = (
        55.0 * (1 - np.exp(-0.09 * age))
        - 35.0 * (wc - 0.30)
        + np.array([cement_bias[c] for c in cement_labels])
        + 0.08 * (temp - 25)
        + 0.12 * age * (0.60 - wc)
    )
    strength = strength_true + np.random.normal(0, 2.5, n)

    X = np.column_stack([age, wc, cement, temp])

    outlier_idx = [10, 50, 120, 180]
    strength[outlier_idx] += np.random.choice([-20, 25], len(outlier_idx))

    print(f"\n  n = {n} observations, 4 features")
    print(f"  y range: [{strength.min():.1f}, {strength.max():.1f}] MPa")

    # -- EDA plots --
    print("\n  -- Exploratory data plots (justifying formula choices) --")

    fig_eda, axs = plt.subplots(2, 3, figsize=(16, 10))
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axs

    ax1.scatter(age, strength, c=wc, cmap='viridis', alpha=0.6, s=14,
                edgecolors='k', linewidth=0.3)
    age_smooth = np.linspace(1, 90, 300)
    ax1.plot(age_smooth, 55 * (1 - np.exp(-0.09 * age_smooth)) - 35 * (0.45 - 0.30)
             + 0.12 * age_smooth * 0.15, 'r--', lw=2, label='True at w/c=0.45')
    plt.colorbar(ax1.collections[0], ax=ax1, label='w/c ratio')
    ax1.set_xlabel('Age (days)')
    ax1.set_ylabel('Compressive Strength (MPa)')
    ax1.set_title('Strength vs Age (nonlinear curing)')
    ax1.legend(fontsize=7)
    ax1.annotate(
        's(age)\nNonlinear:\nrapid early gain,\nplateau after ~28d',
        xy=(45, 30), fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                   edgecolor='orange', alpha=0.9),
    )

    wc_bins = ['Low w/c\n(0.30-0.40)', 'Mid w/c\n(0.40-0.50)', 'High w/c\n(0.50-0.60)']
    wc_colors = ['#2166ac', '#f4a582', '#b2182b']
    for bi, (lo, hi) in enumerate([(0.30, 0.40), (0.40, 0.50), (0.50, 0.60)]):
        mask = (wc >= lo) & (wc < hi)
        ax2.scatter(age[mask], strength[mask], c=wc_colors[bi],
                     alpha=0.5, s=12, label=wc_bins[bi])
        if mask.sum() > 5:
            idx_sort = np.argsort(age[mask])
            age_s = age[mask][idx_sort]
            str_s = strength[mask][idx_sort]
            window = max(5, mask.sum() // 6)
            roll = np.convolve(str_s, np.ones(window) / window, mode='same')
            ax2.plot(age_s, roll, '-', color=wc_colors[bi], lw=2.5)
    ax2.set_xlabel('Age (days)')
    ax2.set_ylabel('Compressive Strength (MPa)')
    ax2.set_title('Strength vs Age by w/c ratio')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.annotate(
        'te(age, w/c)\nInteraction:\nhigh w/c concretes\ngain strength slower',
        xy=(50, 15), fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                   edgecolor='orange', alpha=0.9),
    )

    cement_cats = ['Ordinary', 'Rapid-Hard.', 'Sulfate-Res.']
    cement_vals = [strength[cement_labels == c] for c in
                   ['Ordinary', 'Rapid-Hardening', 'Sulfate-Resistant']]
    bp = ax3.boxplot(cement_vals, tick_labels=cement_cats, patch_artist=True,
                      widths=0.5, showfliers=False)
    for patch, color in zip(bp['boxes'], ['#66c2a5', '#fc8d62', '#8da0cb']):
        patch.set_facecolor(color)
    for i, vals in enumerate(cement_vals):
        jitter = np.random.normal(0, 0.05, len(vals))
        ax3.scatter(np.full_like(vals, i + 1) + jitter, vals,
                     alpha=0.4, s=10, color='black')
    ax3.set_ylabel('Compressive Strength (MPa)')
    ax3.set_title('Strength by Cement Type')
    ax3.annotate(
        'f(cement)\nCategorical:\nthree discrete\ngroups, no\nnatural order',
        xy=(2.5, 30), fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                   edgecolor='orange', alpha=0.9),
    )

    residual_temp = strength - 55 * (1 - np.exp(-0.09 * age)) + 35 * (wc - 0.30) - \
                    np.array([cement_bias[c] for c in cement_labels])
    ax4.scatter(temp, residual_temp, c='steelblue', alpha=0.5, s=14,
                edgecolors='k', linewidth=0.3)
    coeffs = np.polyfit(temp, residual_temp, 1)
    ax4.plot(np.sort(temp), np.polyval(coeffs, np.sort(temp)), 'r-', lw=2,
             label=f'slope={coeffs[0]:.3f}')
    ax4.set_xlabel('Curing Temperature (deg C)')
    ax4.set_ylabel('Approx. Residual (MPa)')
    ax4.set_title('Residual vs Temperature')
    ax4.legend(fontsize=7)
    ax4.annotate(
        'l(temp)\nLinear:\nweak monotonic\ntrend, 1 df\nis sufficient',
        xy=(28, 3), fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                   edgecolor='orange', alpha=0.9),
    )

    wc_effect_mask = (age > 25) & (age < 35)
    if wc_effect_mask.sum() > 10:
        ax5.scatter(wc[wc_effect_mask], strength[wc_effect_mask],
                     c='darkgreen', alpha=0.5, s=14, edgecolors='k', linewidth=0.3)
        coeffs2 = np.polyfit(wc[wc_effect_mask], strength[wc_effect_mask], 1)
        wc_sorted = np.sort(wc[wc_effect_mask])
        ax5.plot(wc_sorted, np.polyval(coeffs2, wc_sorted), 'r-', lw=2)
    else:
        ax5.scatter(wc, strength, c='darkgreen', alpha=0.5, s=14,
                    edgecolors='k', linewidth=0.3)
    ax5.set_xlabel('Water/Cement Ratio')
    ax5.set_ylabel('Strength at ~28d (MPa)')
    ax5.set_title('W/C ratio effect (age ~28d)')

    ax6.axis('off')
    summary_text = (
        "GAM Additive Decomposition\n"
        "=======================\n\n"
        "y = s(age)        <- nonlinear curing\n"
        "  + te(age, w/c)  <- interaction\n"
        "  + f(cement)     <- categorical offset\n"
        "  + l(temp)       <- linear correction\n"
        "  + intercept\n\n"
        "Why GAM instead of linear model?\n"
        "- s() captures the S-shaped curing\n"
        "  curve without manual feature engineering\n"
        "- te() models interaction without\n"
        "  pre-specifying its functional form\n"
        "- f() automatically handles categorical\n"
        "  variables via one-hot encoding\n"
        "- lam penalty prevents overfitting\n"
        "  (chosen automatically via GCV)"
    )
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                        edgecolor='orange', alpha=0.9))

    fig_eda.suptitle('Exploratory Data Analysis', fontsize=14, fontweight='bold')
    fig_eda.tight_layout()

    # -- Model --
    formula = "s(0, n_splines=10) + te(0, 1, n_splines=6) + f(2) + l(3)"

    print(f"\n  Formula: {formula}")
    print("    s(0)     = age (nonlinear curing curve)")
    print("    te(0,1)  = age x w/c interaction")
    print("    f(2)     = cement type (categorical)")
    print("    l(3)     = curing temperature (linear)")

    # Standard fit
    print("\n  -- Standard fit --")
    model_std = GAMCore(formula)
    model_std.fit(X, strength, robust=False)
    s_std = model_std.statistics_
    print(f"    GCV = {s_std['GCV']:.3f}")
    print(f"    EDF = {s_std['edof']:.2f}")
    print(f"    RSS = {s_std['rss']:.1f}")
    print(f"    sigma_hat = {s_std['scale']:.2f} MPa")

    # Robust fit
    print("\n  -- Robust fit (Huber IRLS) --")
    model_rob = GAMCore(formula)
    model_rob.fit(X, strength, robust=True)
    stats_r = model_rob.statistics_
    print(f"    GCV = {stats_r['GCV']:.3f}")
    print(f"    RSS = {stats_r['rss']:.1f}")

    # Per-term contributions
    print("\n  -- Per-term additive contributions --")
    for i in range(model_std.n_terms):
        term = model_std._terms[i]
        pdep = model_std.partial_dependence(i, X)
        if isinstance(term, _SplineTerm):
            print(f"    s({term.feature})  K={term.n_splines:2d}  lam={term.lam:.3f}  "
                  f"contrib range [{pdep.min():.1f}, {pdep.max():.1f}] MPa")
        elif isinstance(term, _TensorTerm):
            print(f"    te({term.features})  "
                  f"K=({','.join(str(k) for k in term.n_splines_per)})  "
                  f"coefs={term.n_coefs}  "
                  f"contrib range [{pdep.min():.1f}, {pdep.max():.1f}] MPa")
        elif isinstance(term, _FactorTerm):
            print(f"    f({term.feature})  "
                  f"levels={getattr(term,'_levels',[])}  lam={term.lam:.3f}  "
                  f"contrib range [{pdep.min():.1f}, {pdep.max():.1f}] MPa")
        elif isinstance(term, _LinearTerm):
            beta = model_std.coef_[model_std._coef_slices[i][0]]
            print(f"    l({term.feature})  beta={beta:.4f}  "
                  f"contrib range [{pdep.min():.1f}, {pdep.max():.1f}] MPa")

    # Predictions at key ages
    print("\n  -- Strength predictions at key ages --")
    for test_age in [1, 3, 7, 14, 28, 56, 90]:
        x_row = np.array([[test_age, 0.45, 0, 20.0]])
        pred = model_std.predict(x_row)[0]
        ci = model_std.confidence_intervals(x_row, width=0.95)[0]
        print(f"    {test_age:3d} days:  {pred:5.1f} MPa  "
              f"(95% CI: [{ci[0]:5.1f}, {ci[1]:5.1f}])")

    # -- Selective grid search demo --
    print("\n  -- Selective grid search: only search s(0) and f(2) --")
    model_sel = GAMCore(formula)
    model_sel.gridsearch(X, strength,
                         lam_grids=[np.logspace(-2, 2, 3)],
                         n_splines_grids=[np.arange(6, 13, 3)],
                         search_terms=[0, 2])  # only s(0) and f(2)
    print(f"    GCV = {model_sel.statistics_['GCV']:.3f}")
    print(f"    s(0) best: K={model_sel._terms[0].n_splines}, "
          f"lam={model_sel._terms[0].lam:.3f}")
    print(f"    te(0,1) (not searched): K=({','.join(str(k) for k in model_sel._terms[1].n_splines_per)}), "
          f"lam={model_sel._terms[1].lam:.3f}")
    print(f"    f(2) best: lam={model_sel._terms[2].lam:.3f}")

    # -- Observed vs Fitted --
    y_hat_std = model_std.predict(X)
    y_hat_rob = model_rob.predict(X)
    residuals_std = strength - y_hat_std
    r2_std = 1 - np.var(residuals_std) / np.var(strength)
    r2_rob = 1 - np.var(strength - y_hat_rob) / np.var(strength)

    fig_fit, (axf1, axf2, axf3) = plt.subplots(1, 3, figsize=(18, 5.5))

    axf1.scatter(strength, y_hat_std, c='steelblue', alpha=0.5, s=14,
                 edgecolors='k', linewidth=0.3)
    lims = [min(strength.min(), y_hat_std.min()) - 2,
            max(strength.max(), y_hat_std.max()) + 2]
    axf1.plot(lims, lims, 'r--', lw=1.5, label='1:1 line')
    axf1.set_xlim(lims); axf1.set_ylim(lims)
    axf1.set_xlabel('Observed Strength (MPa)')
    axf1.set_ylabel('Fitted Strength (MPa)')
    axf1.set_title(f'Standard GAM: Observed vs Fitted\n'
                   f'R2 = {r2_std:.3f}  |  GCV = {s_std["GCV"]:.3f}')
    axf1.legend(fontsize=8)
    axf1.set_aspect('equal')

    axf2.scatter(strength, y_hat_rob, c='darkorange', alpha=0.5, s=14,
                 edgecolors='k', linewidth=0.3)
    axf2.plot(lims, lims, 'r--', lw=1.5, label='1:1 line')
    axf2.set_xlim(lims); axf2.set_ylim(lims)
    axf2.set_xlabel('Observed Strength (MPa)')
    axf2.set_ylabel('Fitted Strength (MPa)')
    axf2.set_title(f'Robust GAM: Observed vs Fitted\n'
                   f'R2 = {r2_rob:.3f}  |  GCV = {stats_r["GCV"]:.3f}')
    axf2.legend(fontsize=8)
    axf2.set_aspect('equal')

    axf3.scatter(y_hat_std, residuals_std, c='steelblue', alpha=0.5, s=14,
                 edgecolors='k', linewidth=0.3)
    axf3.axhline(0, color='red', ls='--', lw=1.5)
    axf3.set_xlabel('Fitted Strength (MPa)')
    axf3.set_ylabel('Residuals (MPa)')
    axf3.set_title('Residuals vs Fitted (Standard)')
    outlier_mask = np.abs(residuals_std) > 3 * np.std(residuals_std)
    if outlier_mask.any():
        axf3.scatter(y_hat_std[outlier_mask], residuals_std[outlier_mask],
                      c='red', s=30, edgecolors='k', linewidth=1, label='>3s outlier')
        axf3.legend(fontsize=8)

    fig_fit.suptitle('Model Fit Diagnostics', fontsize=13, fontweight='bold')
    fig_fit.tight_layout()

    # Decomposition plot
    model_std.plot_decomposition(X, strength, extrap_frac=0.15)
    plt.show()


if __name__ == "__main__":
    _concrete_demo()
