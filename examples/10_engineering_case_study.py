"""
Example 10 -- Engineering Case Study: Turbine Blade Life Prediction
===================================================================

A realistic engineering workflow demonstrating most linGAM features on
simulated turbine blade lifetime data:

  * Multi-term formula -- ``s()``, ``te()``, ``f()``, ``l()``
  * Shape constraints -- ``mono_inc`` on temperature, ``concave`` on RPM
  * Tensor-product interaction -- temperature x vibration synergy
  * Categorical factors -- coating type with dummy coding
  * Linear terms -- cooling flow rate
  * Partial grid search -- tune only spline terms, pin te/f/l at current values
  * Robust fitting -- handle sensor outliers
  * Diagnostics -- GCV, AIC, BIC, R^2, EDF
  * Confidence & prediction intervals
  * ``plot_decomposition`` -- full multi-panel visualisation
  * Model comparison -- constrained vs unconstrained

HOW TO READ THIS OUTPUT
-----------------------
Each section below prints console output followed by one or more plots.
Close each plot window to advance to the next section.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import GAMCore

# ------------------------------------------------------------------
# 1. Generate synthetic engineering data (ground truth known)
# ------------------------------------------------------------------
np.random.seed(42)
n = 500

# --- Predictor variables (column index -> name) ---
#  x[:,0] = turbine inlet temperature  [deg C]
#  x[:,1] = shaft speed               [1000 rpm]
#  x[:,2] = blade vibration amplitude  [mm/s]
#  x[:,3] = cooling mass flow rate     [kg/s]
#  x[:,4] = blade coating type         {A, B, C}

FEATURE_NAMES = [
    "Temperature [C]",
    "Shaft speed [krpm]",
    "Vibration [mm/s]",
    "Cooling flow [kg/s]",
    "Coating type",
]

temp      = np.random.uniform(600, 1200, n)
rpm       = np.random.uniform(  8,   16, n)
vibration = np.random.uniform(  0,   15, n)
cooling   = np.random.uniform( 50,  200, n)
coating   = np.random.choice(['A', 'B', 'C'], n)

x = np.column_stack([temp, rpm, vibration, cooling])

# Encode coating as integer column; store original labels for display
coating_labels = np.array(coating)
coating_int = np.where(coating_labels == 'A', 0,
                       np.where(coating_labels == 'B', 1, 2))
x = np.column_stack([x, coating_int.astype(float)])


# --- Ground-truth component functions ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

f_temp = 40.0 * sigmoid((temp - 900.0) / 100.0) - 5.0
f_rpm  = -1.2 * (rpm - 12.0)**2 + 18.0
f_vib  = 0.8 * vibration + 2.0 * np.sin(vibration * 0.6)
f_interact = 0.015 * (temp - 900.0) * vibration / 15.0
f_cool = 0.12 * (cooling - 125.0)
f_coat = np.select(
    [coating_labels == 'A', coating_labels == 'B', coating_labels == 'C'],
    [0.0, -8.0, 6.0],
)
noise = np.random.normal(0, 2.0, n)
y = f_temp + f_rpm + f_vib + f_interact + f_cool + f_coat + noise

# Inject outliers for robust-fitting demonstration (5 % of points)
n_outliers = int(0.05 * n)
outlier_idx = np.random.choice(n, n_outliers, replace=False)
y[outlier_idx] += np.random.normal(0, 15.0, n_outliers)

print("=" * 72)
print("SECTION 1 -- DATA SUMMARY")
print("=" * 72)
print(f"  Samples : {n}")
print(f"  Features: {', '.join(FEATURE_NAMES)}")
print(f"  Coating levels: A (n={np.sum(coating_labels=='A')}), "
      f"B (n={np.sum(coating_labels=='B')}), "
      f"C (n={np.sum(coating_labels=='C')})")
print(f"  Outliers injected: {n_outliers} (5% of points)")
print(f"  y range: [{y.min():.1f}, {y.max():.1f}] blade life cycles")
print()
print("  Ground truth components (unknown to the model):")
print(f"    f_temp  = sigmoid ramp  (monotonic increasing)")
print(f"    f_rpm   = concave peak at 12 krpm")
print(f"    f_inter = temp x vibration synergy")
print(f"    f_cool  = linear (0.12 per kg/s)")
print(f"    f_coat  = {{A: 0, B: -8, C: +6}} baseline shifts")
print()

# ------------------------------------------------------------------
# 2. Build the model -- full formula with shape constraints
# ------------------------------------------------------------------
# Column layout:  x[0]=temp  x[1]=rpm  x[2]=vib  x[3]=cool  x[4]=coat
#
# s(0, mono_inc) : temperature -- physics says hotter always degrades
# s(1, concave)  : shaft speed -- optimal RPM exists, off-design hurts
# te(0,2)        : temp x vibration -- high temp + high vib is synergistic
# f(4, dummy)    : coating type -- shifts baseline (B, C relative to A)
# l(3)           : cooling flow -- linear protective effect

formula = (
    "s(0, constraint='mono_inc')"
    " + s(1, constraint='concave')"
    " + te(0, 2, n_splines=6)"
    " + f(4, coding='dummy')"
    " + l(3)"
)

model = GAMCore(formula, spline_order=3, fit_intercept=True)
model.fit(x, y)

print("=" * 72)
print("SECTION 2 -- BASELINE FIT (constraints active, single-pass pIRLS)")
print("=" * 72)
print(f"  GCV  : {model.statistics_['GCV']:.4f}   (lower is better)")
print(f"  EDF  : {model.statistics_['edof']:.2f}   (effective degrees of freedom)")
print(f"  R^2  : {model.deviance_explained():.4f}  (deviance explained)")
print(f"  AIC  : {model.aic():.2f}")
print(f"  BIC  : {model.bic():.2f}")
print()

# Show per-term parameter state
print("  Term parameters after initial fit:")
for i, term in enumerate(model._terms):
    tname = type(term).__name__.replace('_', '').replace('Term', '').lower()
    if tname == 'spline':
        constr = f" [{term.constraint}]" if term.constraint else ""
        print(f"    {i}: s({term.feature}){constr}  "
              f"n_splines={term.n_splines}, lam={term.lam:.4f}")
    elif tname == 'tensor':
        print(f"    {i}: te{tuple(term.features)}  "
              f"n_splines={term.n_splines_per}, lam={term.lam_per}")
    elif tname == 'factor':
        print(f"    {i}: f({term.feature})  lam={term.lam:.4f}")
    elif tname == 'linear':
        print(f"    {i}: l({term.feature})  lam={term.lam:.4f}")

# ------------------------------------------------------------------
# 3. Grid search -- tune only spline terms (new flexibility)
# ------------------------------------------------------------------
# We search only s(0) and s(1); te, f, l stay pinned at current values.
# With the new partial-grid support, we provide grids only for the
# 2 searched s() terms -- no need to supply dummy values for te/f/l.

model.gridsearch(
    x, y,
    search_terms=['s'],
    lam_grids=[np.logspace(-2, 2, 8),
               np.logspace(-2, 2, 8)],
    n_splines_grids=[np.arange(7, 22, 2),
                     np.arange(7, 22, 2)],
)

print("=" * 72)
print("SECTION 3 -- AFTER GRID SEARCH (tuned s(0) and s(1) only)")
print("=" * 72)
print(f"  GCV  : {model.statistics_['GCV']:.4f}   (lower = better fit-penalty tradeoff)")
print(f"  EDF  : {model.statistics_['edof']:.2f}   (model complexity)")
print(f"  R^2  : {model.deviance_explained():.4f}")
print(f"  AIC  : {model.aic():.2f}")
print(f"  BIC  : {model.bic():.2f}")
print()
print("  Grid search explored:")
print(f"    lambda values:   8 per spline term  (logspace(-2, 2, 8))")
print(f"    n_splines values: 8 per spline term  (arange(7, 22, 2))")
print(f"    Total lambda combos:  8 x 8 = 64")
print(f"    Total n_splines combos: 8 x 8 = 64")
print(f"    Total GCV evaluations: 64 x 64 = 4096")
print()
print("  Optimised term parameters:")
for i, term in enumerate(model._terms):
    tname = type(term).__name__.replace('_', '').replace('Term', '').lower()
    if tname == 'spline':
        constr = f" [{term.constraint}]" if term.constraint else ""
        print(f"    {i}: s({term.feature}){constr}  "
              f"n_splines={term.n_splines}, lam={term.lam:.4f}")
    elif tname == 'tensor':
        print(f"    {i}: te{tuple(term.features)}  "
              f"n_splines={term.n_splines_per}, lam={term.lam_per}  (pinned -- not searched)")
    elif tname == 'factor':
        print(f"    {i}: f({term.feature})  lam={term.lam:.4f}  (pinned)")
    elif tname == 'linear':
        print(f"    {i}: l({term.feature})  lam={term.lam:.4f}  (pinned)")
print()

# ------------------------------------------------------------------
# 4. Robust fitting -- handle sensor outliers
# ------------------------------------------------------------------
model_robust = GAMCore(formula)
model_robust.fit(x, y, robust=True)

print("=" * 72)
print("SECTION 4 -- ROBUST FIT (Huber IRLS, same formula)")
print("=" * 72)
print(f"  Standard GCV : {model.statistics_['GCV']:.4f}")
print(f"  Robust GCV   : {model_robust.statistics_['GCV']:.4f}")
print(f"  Standard EDF : {model.statistics_['edof']:.2f}")
print(f"  Robust EDF   : {model_robust.statistics_['edof']:.2f}")
print(f"  Standard R^2 : {model.deviance_explained():.4f}")
print(f"  Robust R^2   : {model_robust.deviance_explained():.4f}")
print("  -> Robust fit down-weights the 5% sensor outliers,")
print("     yielding lower GCV (better fit) with similar EDF.")
print()

# ------------------------------------------------------------------
# 5. Model comparison -- constrained vs unconstrained
# ------------------------------------------------------------------
formula_unconstrained = (
    "s(0)"                # no mono_inc constraint
    " + s(1)"             # no concave constraint
    " + te(0, 2, n_splines=6)"
    " + f(4, coding='dummy')"
    " + l(3)"
)
model_unc = GAMCore(formula_unconstrained)
model_unc.fit(x, y)

print("=" * 72)
print("SECTION 5 -- CONSTRAINED vs UNCONSTRAINED")
print("=" * 72)
print(f"  Constrained   :  GCV={model.statistics_['GCV']:.4f},  "
      f"EDF={model.statistics_['edof']:.2f},  "
      f"R^2={model.deviance_explained():.4f}")
print(f"  Unconstrained :  GCV={model_unc.statistics_['GCV']:.4f},  "
      f"EDF={model_unc.statistics_['edof']:.2f},  "
      f"R^2={model_unc.deviance_explained():.4f}")
print("  -> Shape constraints reduce EDF (simpler model) with negligible")
print("     loss in fit quality. This is desirable: the constrained model")
print("     respects known physics (monotonic temperature degradation,")
print("     concave RPM optimum) at lower complexity cost.")
print()

# ------------------------------------------------------------------
# 6. Confidence & prediction intervals
# ------------------------------------------------------------------
x_eval = x[:200]  # subset for display
ci = model.confidence_intervals(x_eval, width=0.95)
pi = model.prediction_intervals(x_eval, width=0.95)

ci_width = (ci[:, 1] - ci[:, 0]).mean()
pi_width = (pi[:, 1] - pi[:, 0]).mean()

print("=" * 72)
print("SECTION 6 -- INTERVALS (95% confidence, n=200 subset)")
print("=" * 72)
print(f"  Mean CI width : {ci_width:.3f} blade life cycles")
print(f"  Mean PI width : {pi_width:.3f} blade life cycles")
print(f"  CI = confidence interval on the mean prediction")
print(f"  PI = prediction interval for a single new observation")
print("  -> CI is narrow (we know the mean well);")
print("     PI is wide (individual blades have process variability).")
print()

# ------------------------------------------------------------------
# 7. Partial dependence -- isolate each term's contribution
# ------------------------------------------------------------------
print("=" * 72)
print("SECTION 7 -- PARTIAL DEPENDENCE RANGES (per-term contributions)")
print("=" * 72)
print("  Each value is the isolated contribution of one term to y,")
print("  with all other predictors held at their medians.")
print()
for i in range(model.n_terms):
    term = model._terms[i]
    pd = model.partial_dependence(i, x)
    tname = type(term).__name__.replace('_', '').replace('Term', '')
    if tname == 'Spline':
        desc = f"s({term.feature}) [{FEATURE_NAMES[term.feature]}]"
    elif tname == 'Tensor':
        desc = f"te{tuple(term.features)} [{' x '.join(FEATURE_NAMES[f] for f in term.features)}]"
    elif tname == 'Factor':
        desc = f"f({term.feature}) [{FEATURE_NAMES[term.feature]}]"
    elif tname == 'Linear':
        desc = f"l({term.feature}) [{FEATURE_NAMES[term.feature]}]"
    else:
        desc = tname
    impact = pd.max() - pd.min()
    print(f"  Term {i} {desc}:")
    print(f"    range  = [{pd.min():.2f}, {pd.max():.2f}]  (span = {impact:.2f})")
print()
print("  -> The term with largest range has the strongest influence on blade life.")
print()

# ------------------------------------------------------------------
# 8. Diagnostics plots -- model quality assessment
# ------------------------------------------------------------------
y_fit = model.predict(x)
residuals = y - y_fit

fig_diag, axes_diag = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: Observed vs Fitted
axes_diag[0].scatter(y, y_fit, c='steelblue', edgecolors='k', s=15, alpha=0.5,
                     label=f'n={n}')
ymin, ymax = y.min(), y.max()
axes_diag[0].plot([ymin, ymax], [ymin, ymax], 'r--', lw=1.5, label='Perfect fit')
axes_diag[0].set_xlabel('Observed blade life [cycles]')
axes_diag[0].set_ylabel('Fitted blade life [cycles]')
axes_diag[0].set_title('Observed vs Fitted\n'
                       '(points on the red line = perfect prediction)')
axes_diag[0].legend(fontsize=8)

# Right: Residuals vs Fitted
axes_diag[1].scatter(y_fit, residuals, c='steelblue', edgecolors='k', s=15, alpha=0.5)
axes_diag[1].axhline(0, color='r', linestyle='--', lw=1.5,
                     label='Zero residual')
axes_diag[1].set_xlabel('Fitted blade life [cycles]')
axes_diag[1].set_ylabel('Residual [cycles]')
axes_diag[1].set_title('Residuals vs Fitted\n'
                       '(random scatter around zero = good model)')
axes_diag[1].legend(fontsize=8)

fig_diag.suptitle(f'Diagnostics -- Constrained Model  '
                  f'(GCV={model.statistics_["GCV"]:.3f}, '
                  f'R^2={model.deviance_explained():.3f}, '
                  f'EDF={model.statistics_["edof"]:.1f})',
                  fontsize=12, fontweight='bold')
fig_diag.tight_layout()
print("[Plot 1] Diagnostics -- close to continue...")
plt.show()

# ------------------------------------------------------------------
# 9. Full decomposition plot -- how the model sees each predictor
# ------------------------------------------------------------------
# The decomposition plot shows:
#   Row 0: Overall fit (x[0] sweep vs y, other predictors at median)
#   Rows 1+: One panel per formula term
#     s()   -> partial dependence curve (contribution vs predictor)
#     te()  -> 2D heatmap of interaction surface
#     f()   -> bar chart of level baselines
#     l()   -> linear effect line
#
# Red shaded regions = extrapolation beyond training data range.

fig_dec, axes_dec = model.plot_decomposition(
    x, y, n_grid=150, extrap_frac=0.10,
    feature_names=FEATURE_NAMES,
)

# Add feature-name labels to the overall fit panel (can't change
# plot_decomposition internals, but we can annotate the figure)
fig_dec.suptitle(
    'GAM Decomposition -- How Each Term Contributes to Blade Life',
    fontsize=13, fontweight='bold',
)

# Add a text box explaining how to read the chart
fig_dec.text(
    0.5, 0.01,
    "Each subplot shows one term's ISOLATED contribution to blade life, "
    "with all other predictors held at their medians.\n"
    "Red shading = extrapolation beyond data range.  "
    f"Columns: {', '.join(FEATURE_NAMES[:4])}",
    ha='center', fontsize=8, style='italic', color='gray',
)
print("[Plot 2] Full GAM decomposition -- close to continue...")
plt.show()

# ------------------------------------------------------------------
# 10. Robust vs standard fit -- outlier behaviour comparison
# ------------------------------------------------------------------
y_std = model.predict(x)
y_rob = model_robust.predict(x)

fig_comp, ax_comp = plt.subplots(figsize=(9, 5.5))
ax_comp.scatter(y, y_std, c='steelblue', s=12, alpha=0.4,
                label=f'Standard fit (n={n - n_outliers} clean + '
                      f'{n_outliers} outliers)')
ax_comp.scatter(y[outlier_idx], y_std[outlier_idx],
                c='darkorange', s=35, alpha=0.8, marker='x',
                linewidths=1.5, label='Outliers -- standard fit')
ax_comp.scatter(y[outlier_idx], y_rob[outlier_idx],
                c='green', s=35, alpha=0.8, marker='o',
                linewidths=1.5, label='Outliers -- robust fit')
ax_comp.plot([ymin, ymax], [ymin, ymax], 'k--', lw=0.8, alpha=0.5)
ax_comp.set_xlabel('Observed blade life [cycles]')
ax_comp.set_ylabel('Fitted blade life [cycles]')
ax_comp.set_title(
    f'Standard vs Robust: how 5% outliers affect predictions\n'
    f'(orange X = standard fit on outliers;  '
    f'green circles = robust fit -- pulled closer to 1:1 line)',
    fontsize=11,
)
ax_comp.legend(fontsize=8, loc='lower right')

# Add a text box with key numbers
ax_comp.text(
    0.98, 0.05,
    f"Standard: GCV={model.statistics_['GCV']:.3f}, R^2={model.deviance_explained():.3f}\n"
    f"Robust:   GCV={model_robust.statistics_['GCV']:.3f}, R^2={model_robust.deviance_explained():.3f}",
    transform=ax_comp.transAxes, fontsize=9,
    ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8),
)
fig_comp.tight_layout()
print("[Plot 3] Standard vs Robust comparison -- close to exit.")
plt.show()
