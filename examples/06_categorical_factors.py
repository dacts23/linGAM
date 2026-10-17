"""
Example 06  -  Categorical Factors
================================

Include a categorical variable with a factor term ``f()``.
"""

import numpy as np
import matplotlib.pyplot as plt
from magli import GAMCore

np.random.seed(42)
n = 500
x = np.column_stack([
    np.random.uniform(0, 10, n),                    # continuous predictor
    np.random.choice([0, 1, 2], n).astype(float),   # categorical group
])
y = (
    np.sin(x[:, 0])
    + np.where(x[:, 1] == 0, -1.5,
        np.where(x[:, 1] == 1, 0.0, 2.0))
    + np.random.normal(0, 0.25, n)
)

model = GAMCore("s(0) + f(1)", backend="eilers")
model.fit(x, y)
model.gridsearch(x, y)

print("=== Categorical Factors ===")
print(f"EDF : {model.statistics_['edof']:.2f}")
print(f"GCV : {model.statistics_['GCV']:.4f}")
print(f"R²  : {model.deviance_explained():.4f}")

# Partial dependence for each term
pdep_s = model.partial_dependence(0, x)
pdep_f = model.partial_dependence(1, x)
print(f"Smooth partial dependence range : [{pdep_s.min():.3f}, {pdep_s.max():.3f}]")
print(f"Factor partial dependence range : [{pdep_f.min():.3f}, {pdep_f.max():.3f}]")

# Plot
fig, axes = model.plot_decomposition(x, y)
plt.show()

# --- Built-in diagnostics plot ---
model.plot_diagnostics(x, y)
plt.show()

# --- Backend comparison: Eilers-Marx vs Wood (side-by-side) ---
# Same s() + f() formula on both backends. Note the defaults differ: Eilers-Marx
# one-hot encodes f() with a ridge penalty, Wood dummy-encodes it as an
# unpenalised parametric factor  -  same fit, different parametrisation.
wood_model = GAMCore("s(0) + f(1)", backend="wood")
wood_model.fit(x, y)
print("\n--- Backend comparison: Eilers-Marx (GCV grid search) vs Wood (REML) ---")
print(f"{'Metric':<18}{'Eilers-Marx':>12}{'Wood':>12}")
print(f"{'EDF':<18}{model.statistics_['edof']:>12.2f}{wood_model.statistics_['edof']:>12.2f}")
print(f"{'RSS':<18}{model.statistics_['rss']:>12.4f}{wood_model.statistics_['rss']:>12.4f}")
print(f"{'R^2':<18}{model.deviance_explained():>12.4f}{wood_model.deviance_explained():>12.4f}")

# --- Same-plot overlay: centered s(0) partial dependence from both backends ---
# (Centered first: the backends split intercept/level differently, so only
# the shape is comparable  -  same convention as the numeric check in 09.)
x_sweep = np.tile(np.median(x, axis=0), (200, 1))
x_sweep[:, 0] = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)
pdep_em = model.partial_dependence(0, x_sweep)
pdep_wd = wood_model.partial_dependence(0, x_sweep)
fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
ax_cmp.plot(x_sweep[:, 0], pdep_em - pdep_em.mean(), 'b-', lw=2, label='Eilers-Marx backend')
ax_cmp.plot(x_sweep[:, 0], pdep_wd - pdep_wd.mean(), 'r-.', lw=2, label='Wood backend')
ax_cmp.set_xlabel('Feature 0')
ax_cmp.set_ylabel('Centered partial dependence')
ax_cmp.set_title('Backend Overlay: s(0) shape')
ax_cmp.legend()
fig_cmp.tight_layout()
plt.show()
