"""
Example 04  -  Shape Constraints
==============================

Demonstrate soft shape constraints on a single smooth term:
monotonic increasing, monotonic decreasing, convex, concave, and periodic.
"""

import numpy as np
import matplotlib.pyplot as plt
from magli import LinGAM

np.random.seed(42)

# 1. Monotonic increasing (dose-response)
dose = np.random.uniform(0, 10, 200)
response = 5 * (1 - np.exp(-0.4 * dose)) + np.random.normal(0, 0.3, 200)

model_inc = LinGAM(n_splines=12, constraint='mono_inc', backend="eilers")
model_inc.fit(dose, response)

# 2. Monotonic decreasing (diminishing returns)
model_dec = LinGAM(n_splines=12, constraint='mono_dec', backend="eilers")
model_dec.fit(dose, 10 - response + np.random.normal(0, 0.3, 200))

# 3. Periodic (seasonal cycle)
hour = np.random.uniform(0, 24, 300)
temp = 20 + 8 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1, 300)

model_per = LinGAM(n_splines=14, constraint='periodic', backend="eilers")
model_per.fit(hour, temp)

# 4. Convex (U-shaped)
x_cv = np.random.uniform(-3, 3, 250)
y_cv = 0.5 * x_cv**2 + np.random.normal(0, 0.3, 250)

model_cvx = LinGAM(n_splines=12, constraint='convex', backend="eilers")
model_cvx.fit(x_cv, y_cv)

# 5. Concave (inverted U)
model_ccv = LinGAM(n_splines=12, constraint='concave', backend="eilers")
model_ccv.fit(x_cv, -0.5 * x_cv**2 + np.random.normal(0, 0.3, 250))

print("=== Shape Constraints ===")
print(f"Mono_inc EDF : {model_inc.statistics_['edof']:.2f}")
print(f"Mono_dec EDF : {model_dec.statistics_['edof']:.2f}")
print(f"Periodic EDF : {model_per.statistics_['edof']:.2f}")
print(f"Convex   EDF : {model_cvx.statistics_['edof']:.2f}")
print(f"Concave  EDF : {model_ccv.statistics_['edof']:.2f}")

# Verify periodic wrap-around (soft constraint  -  approximate equality)
print(f"predict(6)  = {model_per.predict([6])[0]:.4f}")
print(f"predict(30) = {model_per.predict([30])[0]:.4f}")
print(f"diff        = {abs(model_per.predict([6])[0] - model_per.predict([30])[0]):.4f}")
assert np.isclose(model_per.predict([6]), model_per.predict([30]), atol=0.5)
print("Periodic wrap-around verified (soft constraint)")

# Plot all five
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.ravel()

def plot_fit(ax, x_data, y_data, model, title, x_grid=None):
    if x_grid is None:
        x_grid = np.linspace(x_data.min(), x_data.max(), 400)
    ax.scatter(x_data, y_data, c='lightgray', s=10)
    ax.plot(x_grid, model.predict(x_grid), 'b-', lw=2)
    ax.set_title(title)

plot_fit(axes[0], dose, response, model_inc, 'Monotonic Increasing')
plot_fit(axes[1], dose, 10 - response, model_dec, 'Monotonic Decreasing')
plot_fit(axes[2], hour, temp, model_per, 'Periodic (24h)', np.linspace(0, 48, 500))
plot_fit(axes[3], x_cv, y_cv, model_cvx, 'Convex')
plot_fit(axes[4], x_cv, -0.5 * x_cv**2, model_ccv, 'Concave')
axes[5].axis('off')

fig.tight_layout()
plt.show()

# --- Built-in diagnostics plot ---
model_inc.plot_diagnostics(dose, response)
plt.show()

# --- Backend comparison: Eilers-Marx vs Wood (side-by-side) ---
# Soft shape constraints (mono_inc/mono_dec/convex/concave) are an Eilers-Marx-backend
# feature; the Wood backend natively supports only the periodic constraint
# (cyclic cubic splines, bs='cc'). So only the seasonal model has a Wood
# counterpart  -  it should match the Eilers-Marx periodic fit up to basis differences.
wood_per = LinGAM(k=14, bs="cc", backend="wood")
wood_per.fit(hour, temp)
print("\n--- Backend comparison: periodic fit on both backends ---")
print(f"{'Metric':<18}{'Eilers-Marx periodic':>22}{'Wood (cc)':>12}")
print(f"{'EDF':<18}{model_per.statistics_['edof']:>22.2f}{wood_per.statistics_['edof']:>12.2f}")
print(f"{'RSS':<18}{model_per.statistics_['rss']:>22.4f}{wood_per.statistics_['rss']:>12.4f}")
print(f"{'R^2':<18}{model_per.deviance_explained():>22.4f}{wood_per.deviance_explained():>12.4f}")
print(f"Eilers-Marx wrap diff |f(6)-f(30)|  = {abs(model_per.predict([6])[0] - model_per.predict([30])[0]):.4f}")
print(f"Wood        wrap diff |f(6)-f(30)|  = {abs(wood_per.predict([6])[0] - wood_per.predict([30])[0]):.4f}")
print("(mono_inc/mono_dec/convex/concave have no Wood equivalent  -  Eilers-Marx-only.)")

# --- Same-plot overlay: periodic fits from both backends ---
hour_grid = np.linspace(0, 48, 500)
fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4))
ax_cmp.scatter(hour, temp, c='lightgray', s=10, label='Data')
ax_cmp.plot(hour_grid, model_per.predict(hour_grid), 'b-', lw=2, label='Eilers-Marx periodic')
ax_cmp.plot(hour_grid, wood_per.predict(hour_grid), 'r-.', lw=2, label="Wood (bs='cc')")
ax_cmp.set_xlabel('Hour')
ax_cmp.set_ylabel('Temperature')
ax_cmp.set_title('Backend Overlay: periodic fits')
ax_cmp.legend()
fig_cmp.tight_layout()
plt.show()
