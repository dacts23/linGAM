"""
Example 04 — Shape Constraints
==============================

Demonstrate soft shape constraints on a single smooth term:
monotonic increasing, monotonic decreasing, convex, concave, and periodic.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import LinGAM

np.random.seed(42)

# 1. Monotonic increasing (dose-response)
dose = np.random.uniform(0, 10, 200)
response = 5 * (1 - np.exp(-0.4 * dose)) + np.random.normal(0, 0.3, 200)

model_inc = LinGAM(n_splines=12, constraint='mono_inc')
model_inc.fit(dose, response)

# 2. Monotonic decreasing (diminishing returns)
model_dec = LinGAM(n_splines=12, constraint='mono_dec')
model_dec.fit(dose, 10 - response + np.random.normal(0, 0.3, 200))

# 3. Periodic (seasonal cycle)
hour = np.random.uniform(0, 24, 300)
temp = 20 + 8 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1, 300)

model_per = LinGAM(n_splines=14, constraint='periodic')
model_per.fit(hour, temp)

# 4. Convex (U-shaped)
x_cv = np.random.uniform(-3, 3, 250)
y_cv = 0.5 * x_cv**2 + np.random.normal(0, 0.3, 250)

model_cvx = LinGAM(n_splines=12, constraint='convex')
model_cvx.fit(x_cv, y_cv)

# 5. Concave (inverted U)
model_ccv = LinGAM(n_splines=12, constraint='concave')
model_ccv.fit(x_cv, -0.5 * x_cv**2 + np.random.normal(0, 0.3, 250))

print("=== Shape Constraints ===")
print(f"Mono_inc EDF : {model_inc.statistics_['edof']:.2f}")
print(f"Mono_dec EDF : {model_dec.statistics_['edof']:.2f}")
print(f"Periodic EDF : {model_per.statistics_['edof']:.2f}")
print(f"Convex   EDF : {model_cvx.statistics_['edof']:.2f}")
print(f"Concave  EDF : {model_ccv.statistics_['edof']:.2f}")

# Verify periodic wrap-around (soft constraint — approximate equality)
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

# --- Diagnostics: observed vs fitted & residuals vs fitted ---
# Use the monotonic-increasing model as representative
y_fit = model_inc.predict(dose)
residuals = response - y_fit

fig_diag, axes_diag = plt.subplots(1, 2, figsize=(10, 4))
axes_diag[0].scatter(response, y_fit, c='steelblue', edgecolors='k', s=20, alpha=0.6)
ymin, ymax = response.min(), response.max()
axes_diag[0].plot([ymin, ymax], [ymin, ymax], 'r--', lw=1.5, label='1:1 line')
axes_diag[0].set_xlabel('Observed y')
axes_diag[0].set_ylabel('Fitted y')
axes_diag[0].set_title('Observed vs Fitted (mono_inc)')
axes_diag[0].legend()

axes_diag[1].scatter(y_fit, residuals, c='steelblue', edgecolors='k', s=20, alpha=0.6)
axes_diag[1].axhline(0, color='r', linestyle='--', lw=1.5)
axes_diag[1].set_xlabel('Fitted y')
axes_diag[1].set_ylabel('Residuals')
axes_diag[1].set_title('Residuals vs Fitted (mono_inc)')
fig_diag.tight_layout()
plt.show()
