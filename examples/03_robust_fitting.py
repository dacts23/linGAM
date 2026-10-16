"""
Example 03 — Robust Fitting with Outliers
=========================================

Compare a standard least-squares fit vs. Huber IRLS robust fit
when the data contains extreme outliers.
"""

import numpy as np
import matplotlib.pyplot as plt
from gamkit import LinGAM

np.random.seed(42)

x = np.random.uniform(0, 10, 250)
y = np.sin(x) + np.random.normal(0, 0.2, len(x))

# Inject extreme outliers
y[20] += 9
y[80] -= 8
y[150] += 7

# Standard fit
model_std = LinGAM(n_splines=14, backend="eilers")
model_std.fit(x, y)

# Robust fit
model_rob = LinGAM(n_splines=14, backend="eilers")
model_rob.fit(x, y, robust=True)

print("=== Robust Fitting ===")
print(f"Standard RSS : {model_std.statistics_['rss']:.4f}")
print(f"Robust RSS   : {model_rob.statistics_['rss']:.4f}")
print(f"Standard GCV : {model_std.statistics_['GCV']:.4f}")
print(f"Robust GCV   : {model_rob.statistics_['GCV']:.4f}")

wood_model = LinGAM(k=14, bs="cr", backend="wood")
wood_model.fit(x, y)

wood_rob = LinGAM(k=14, bs="cr", backend="wood")
wood_rob.fit(x, y, robust=True)

x_grid = np.linspace(0, 10, 500)

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(x, y, c='lightgray', s=15, label='Data (with outliers)')
ax.plot(x_grid, model_std.predict(x_grid), 'r--', lw=2, label='Eilers-Marx std')
ax.plot(x_grid, model_rob.predict(x_grid), 'g-', lw=2, label='Eilers-Marx robust')
ax.plot(x_grid, wood_model.predict(x_grid), 'b-.', lw=1.5, label='Wood std')
ax.plot(x_grid, wood_rob.predict(x_grid), 'm:', lw=2, label='Wood robust')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Standard vs Robust Fit (Eilers-Marx & Wood)')
ax.legend()
fig.tight_layout()
plt.show()

# --- Built-in diagnostics plot ---
for model in (model_std, model_rob):
    model.plot_diagnostics(x, y)
    plt.show()

# --- Backend comparison: Eilers-Marx vs Wood (side-by-side) ---
print("\n--- Backend comparison: Standard vs Robust (Huber IRLS) ---")
print(f"{'Metric':<18}{'Eilers-Marx std':>16}{'E-M robust':>14}{'Wood std':>12}{'Wood robust':>14}")
print(f"{'EDF':<18}{model_std.statistics_['edof']:>16.2f}{model_rob.statistics_['edof']:>14.2f}{wood_model.statistics_['edof']:>12.2f}{wood_rob.statistics_['edof']:>14.2f}")
print(f"{'RSS':<18}{model_std.statistics_['rss']:>16.4f}{model_rob.statistics_['rss']:>14.4f}{wood_model.statistics_['rss']:>12.4f}{wood_rob.statistics_['rss']:>14.4f}")
print(f"{'R^2':<18}{model_std.deviance_explained():>16.4f}{model_rob.deviance_explained():>14.4f}{wood_model.deviance_explained():>12.4f}{wood_rob.deviance_explained():>14.4f}")
