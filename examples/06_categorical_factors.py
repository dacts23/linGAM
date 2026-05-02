"""
Example 06 — Categorical Factors
================================

Include a categorical variable with a factor term ``f()``.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import GAMCore

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

model = GAMCore("s(0) + f(1)")
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

# --- Diagnostics: observed vs fitted & residuals vs fitted ---
y_fit = model.predict(x)
residuals = y - y_fit

fig_diag, axes_diag = plt.subplots(1, 2, figsize=(10, 4))
axes_diag[0].scatter(y, y_fit, c='steelblue', edgecolors='k', s=20, alpha=0.6)
ymin, ymax = y.min(), y.max()
axes_diag[0].plot([ymin, ymax], [ymin, ymax], 'r--', lw=1.5, label='1:1 line')
axes_diag[0].set_xlabel('Observed y')
axes_diag[0].set_ylabel('Fitted y')
axes_diag[0].set_title('Observed vs Fitted')
axes_diag[0].legend()

axes_diag[1].scatter(y_fit, residuals, c='steelblue', edgecolors='k', s=20, alpha=0.6)
axes_diag[1].axhline(0, color='r', linestyle='--', lw=1.5)
axes_diag[1].set_xlabel('Fitted y')
axes_diag[1].set_ylabel('Residuals')
axes_diag[1].set_title('Residuals vs Fitted')
fig_diag.tight_layout()
plt.show()
