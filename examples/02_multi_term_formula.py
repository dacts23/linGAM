"""
Example 02 — Multi-Term Formula Interface
=========================================

GAMCore with a smooth term and a linear term.
Shows how the formula string maps features to model components.
"""

import numpy as np
import matplotlib.pyplot as plt
from lingam import GAMCore

np.random.seed(42)
n = 400
x = np.column_stack([
    np.random.uniform(0, 10, n),   # feature 0 — smooth
    np.random.uniform(0, 5, n),    # feature 1 — linear
])
y = (
    np.sin(x[:, 0])
    + 0.4 * x[:, 1]
    + np.random.normal(0, 0.25, n)
)

# Fit a 2-term model
model = GAMCore("s(0, n_splines=12) + l(1)")
model.fit(x, y)

print("=== Multi-Term GAMCore ===")
print(f"EDF : {model.statistics_['edof']:.2f}")
print(f"GCV : {model.statistics_['GCV']:.4f}")
print(f"AIC : {model.aic():.2f}")
print(f"BIC : {model.bic():.2f}")
print(f"R²  : {model.deviance_explained():.4f}")

# Grid search to tune both terms
model.gridsearch(x, y)
print(f"\nAfter gridsearch:")
print(f"EDF : {model.statistics_['edof']:.2f}")
print(f"GCV : {model.statistics_['GCV']:.4f}")

# Decomposition plot
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
