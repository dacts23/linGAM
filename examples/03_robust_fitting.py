"""
Example 03 — Observation Weights (Huber & Tukey) for Outlier Resistance
=======================================================================

Compute observation ``weights`` beforehand in a single pass using a
MAD-based scale estimate (sigma = MAD / 0.6745), then pass them to ``fit()``:

* **Huber weights** (c = 1.345):
  w_i = min(1, c / |u_i|).  Down-weights outliers proportionally (1/|u|)
  without zeroing them out.  Choose Huber for moderate tail noise or when
  extreme observations still carry partial signal.

* **Tukey biweight** (c = 4.685):
  w_i = (1 - (u_i / c)^2)^2 for |u_i| < c, else 0.  Completely rejects
  gross outliers beyond 4.685 sigma.  Choose Tukey for corrupted sensor
  spikes or data-entry errors that should have zero influence.
"""

import matplotlib.pyplot as plt
import numpy as np
from gamkit import LinGAM

np.random.seed(42)

x = np.sort(np.random.uniform(0, 10, 250))
y_true = np.sin(x)
y = y_true + np.random.normal(0, 0.2, len(x))

# Inject gross sensor spikes
outlier_idx = [20, 80, 150, 210]
y[outlier_idx] += np.array([9.0, -8.0, 7.5, -8.5])

# --- Precompute observation weights once using MAD-based sigma ---
r = y - np.median(y)
sigma = np.median(np.abs(r)) / 0.6745
u = np.abs(r) / sigma

w_huber = np.minimum(1.0, 1.345 / np.maximum(u, 1e-6))
w_tukey = np.where(u < 4.685, (1.0 - (u / 4.685) ** 2) ** 2, 1e-6)

# --- Fit Eilers-Marx & Wood models with precomputed weights ---
em_ols = LinGAM(n_splines=14, backend="eilers").fit(x, y)
em_hub = LinGAM(n_splines=14, backend="eilers").fit(x, y, weights=w_huber)
em_tuk = LinGAM(n_splines=14, backend="eilers").fit(x, y, weights=w_tukey)

wd_ols = LinGAM(k=14, bs="cr", backend="wood").fit(x, y)
wd_hub = LinGAM(k=14, bs="cr", backend="wood").fit(x, y, weights=w_huber)
wd_tuk = LinGAM(k=14, bs="cr", backend="wood").fit(x, y, weights=w_tukey)

rmse = lambda m: float(np.sqrt(np.mean((m.predict(x) - y_true) ** 2)))

print("=== RMSE vs True sin(x) (Single-Pass MAD Weights) ===")
print(f"{'Weights':<18}{'Eilers-Marx':>14}{'Wood':>12}{'Mean Outlier w':>18}")
print("-" * 62)
print(f"{'Unweighted (OLS)':<18}{rmse(em_ols):>14.4f}{rmse(wd_ols):>12.4f}{1.0:>18.4f}")
print(f"{'Huber (c=1.345)':<18}{rmse(em_hub):>14.4f}{rmse(wd_hub):>12.4f}{w_huber[outlier_idx].mean():>18.4f}")
print(f"{'Tukey (c=4.685)':<18}{rmse(em_tuk):>14.4f}{rmse(wd_tuk):>12.4f}{w_tukey[outlier_idx].mean():>18.6f}")

x_grid = np.linspace(0, 10, 500)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

for ax, title, m_ols, m_hub, m_tuk in [
    (axes[0], "Eilers-Marx Backend", em_ols, em_hub, em_tuk),
    (axes[1], "Wood Backend (REML)", wd_ols, wd_hub, wd_tuk),
]:
    ax.scatter(x, y, c="lightgray", s=15, label="Noisy data")
    ax.scatter(x[outlier_idx], y[outlier_idx], c="crimson", marker="x", s=50, label="Outliers")
    ax.plot(x_grid, np.sin(x_grid), "k:", lw=1.5, label="True sin(x)")
    ax.plot(x_grid, m_ols.predict(x_grid), "r--", lw=1.8, label=f"OLS (RMSE={rmse(m_ols):.3f})")
    ax.plot(x_grid, m_hub.predict(x_grid), "b-", lw=2.0, label=f"Huber (RMSE={rmse(m_hub):.3f})")
    ax.plot(x_grid, m_tuk.predict(x_grid), "g-", lw=2.0, label=f"Tukey (RMSE={rmse(m_tuk):.3f})")
    ax.set(xlabel="x", ylabel="y", title=title)
    ax.legend(fontsize=8)

fig.tight_layout()
plt.show()
