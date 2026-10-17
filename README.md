# gamkit

**Modern Pythonic Generalized Additive Models (GAMs)** based on classical statistical literature. It supports both single-term smooths (`GAM` / `LinGAM`) and multi-term formula-based models (`GAMCore`) with automatic hyperparameter tuning.

> **Full documentation:** [https://dacts23.github.io/linGAM/](https://dacts23.github.io/linGAM/)

---

## Which Backend Engine Should I Use?

`gamkit` provides two complementary statistical engines under a unified API:

| Feature / Goal | **`backend="wood"`** (Default) | **`backend="eilers"`** |
| :--- | :--- | :--- |
| **Smoothing parameter ($\lambda$)** | **Continuous REML or GCV optimization** (automatic during `.fit()`, no grid search needed) | **Discrete GCV grid search** over candidate $(\lambda, K)$ pairs |
| **Basis function suite** | Thin plate (`tp`, `ts`), Cubic (`cr`, `cs`), Cyclic (`cc`), P-splines (`ps`, `cp`), Random effects (`re`) | Uniform B-splines (`bs`) with linear extrapolation |
| **Statistical summaries** | Full summary tables with **approximate F-tests and p-values** (Wood 2013) | Basic statistics (EDF, GCV, deviance explained) |
| **Shape constraints** | Periodic boundaries via cyclic bases (`cc`, `cp`) | **Soft constraints**: `mono_inc`, `mono_dec`, `convex`, `concave`, `periodic` |
| **Outlier resistance** | Standard penalized least-squares | **Huber IRLS** robust regression (`robust=True`) |

* **Use `wood`** for general regression, automatic parameter tuning, hypothesis testing, and maximum flexibility in basis types (equivalent to R's `mgcv`).
* **Use `eilers`** when you have physical domain constraints (e.g. monotonicity or convexity), heavy sensor noise/outliers, or want explicit B-spline grid sweeps.

---

## Features

- **Dual backend engines** — choose between `wood` (Wood 2011) and `eilers` (Eilers & Marx 1996)
- **Automatic hyperparameter tuning** — continuous REML/GCV optimization or cached QR+Cholesky GCV grid search
- **Rich basis suite** — thin plate splines, cubic regression splines, cyclic cubic splines, P-splines, B-splines, random effects
- **Tensor-product interactions** — multidimensional smooths (`te`) with Kronecker sum penalties
- **Shape constraints** — monotonic increasing/decreasing, convex, concave, and periodic
- **Robust fitting** — Huber-weighted iteratively reweighted least squares (IRLS)
- **Uncertainty quantification** — Bayesian posterior covariance and frequentist confidence/prediction intervals
- **Summary tables** — parametric coefficients, EDF, approximate F-statistics, and p-values (Wood 2013)

---

## Installation

```bash
pip install numpy scipy matplotlib
pip install -e .
```

---

## Quick Start

### 1. Wood (2011) Engine (Automatic REML Smoothing)

In the Wood engine, smoothing parameters are estimated continuously during fit via REML (no manual grid search required):

```python
import numpy as np
from gamkit import GAM, GAMCore

np.random.seed(42)
x = np.linspace(0, 1, 200)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))

# Single-term model with cubic regression splines
model = GAM(k=10, bs="cr", backend="wood")
model.fit(x, y)

print(model.summary())
y_pred = model.predict(x)
ci = model.confidence_intervals(x, width=0.95)
```

Multi-term formula with tensor-product interactions and linear terms:

```python
# Multi-term dataset: 2 interactive variables + 1 linear variable
X = np.random.uniform(0, 10, (200, 3))
y_multi = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 1.2 * X[:, 2] + np.random.normal(0, 0.2, 200)

gam = GAMCore("te(0, 1, bs='cr', k=4) + l(2)", backend="wood")
gam.fit(X, y_multi)
print(gam.summary())
```

### 2. Eilers & Marx (1996) Engine (B-Splines & Constraints)

The Eilers-Marx engine provides discrete GCV grid searches and shape constraints:

```python
import numpy as np
from gamkit import GAM

# Reusing synthetic data x, y or generating new series
x = np.linspace(0, 1, 200)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))

# Fast B-spline grid search over candidate lambda and spline counts
model = GAM(backend="eilers")
model.gridsearch(x, y)

y_pred = model.predict(x)
ci = model.confidence_intervals(x, width=0.95)
model.plot_decomposition(x, y)
```

Enforcing soft shape constraints (e.g. monotonic increasing dose-response):

```python
# Enforce monotonic increasing trend
mono_model = GAM(n_splines=12, constraint="mono_inc", backend="eilers")
mono_model.fit(x, y)
```

Robust fitting against sensor outliers:

```python
# Huber IRLS fitting (outlier-tolerant)
robust_model = GAM(n_splines=12, backend="eilers")
robust_model.fit(x, y, robust=True)
```

---

## Formula Syntax

`GAMCore` uses an intuitive additive formula specification:

| Term Syntax | Meaning | Example |
| :--- | :--- | :--- |
| `s(col, ...)` | Univariate smooth spline | `s(0, k=10, bs='cr')` or `s(0, n_splines=12, constraint='mono_inc')` |
| `te(c1, c2, ...)` | Multidimensional tensor product | `te(0, 1, k=5, bs='cr')` |
| `f(col, ...)` | Categorical factor | `f(2, coding='dummy')` or `f(2, coding='one-hot')` |
| `l(col, ...)` | Parametric linear effect | `l(3)` |

*Note: Predictor variables are identified by 0-indexed column numbers in input arrays.*

---

## Documentation

Full documentation, mathematical foundations, and end-to-end tutorials are available at:

### [→ https://dacts23.github.io/linGAM/](https://dacts23.github.io/linGAM/)

- **[Quick Start](https://dacts23.github.io/linGAM/quickstart.html)** — installation and basic workflows
- **[Mathematical Foundations](https://dacts23.github.io/linGAM/math.html)** — theory behind both engines:
  - *Part I:* Shared Foundations & Additive Decomposition
  - *Part II:* Eilers & Marx B-Splines, Penalties, & SVD Solver
  - *Part III:* Wood Basis Suite, Identifiability, & REML Optimization
  - *Part IV:* Side-by-Side Comparison & Engine Selection
- **[Shape Constraints](https://dacts23.github.io/linGAM/constraints.html)** — monotonic, convex, concave, and periodic splines
- **[Smoothing Optimization](https://dacts23.github.io/linGAM/gridsearch.html)** — continuous REML and discrete GCV grid search
- **[Examples & Case Studies](https://dacts23.github.io/linGAM/examples.html)** — worked examples across diverse problems
- **[API Reference](https://dacts23.github.io/linGAM/api.html)** — full class and method documentation

---

## Demo Scripts

Run any of the 14 self-contained demo scripts in the `examples/` directory:

```bash
python examples/01_basic_lingam.py
python examples/10_engineering_case_study.py
python examples/11_wood_backend.py
python examples/12_concrete_demo.py
python examples/13_eilers_vs_wood_comparison.py
python examples/14_wood_basis_decomposition_plots.py
```

---

## License

GNU General Public License v3.0 (GPL-3.0-or-later) © 2026 David Ávila Cortés.
See [LICENSE](LICENSE) for full terms and third-party attributions (Simon N. Wood / `mgcv` and Daniel Serven / `pygam`).
