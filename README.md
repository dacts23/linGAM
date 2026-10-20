# magli: Generalized Additive Models (GAMs) in Python

**magli (Modelo Aditivo Generalizado Lineal)**: Modern Pythonic Generalized Additive Models (GAMs) based on classical statistical literature. It supports both single-term smooths (`GAM` / `LinGAM`) and multi-term formula-based models (`GAMCore`) with automatic hyperparameter tuning.

> **Full documentation:** [https://dacts23.github.io/magli/](https://dacts23.github.io/magli/)

---

## Which Backend Engine Should I Use?

`magli` provides two complementary statistical engines under a unified API:

| Feature / Goal | **`backend="wood"`** (Default) | **`backend="eilers"`** |
| :--- | :--- | :--- |
| **Smoothing parameter ($\lambda$)** | **Continuous REML or GCV optimization** (automatic during `.fit()`, no grid search needed) | **Discrete GCV grid search** over candidate $(\lambda, K)$ pairs |
| **Basis function suite** | Thin plate (`tp`, `ts`), Cubic (`cr`, `cs`), Cyclic (`cc`), P-splines (`ps`, `cp`), B-splines (`bs`), Random effects (`re`) | Uniform B-splines (`bs`) with linear extrapolation |
| **Statistical summaries** | Full summary tables with **approximate F-tests and p-values** (Wood 2013) | Basic statistics (EDF, GCV, deviance explained) |
| **Shape constraints** | Periodic boundaries via cyclic bases (`cc`, `cp`) | **Soft constraints**: `mono_inc`, `mono_dec`, `convex`, `concave`, `periodic` |
| **Observation weights** | Weighted penalized least-squares (`weights=...`) | Weighted penalized least-squares (`weights=...`) |

* **Use `wood`** for general regression, automatic parameter tuning, hypothesis testing, and maximum flexibility in basis types (equivalent to R's `mgcv`).
* **Use `eilers`** when you have physical domain constraints (e.g. monotonicity or convexity) or want explicit B-spline grid sweeps.

---

## Why magli?

* **R's `mgcv` parity in pure Python**: `magli` implements Simon Wood's continuous REML smoothing parameter selection, Wood 2013 approximate F-tests and p-values, thin plate splines, and cyclic cubic bases without requiring an R runtime or rpy2.
* **Shape constraints**: Soft shape constraints (monotonic increasing, monotonic decreasing, convex, concave) via the Eilers-Marx B-spline engine.
* **Fast and numerically stable**: SVD data augmentation and cached QR with Cholesky factorization avoid squaring condition numbers.
* **Interpretable machine learning**: Every term produces clean component decomposition plots and Bayesian confidence intervals.

---

## Features

- **Dual backend engines**: choose between `wood` (Wood 2011) and `eilers` (Eilers & Marx 1996)
- **Automatic hyperparameter tuning**: continuous REML/GCV optimization or cached QR+Cholesky GCV grid search
- **Rich basis suite**: thin plate splines, cubic regression splines, cyclic cubic splines, P-splines, B-splines, random effects
- **Tensor-product interactions**: multidimensional smooths (`te`) with Kronecker sum penalties
- **Shape constraints**: monotonic increasing/decreasing, convex, concave, and periodic
- **Observation weights**: prior precision or frequency weights (`weights=...`) in `fit()` and `gridsearch()`
- **Uncertainty quantification**: Bayesian posterior covariance and frequentist confidence/prediction intervals
- **Summary tables**: parametric coefficients, EDF, approximate F-statistics, and p-values (Wood 2013)

---

## Installation

```bash
pip install magli
```

Or install from source for development:

```bash
git clone https://github.com/dacts23/magli.git
cd magli
pip install -e .
```

---

## Quick Start

### 1. Wood (2011) Engine (Automatic REML Smoothing)

In the Wood engine, smoothing parameters are estimated continuously during fit via REML (no manual grid search required):

```python
import numpy as np
from magli import GAM, GAMCore

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
from magli import GAM

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

Fitting with observation weights (e.g. inverse variance or down-weighted outliers):

```python
# Weighted penalized least-squares
weighted_model = GAM(n_splines=12, backend="eilers")
weighted_model.fit(x, y, weights=w)
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

### [→ https://dacts23.github.io/magli/](https://dacts23.github.io/magli/)

- **[Quick Start](https://dacts23.github.io/magli/quickstart.html)**: installation and basic workflows
- **[Mathematical Foundations](https://dacts23.github.io/magli/math.html)**: theory behind both engines:
  - *Part I:* [Shared Foundations & Additive Decomposition](https://dacts23.github.io/magli/math/foundations.html)
  - *Part II:* [Eilers & Marx B-Splines, Penalties, & SVD Solver](https://dacts23.github.io/magli/math/eilers.html)
  - *Part III:* [Wood Basis Suite, Identifiability, & REML Optimization](https://dacts23.github.io/magli/math/wood.html)
  - *Part IV:* [Side-by-Side Comparison & Engine Selection](https://dacts23.github.io/magli/math/comparison.html)
- **[Shape Constraints](https://dacts23.github.io/magli/constraints.html)**: monotonic, convex, concave, and periodic splines
- **[Smoothing Optimization](https://dacts23.github.io/magli/gridsearch.html)**: continuous REML and discrete GCV grid search
- **[Examples & Case Studies](https://dacts23.github.io/magli/examples.html)**: worked examples across diverse problems
- **[API Reference](https://dacts23.github.io/magli/api.html)**: full class and method documentation
- **[Changelog](https://dacts23.github.io/magli/changelog.html)**: release history and version notes

---

## Interactive Tutorials & Demo Scripts

Step through the interactive Jupyter notebooks in [`examples/`](examples):

| Notebook | What it covers |
| :--- | :--- |
| [`gam_tutorial.ipynb`](examples/gam_tutorial.ipynb) | Hands-on GAM tour: underfitting vs. overfitting, EDF interpretation, 2-D tensor interactions, shape constraints |
| [`wood_backend_math_tutorial.ipynb`](examples/wood_backend_math_tutorial.ipynb) | The Wood backend from the inside: basis zoo, `S.scale` and sum-to-zero absorption, continuous REML/GCV optimisation, Wood (2013) approximate F-tests, and an end-to-end workflow |
| [`eilers_backend_math_tutorial.ipynb`](examples/eilers_backend_math_tutorial.ipynb) | The Eilers-Marx backend from the inside: De Boor B-splines, difference penalties, the SVD data-augmentation solver, GCV grid search, shape constraints, and the same workflow |

Every formula in the two backend notebooks is followed by code that asserts it
numerically, and each closes with a pass/fail audit of its identities.
(Running notebooks needs the Jupyter stack: pip install -e .[notebooks].)

Or run any of the self-contained demo scripts:

```bash
python examples/01_basic_lingam.py
python examples/10_engineering_case_study.py
python examples/11_wood_backend.py
python examples/12_concrete_demo.py
python examples/13_eilers_vs_wood_comparison.py
python examples/14_wood_basis_decomposition_plots.py
python examples/15_eilers_basis_decomposition_plots.py
```

---

## License

GNU General Public License v3.0 (GPL-3.0-or-later) © 2026 David Ávila Cortés.
See [LICENSE](LICENSE) for full terms and third-party attributions (Simon N. Wood / `mgcv` and Daniel Serven / `pygam`).
