# linGAM

**Simple Fast Linear GAM** — a minimal Python implementation of **Generalized Additive Models (GAMs)** using penalized B-splines. It supports both single-term smooths (`LinGAM`) and multi-term formula-based models (`GAMCore`) with automatic hyperparameter tuning via GCV grid search.

> **Full documentation:** [https://dacts23.github.io/linGAM/](https://dacts23.github.io/linGAM/)

## Features

- **Dual backend engines** — choose between `pygam` (fast B-splines, GCV grid search, shape constraints, Huber IRLS) and `mgcv` (faithful R mgcv clone with REML smoothing parameter estimation, multiple basis families, and Wood 2013 hypothesis tests).
- **Penalized B-spline regression** — De Boor recursion with linear extrapolation
- **Automatic hyperparameter tuning** — GCV grid search over splines and smoothing (pyGAM) or continuous REML/GCV optimization during fit (mgcv)
- **Comprehensive basis suite (mgcv)** — Thin plate (`tp`, `ts`), Cubic regression (`cr`, `cs`), Cyclic cubic (`cc`), P-splines (`ps`, `cp`), B-splines (`bs`), Random effects (`re`), and Tensor products (`te`)
- **Shape constraints** — monotonicity, convexity, concavity, periodicity
- **Robust fitting** — Huber-weighted IRLS for outlier resistance
- **Confidence & prediction intervals** — Bayesian and frequentist intervals
- **R-style summary tables** — approximate F-statistics and p-values matching `summary.gam`
- **Tensor-product interactions** — multidimensional smooths via Khatri-Rao and Kronecker sum penalties

## Installation

```bash
pip install numpy scipy matplotlib
pip install -e .
```

## Quick Start

### 1. R `mgcv` Engine (Automatic REML smoothing)

```python
from lingam import LinGAM, GAMCore
import numpy as np

x = np.linspace(0, 1, 200)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))

# In mgcv backend, lambda (sp) is estimated automatically during fit via REML
model = LinGAM(k=10, bs="cr", backend="mgcv")
model.fit(x, y)

print(model.summary())
y_pred = model.predict(x)
ci = model.confidence_intervals(x, width=0.95)
```

Multi-term formula with tensor products:

```python
gam = GAMCore("te(0, 1, bs='cr', k=4) + l(2)")
gam.fit(X, y)
print(gam.summary())
```

### 2. PyGAM Engine (B-spline GCV Grid Search)

```python
from lingam import LinGAM

model = LinGAM(backend="pygam")
model.gridsearch(x, y)

y_pred = model.predict(x)
ci = model.confidence_intervals(x, width=0.95)
model.plot_decomposition(x, y)
```

## Documentation

All API details, mathematical foundations, examples, and advanced usage are hosted on **GitHub Pages**:

### [→ https://dacts23.github.io/linGAM/](https://dacts23.github.io/linGAM/)

## Demo

```bash
python examples/12_concrete_demo.py
```

## License

MIT © 2026 David Ávila Cortés
