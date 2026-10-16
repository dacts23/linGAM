# gamkit

**Modern Pythonic Generalized Additive Models (GAMs)** based on classical statistical literature. It supports both single-term smooths (`GAM` / `LinGAM`) and multi-term formula-based models (`GAMCore`) with automatic hyperparameter tuning.

> **Full documentation:** [https://dacts23.github.io/linGAM/](https://dacts23.github.io/linGAM/)

## Features

- **Dual backend engines** — choose between `eilers` (Eilers & Marx 1996: fast B-splines, discrete difference penalties, GCV grid search, shape constraints, Huber IRLS) and `wood` (Wood 2011: continuous REML smoothing parameter estimation, multiple basis families, and Wood 2013 hypothesis tests).
- **Penalized B-spline regression** — De Boor recursion with linear extrapolation
- **Automatic hyperparameter tuning** — GCV grid search over splines and smoothing (Eilers-Marx) or continuous REML/GCV optimization during fit (Wood)
- **Comprehensive basis suite (Wood)** — Thin plate (`tp`, `ts`), Cubic regression (`cr`, `cs`), Cyclic cubic (`cc`), P-splines (`ps`, `cp`), B-splines (`bs`), Random effects (`re`), and Tensor products (`te`)
- **Shape constraints** — monotonicity, convexity, concavity, periodicity
- **Robust fitting** — Huber-weighted IRLS for outlier resistance
- **Confidence & prediction intervals** — Bayesian and frequentist intervals
- **Summary tables** — approximate F-statistics and p-values (Wood 2013)
- **Tensor-product interactions** — multidimensional smooths via Khatri-Rao and Kronecker sum penalties

## Installation

```bash
pip install numpy scipy matplotlib
pip install -e .
```

## Quick Start

### 1. Wood (2011) Engine (Automatic REML smoothing)

```python
from gamkit import GAM, GAMCore
import numpy as np

x = np.linspace(0, 1, 200)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))

# In Wood backend, lambda (sp) is estimated automatically during fit via REML
model = GAM(k=10, bs="cr", backend="wood")
model.fit(x, y)

print(model.summary())
y_pred = model.predict(x)
ci = model.confidence_intervals(x, width=0.95)
```

Multi-term formula with tensor products:

```python
gam = GAMCore("te(0, 1, bs='cr', k=4) + l(2)", backend="wood")
gam.fit(X, y)
print(gam.summary())
```

### 2. Eilers & Marx (1996) Engine (B-spline GCV Grid Search)

```python
from gamkit import GAM

model = GAM(backend="eilers")
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

GNU General Public License v3.0 (GPL-3.0-or-later) © 2026 David Ávila Cortés.
See [LICENSE](LICENSE) for full terms and third-party attributions (Simon N. Wood / `mgcv` and Daniel Serven / `pygam`).
