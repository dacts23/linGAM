# linGAM

**Simple Fast Linear GAM** — a minimal Python implementation of **Generalized Additive Models (GAMs)** using penalized B-splines. It supports both single-term smooths (`LinGAM`) and multi-term formula-based models (`GAMCore`) with automatic hyperparameter tuning via GCV grid search.

> **Full documentation:** [https://dacts.github.io/linGAM](https://dacts.github.io/linGAM)

## Features

- **Penalized B-spline regression** — De Boor recursion with linear extrapolation
- **Automatic hyperparameter tuning** — GCV grid search over splines and smoothing
- **Fast grid search** — QR precomputation + threaded Cholesky evaluation
- **Shape constraints** — monotonicity, convexity, concavity, periodicity
- **Robust fitting** — Huber-weighted IRLS for outlier resistance
- **Confidence & prediction intervals** — t-distribution-based
- **Tensor-product interactions** — multidimensional smooths via Khatri-Rao products

## Installation

```bash
pip install numpy scipy matplotlib
pip install -e .
```

## Quick Start

```python
from lingam import LinGAM
import numpy as np

x = np.linspace(0, 1, 200)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.3, len(x))

model = LinGAM()
model.gridsearch(x, y)

y_pred = model.predict(x)
ci = model.confidence_intervals(x, width=0.95)
model.plot_decomposition(x, y)
```

### Multi-term formula interface

```python
from lingam import GAMCore

model = GAMCore("s(0, n_splines=12) + l(1)")
model.fit(x, y)
model.gridsearch(x, y)
```

## Documentation

All API details, mathematical foundations, examples, and advanced usage are hosted on **GitHub Pages**:

### [→ https://dacts.github.io/linGAM](https://dacts.github.io/linGAM)

## Demo

```bash
python -m lingam
```

Runs a synthetic benchmark with standard and robust fits, and generates fit and decomposition plots.

## License

MIT © 2026 David Ávila Cortés
