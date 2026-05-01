# linGAM

**Simple Fast Linear GAM** — a minimal, single-file Python implementation of a **Generalized Additive Model (GAM)** using penalized B-splines. It is a numerically exact equivalent of [pyGAM](https://github.com/dswah/pyGAM)'s `LinearGAM` for a single smooth term, with automatic hyperparameter selection via GCV grid search.

## Features

- **Penalized B-spline regression** — De Boor recursion for constructing the B-spline basis
- **Automatic hyperparameter tuning** — GCV grid search over number of splines and smoothing parameter λ
- **Fast mode** — QR precomputation + Cholesky decomposition for ~2× faster grid search
- **Robust fitting** — Huber-weighted Iteratively Reweighted Least Squares (IRLS) to handle outliers
- **Confidence & prediction intervals** — t-distribution-based intervals
- **Basis decomposition visualization** — plot basis functions, coefficients, and weighted-basis sum

## Installation

```bash
pip install numpy scipy matplotlib
```

Then copy `linGAM.py` into your project.

## Quick Start

```python
from linGAM import LinGAM
import numpy as np

# Generate synthetic data
x = np.linspace(0, 1, 200)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.3, len(x))

# Fit with automatic GCV grid search
model = LinGAM()
model.gridsearch(x, y, fast=True)

# Predict and get intervals
y_pred = model.predict(x)
ci = model.confidence_intervals(x, width=0.95)   # CI for the mean
pi = model.prediction_intervals(x, width=0.95)    # PI for new observations

# Visualize B-spline decomposition
model.plot_decomposition(x, y)
```

### With robust fitting for outlier-heavy data

```python
model = LinGAM()
model.gridsearch(x, y, fast=True, robust=True)
```

## API Reference

```python
# Constructor
model = LinGAM(n_splines=10, lam=1.0, spline_order=3)

# Fit with user-supplied hyperparameters
model.fit(x, y, robust=False)

# Automatic GCV grid search over (n_splines, lam)
model.gridsearch(x, y,
    lam=None,           # candidate lambdas (default: np.logspace(-3, 3, 11))
    n_splines=None,     # candidate K values  (default: np.arange(5, 25, 2))
    gamma=1.4,          # EDF penalty multiplier in GCV
    fast=True,          # use QR+Cholesky fast path
    robust=False)       # enable Huber IRLS

# Predict
y_pred = model.predict(x)

# Intervals (shape: n × 2)
model.confidence_intervals(x, width=0.95)
model.prediction_intervals(x, width=0.95)

# Diagnostics
model.is_fitted            # bool
model.coef_                # fitted coefficients (shape: K+1, includes intercept)
model.statistics_          # dict: edof, edof_per_coef, scale, cov, se, rss, GCV
model.knots_               # knot locations (in original x-scale)
model.gcv_results_         # list of all (GCV, K, lam, coef, edof) evaluated
```

### Hyperparameters

| Parameter   | Type  | Default | Description |
|------------|-------|---------|-------------|
| `n_splines` | `int` | 10 | Number of B-spline basis functions *K*. Must satisfy *K > spline_order*. |
| `lam`       | `float` | 1.0 | Smoothing penalty λ ≥ 0. Larger values produce flatter curves. |
| `spline_order` | `int` | 3 | B-spline order *k* (polynomial degree). Cubic splines correspond to *k = 3*. |
| `gamma`     | `float` | 1.4 | Multiplicative penalty on the effective degrees of freedom in GCV. |

## Mathematical Formulation

### 1. Model

The model assumes the relationship between a predictor *x* and response *y* is a smooth, additive function:

$$y = f(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

The smooth function *f(x)* is approximated as a linear combination of *K* B-spline basis functions plus an intercept term:

$$f(x) = \sum_{j=1}^{K} \beta_j B_j(x) + \beta_0$$

Equivalently, for *n* observations:

$$\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

where **X** is the *n* × *(K+1)* design matrix whose columns are the B-spline basis functions evaluated at each *xᵢ*, plus a column of ones for the intercept.

### 2. B-Spline Basis

The basis matrix **B(x)** is constructed using **De Boor's recursion**:

$$
B_{j,1}(x) = \begin{cases} 1 & \text{if } t_j \le x < t_{j+1} \\ 0 & \text{otherwise} \end{cases}
$$

$$
B_{j,m}(x) = \frac{x - t_j}{t_{j+m-1} - t_j} B_{j,m-1}(x) + \frac{t_{j+m} - x}{t_{j+m} - t_{j+1}} B_{j+1,m-1}(x)
$$

for *m = 2, …, k+1*, where *tⱼ* are the knot sequence. The implementation uses a fully vectorized form of this recurrence.

**Extrapolation** beyond the boundary knots is handled by linear extension using the gradient of the highest-order basis functions at the boundaries:

$$B_j(x) = \begin{cases} \nabla B_j(0) \cdot x + B_j(0) & x < 0 \\ \nabla B_j(1) \cdot (x - 1) + B_j(1) & x > 1 \end{cases}$$

### 3. Penalized Least Squares

To prevent overfitting, a roughness penalty is added. Fitting minimizes the penalized residual sum of squares:

$$\mathcal{L}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \,\boldsymbol{\beta}^T \mathbf{S} \,\boldsymbol{\beta}$$

The penalty matrix **S** is derived from the second-difference operator **D**₂:

$$\mathbf{D}_2 = \begin{bmatrix} 
1 & -2 & 1 & 0 & \cdots & 0 \\
0 & 1 & -2 & 1 & \cdots & 0 \\
\vdots & \ddots & \ddots & \ddots & \ddots & \vdots \\
0 & \cdots & 0 & 1 & -2 & 1
\end{bmatrix} \in \mathbb{R}^{(K-2) \times K}$$

$$\mathbf{S} = \mathbf{D}_2^T \mathbf{D}_2$$

The full *(K+1) × (K+1)* penalty matrix **P** is block-diagonal, padding **S** with zeros for the unpenalized intercept:

$$\mathbf{P} = \begin{bmatrix} \lambda \mathbf{S} & \mathbf{0} \\ \mathbf{0} & 0 \end{bmatrix}$$

The minimizer is the solution to the penalized normal equations:

$$(\mathbf{X}^T \mathbf{X} + \mathbf{P}) \hat{\boldsymbol{\beta}} = \mathbf{X}^T \mathbf{y}$$

### 4. Numerical Solution via Data Augmentation

**Why data augmentation?** Forming **X**ᵀ**X** directly squares the condition number — even mild near-dependency among basis functions gets amplified into numerical instability. Instead, the solver treats the penalty as additional "fake data rows" that softly pull coefficients toward zero, solving the augmented system without ever forming **X**ᵀ**X**.

A tiny ridge term δ**I** (δ = √ε₁ₐₕ) is added to **P** before Cholesky factorization — this ensures positive-definiteness even when λ = 0 (no penalty), preventing the Cholesky from failing. The penalty is factorised:

$$\mathbf{P} + \delta\mathbf{I} = \mathbf{E}^T \mathbf{E}, \quad \mathbf{E} = \mathrm{chol}(\mathbf{P} + \delta \mathbf{I})^T$$

The QR decomposition of **X** is computed, separating orthogonal structure (**Q**) from scale (**R**):

$$\mathbf{X} = \mathbf{Q} \mathbf{R}$$

The stacked matrix **[R; E]** is then decomposed via SVD — the most numerically stable way to solve linear systems, gracefully handling near-singular matrices by truncating tiny singular values:

$$\begin{bmatrix} \mathbf{R} \\ \mathbf{E} \end{bmatrix} = \mathbf{U} \mathbf{D} \mathbf{V}^T$$

The solution reduces to:

$$\hat{\boldsymbol{\beta}} = \mathbf{V} \mathbf{D}^{-1} \mathbf{U}_1^T \mathbf{Q}^T \mathbf{y}$$

where **U**₁ is the top portion of **U** corresponding to the data matrix **R** (the bottom portion corresponds to **E** and is not needed for solving). The component matrix:

$$\mathbf{B}_{\text{solve}} = \mathbf{V} \mathbf{D}^{-1} \mathbf{U}_1^T \mathbf{Q}^T$$

is the "inverse operator" mapping any **y** to its coefficient vector **β̂**. It is stored and reused for the covariance:

$$\mathrm{Cov}(\hat{\boldsymbol{\beta}}) = \hat{\sigma}^2 \cdot \mathbf{B}_{\text{solve}} \mathbf{B}_{\text{solve}}^T$$

### 5. Effective Degrees of Freedom (EDF)

The EDF measures the flexibility consumed by the smooth term. It is computed from the SVD components:

$$\text{EDF} = \sum_{i} \sum_{j} (\mathbf{U}_1)_{ij}^2$$

In the fast grid search (Cholesky path), EDF is computed via the trace of the hat sub-matrix:

$$\text{EDF} = \mathrm{tr}\!\left( (\mathbf{R}^T \mathbf{R} + \lambda \mathbf{S})^{-1} \mathbf{R}^T \mathbf{R} \right)$$

### 6. Generalized Cross Validation (GCV)

Given the flexibility of the spline, the bias-variance tradeoff is guided by GCV:

$$\text{GCV}(\lambda, K) = \frac{n \cdot \text{RSS}}{(n - \gamma \cdot \text{EDF})^2}$$

where:

$$\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

The parameter *γ* (default 1.4) multiplies the EDF in the denominator, acting as a heavier penalty on model complexity to prevent overfitting. The optimal hyperparameters (*K*, *λ*) are those that minimize GCV.

### 7. Grid Search

The GCV objective is evaluated over a Cartesian product of *K* (number of splines) and *λ* (smoothing parameter) candidates.

**Fast mode** (default) uses a two-phase approach:

- **Phase 1** — For each candidate *K*, precompute the QR decomposition of **X** and the quantities **R**ᵀ**R** and **R**ᵀ**Q**ᵀ**y**.
- **Phase 2** — For each (*K*, *λ*) pair, build the penalized system and solve using **Cholesky factorization**:

$$\mathbf{B} = \mathbf{R}^T \mathbf{R} + \lambda \mathbf{S}$$

$$\mathbf{c} = \mathrm{cho\_factor}(\mathbf{B})$$

$$\hat{\boldsymbol{\beta}} = \mathrm{cho\_solve}(\mathbf{c}, \mathbf{R}^T \mathbf{Q}^T \mathbf{y})$$

$$\text{EDF} = \mathrm{tr}(\mathrm{cho\_solve}(\mathbf{c}, \mathbf{R}^T \mathbf{R}))$$

Cholesky is ~2× faster than LU decomposition. Both phases are multithreaded.

**Basic mode** uses multithreading only over *λ* within each *K* and uses the full SVD-based solver.

### 8. Robust Fitting (Huber IRLS)

Ordinary least squares is easily thrown off by outliers — a single extreme point can yank the fitted curve far from the rest of the data. When `robust=True`, the fit uses **Huber-weighted Iteratively Reweighted Least Squares (IRLS)** to progressively down-weight outliers.

The **Huber loss** is a compromise between squared-error loss (sensitive to outliers) and absolute-error loss (robust but less efficient for well-behaved data):

- Points with small residuals (|r| ≤ c) are **fully trusted** — weight = 1 (uses OLS, which is statistically optimal for Gaussian errors).
- Points with large residuals (|r| > c) are **down-weighted** as c / |r| — the bigger the residual, the less influence the point has.

The procedure iterates:

1. **Initialize** residuals around the global median: *rᵢ = yᵢ − median(y)*. The median is naturally robust to outliers, giving a much better starting point than the mean.

2. **Estimate scale** via Median Absolute Deviation (MAD). The MAD is a robust alternative to standard deviation — it measures spread using medians instead of means, making it largely immune to outliers. The constant 0.6745 converts MAD to a consistent estimate of σ for Gaussian errors:

$$\hat{\sigma}_{\text{MAD}} = \frac{\mathrm{median}(|r_i - \mathrm{median}(r_i)|)}{0.6745}, \quad 0.6745 = \Phi^{-1}(3/4)$$

3. **Compute Huber weights** with tuning constant *c = 1.345 · σ̂*:

$$w_i = \begin{cases} 1 & \text{if } |r_i| \le c \\ \dfrac{c}{|r_i|} & \text{if } |r_i| > c \end{cases}$$

The constant 1.345 gives approximately **95% asymptotic efficiency** when the errors are actually Gaussian — meaning we lose almost nothing by being robust, but gain a great deal when outliers are present.

4. **Solve weighted penalized least squares** by applying √w to each row of **X** and **y**:

$$\mathbf{X}_w = \sqrt{\mathbf{w}} \odot \mathbf{X}, \quad \mathbf{y}_w = \sqrt{\mathbf{w}} \odot \mathbf{y}$$

$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left(\|\mathbf{y}_w - \mathbf{X}_w \boldsymbol{\beta}\|^2 + \lambda \,\boldsymbol{\beta}^T \mathbf{S} \,\boldsymbol{\beta}\right)$$

Row-weighting with √w avoids forming weighted normal equations (**X**ᵀ**WX**), which would square the condition number. Instead, the weighted matrices pass directly into the QR+SVD solve pipeline, preserving numerical stability.

5. **Recompute residuals** on the original (unweighted) **X**: *r = y − Xβ̂*. This ensures the weights in the next iteration reflect actual prediction error, not weighted error. Repeat for up to 15 iterations (robust fit) or 10 iterations (robust grid search).

### 9. Interval Estimation

Both confidence and prediction intervals use the **t-distribution** with *n −* EDF degrees of freedom.

**Confidence interval** for the mean *f(x)* — captures uncertainty in the fitted curve:

$$\hat{y} \pm t_{1-\alpha/2,\, \nu} \cdot \sqrt{\mathbf{x}^T \mathrm{Cov}(\hat{\boldsymbol{\beta}})\, \mathbf{x}}$$

**Prediction interval** for a new observation — additionally includes residual variance:

$$\hat{y} \pm t_{1-\alpha/2,\, \nu} \cdot \sqrt{\hat{\sigma}^2 + \mathbf{x}^T \mathrm{Cov}(\hat{\boldsymbol{\beta}})\, \mathbf{x}}$$

where *ν = n −* EDF and *σ̂²* is the residual scale.

### 10. Diagnostics

The `statistics_` dictionary stores:

| Key              | Description |
|------------------|-------------|
| `edof`           | Effective degrees of freedom |
| `edof_per_coef`  | Per-coefficient contribution to EDF |
| `scale`          | Residual scale *σ̂* |
| `cov`            | Coefficient covariance matrix (*(K+1) × (K+1)*) |
| `se`             | Standard errors of coefficients |
| `rss`            | Residual sum of squares |
| `n_samples`      | Number of observations |
| `GCV`            | GCV score at the fitted hyperparameters (using *γ = 1.4*) |

## Demo

```bash
python linGAM.py
```

Runs a synthetic benchmark with clustered data and artificially injected outliers, performing both standard and robust GCV grid searches, and optionally comparing with pyGAM. Generates a fit plot (with 95% CI and PI bands) and a basis decomposition figure.

## License

MIT © 2026 David Avila Cortes
