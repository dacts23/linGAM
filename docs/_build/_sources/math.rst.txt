Mathematical Foundations
========================

This chapter derives the core mathematics behind linGAM: B-spline bases,
penalised least squares, the SVD data-augmentation solver, effective degrees
of freedom, and Generalised Cross Validation (GCV).


B-spline bases
--------------

A B-spline of order :math:`k` (polynomial degree :math:`k-1`) is built from
:math:`K` basis functions via the De Boor recursion.  Given data
:math:`x \in [x_{\min}, x_{\max}]`, we first map to :math:`[0, 1]`:

.. math::

   z = \frac{x - x_{\min}}{x_{\max} - x_{\min}}.

Internal knots are placed uniformly:

.. math::

   \tau_0 = 0, \; \tau_1, \dots, \tau_{K-k+1} = 1.

The augmented knot sequence extends :math:`k` knots beyond each boundary.

**De Boor recursion.**  Start with order-1 (piecewise constant) bases:

.. math::

   B_{i,1}(z) = \mathbf{1}_{[\tau_i, \tau_{i+1})}(z).

For orders :math:`m = 2, \dots, k`:

.. math::

   B_{i,m}(z)
   = \frac{z - \tau_i}{\tau_{i+m-1} - \tau_i} \, B_{i,m-1}(z)
   + \frac{\tau_{i+m} - z}{\tau_{i+m} - \tau_{i+1}} \, B_{i+1,m-1}(z).

The result is an :math:`n \times K` design matrix :math:`B` with the property
that each row sums to 1 and each column has compact support (localised).

**Linear extrapolation.**  Outside :math:`[0,1]`, B-splines shoot to zero or
oscillate wildly.  linGAM replaces the B-spline values with a linear
extension using the boundary gradient:

.. math::

   f(z) \approx f(0) + f'(0) \, z \quad (z < 0),
   \qquad
   f(z) \approx f(1) + f'(1) \, (z-1) \quad (z > 1).

The derivative :math:`f'(0)` is computed analytically from the
:math:`(k-1)`-order basis saved during recursion.

**Periodic basis.**  When ``constraint='periodic'``, the basis is wrapped so
that the first :math:`k-1` columns are added to the last :math:`k-1`
columns.  This enforces :math:`f(0) = f(1)` and reduces the coefficient
count from :math:`K` to :math:`K - k + 1`.


Penalised least squares
-----------------------

The model is

.. math::

   \mathbf{y} = X \boldsymbol{\beta} + \boldsymbol{\varepsilon},

where :math:`X` stacks all term design matrices (plus an intercept column).
We minimise the penalised sum of squares:

.. math::

   S(\boldsymbol{\beta})
   = \|\mathbf{y} - X \boldsymbol{\beta}\|^2
   + \boldsymbol{\beta}^\top P \boldsymbol{\beta}.

The penalty :math:`P` is block-diagonal, with one block per term.  The
intercept block is zero (unpenalised).

**Second-difference penalty** (smoothness).  For a spline with :math:`K`
coefficients:

.. math::

   D_2 = \begin{pmatrix}
   1 & -2 & 1 &        &   \\
     &  1 & -2 & 1     &   \\
     &    & \ddots & \ddots & \ddots \\
   \end{pmatrix}_{(K-2) \times K},
   \qquad
   P = \lambda \, D_2^\top D_2.

This penalises curvature: neighbouring coefficients that differ wildly incur
a large cost.

**First-difference penalty** (monotonicity).  For ``constraint='mono_inc'``
or ``'mono_dec'``:

.. math::

   D_1 = \begin{pmatrix}
   -1 & 1 &   &   \\
      & -1 & 1 &   \\
      &    & \ddots & \ddots
   \end{pmatrix}_{(K-1) \times K},
   \qquad
   P = \lambda \, D_1^\top D_1.

Minimising :math:`\beta^\top D_1^\top D_1 \beta = \sum_i (\beta_{i+1} - \beta_i)^2`
forces adjacent coefficients to be similar, which encourages a monotonic
shape.  Combined with the natural ordering of B-spline basis functions,
this produces an increasing (or decreasing) smooth.

**Circular second-difference** (periodicity).  For
``constraint='periodic'``:

.. math::

   P_{ij} = \lambda \times \begin{cases}
   4  & i = j \\
  -2  & j = (i \pm 1) \bmod K_{\text{eff}} \\
   1  & j = (i \pm 2) \bmod K_{\text{eff}} \\
   0  & \text{otherwise}
   \end{cases}

This wraps the second-difference operator, connecting the first and last
coefficients so the curve is closed.

**Ridge penalty.**  For factor and linear terms:

.. math::

   P = \lambda I_K.


SVD data-augmentation solver
----------------------------

The normal equations are :math:`(X^\top X + P)\beta = X^\top y`.  Forming
:math:`X^\top X` squares the condition number and amplifies numerical
instability.  linGAM uses a **data-augmentation** trick instead.

1. **Cholesky of the penalty.**  Factor :math:`P + \varepsilon I = E^\top E`
   where :math:`\varepsilon = \sqrt{\text{machine epsilon}}` is a tiny ridge
   that guarantees positive definiteness even when :math:`\lambda = 0`.

2. **QR of the design matrix.**  :math:`X = QR` where :math:`Q` is
   orthogonal and :math:`R` is upper-triangular.  Because :math:`Q` preserves
   norms, working with :math:`R` (size :math:`m \times m`) instead of
   :math:`X` (size :math:`n \times m`) is numerically safe and cheaper when
   :math:`n \gg m`.

3. **SVD of the stacked matrix.**  Form :math:`[R; E]` and decompose:

   .. math::

      [R; E] = U \, \text{diag}(d) \, V^\top.

   This is equivalent to solving the augmented least-squares system

   .. math::

      \begin{pmatrix} X \\ E \end{pmatrix} \beta
      = \begin{pmatrix} y \\ 0 \end{pmatrix}

   but never forms :math:`X^\top X` explicitly.

4. **Solution operator.**  Truncate to the first :math:`\min(m,n)` singular
   values and compute:

   .. math::

      B_{\text{solve}} = V \, \text{diag}(d^{-1}) \, U_1^\top Q^\top,
      \qquad
      \hat\beta = B_{\text{solve}} \, y.

   Saving :math:`B_{\text{solve}}` lets us compute covariance and effective
   degrees of freedom later without re-fitting.

**Why this works.**  The stacked SVD is the most numerically stable way to
solve a penalised linear system.  Tiny singular values are handled gracefully
(by division, not inversion of a near-singular matrix), and the condition
number of :math:`[R; E]` is the square root of the condition number of
:math:`X^\top X + P`.


Effective degrees of freedom (EDF)
----------------------------------

In ordinary linear regression, the degrees of freedom equal the number of
parameters.  In a penalised spline, the penalty "shrinks" coefficients,
so each one uses *less* than one full degree of freedom.

The EDF is the trace of the **influence** (hat) matrix:

.. math::

   \text{EDF} = \text{tr}(H),
   \qquad
   H = X (X^\top X + P)^{-1} X^\top.

Using the SVD augmentation, this simplifies to

.. math::

   \text{EDF} = \sum_{i=1}^{\min(m,n)} \sigma_i(U_1)^2
   = \|U_1\|_F^2,

where :math:`U_1` is the top-left :math:`\min(m,n) \times \min(m,n)` block
of :math:`U`.  This is computed in a single NumPy call with no extra matrix
inversions.

Per-coefficient EDF tells us how much each individual parameter contributes:

.. math::

   \text{EDF}_j = \sum_i U_{1,ij}^2.


Generalised Cross Validation (GCV)
----------------------------------

Leave-one-out cross validation is expensive: fit the model :math:`n` times,
dropping one point each time.  GCV is a clever closed-form approximation:

.. math::

   \text{GCV} = \frac{n \cdot \text{RSS}}{(n - \gamma \cdot \text{EDF})^2},

where

* :math:`n` = number of observations,
* :math:`\text{RSS} = \sum_i (y_i - \hat y_i)^2` = residual sum of squares,
* :math:`\text{EDF}` = effective degrees of freedom,
* :math:`\gamma` = complexity penalty multiplier (default 1.4).

The denominator penalises model complexity.  When :math:`\text{EDF}` grows,
the denominator shrinks and GCV explodes — this prevents overfitting.
Setting :math:`\gamma > 1` (e.g. 1.4) is analogous to AIC/BIC corrections
that prefer slightly simpler models.

**Why GCV and not plain CV?**  GCV requires only a single fit.  It is
asymptotically equivalent to leave-one-out CV but computes in :math:`O(1)`
extra time after the fit.


Fast QR+Cholesky grid search
----------------------------

Grid search evaluates every combination of :math:`(K, \lambda)` candidates.
A naïve implementation rebuilds :math:`X` and solves from scratch for each
combination.  linGAM's fast path avoids redundant work.

**Phase 1 — Precompute per :math:`K` (threaded).**
For each candidate basis count :math:`K`:

1. Build :math:`X_K` and compute its QR decomposition: :math:`X_K = Q_K R_K`.
2. Form :math:`R_K^\top R_K` and :math:`R_K^\top Q_K^\top y`.
   These are **independent of :math:`\lambda`**.
3. Precompute the unscaled penalty basis :math:`S = D^\top D` (also
   independent of :math:`\lambda`).

All of this is done once per :math:`K` and cached.

**Phase 2 — Evaluate per :math:`(K, \lambda)` (threaded).**
For each combination:

1. Assemble the penalty: :math:`P = \lambda S`.
2. Build :math:`B = R_K^\top R_K + P`.
3. Cholesky solve: :math:`\hat\beta = B^{-1} (R_K^\top Q_K^\top y)`.
4. Compute EDF via :math:`\text{tr}(B^{-1} R_K^\top R_K)`.
5. Compute RSS and GCV.

Cholesky costs :math:`O(m^3/3)` versus :math:`O(m^3)` for SVD, and both
phases are fully parallel via ``concurrent.futures.ThreadPoolExecutor``.

**Precomputed penalty bases.**  For multi-term models, each term's penalty
is built from unscaled base matrices (e.g. :math:`D_2^\top D_2` at
:math:`\lambda = 1`).  During Phase 2, assembly is just element-wise
scaling: :math:`P_{ij} += \lambda_k \cdot S_{ij}`.  No matrix is rebuilt
from scratch.


Tensor-product interactions
---------------------------

A tensor term ``te(f1, f2, ..., fd)`` models interactions without assuming
additivity.  The design matrix is the **Khatri-Rao** (column-wise
Kronecker) product of marginal B-spline bases:

.. math::

   B_{\text{tensor}} = B_1 \odot B_2 \odot \dots \odot B_d,

where :math:`\odot` denotes column-wise product.  If :math:`B_i` is
:math:`n \times K_i`, the result is :math:`n \times (K_1 K_2 \dots K_d)`.

The penalty is the **Kronecker sum** of marginal penalties:

.. math::

   P_{\text{tensor}}
   = \lambda_1 (P_1 \otimes I_{K_2} \otimes \dots \otimes I_{K_d})
   + \lambda_2 (I_{K_1} \otimes P_2 \otimes \dots \otimes I_{K_d})
   + \dots
   + \lambda_d (I_{K_1} \otimes \dots \otimes I_{K_{d-1}} \otimes P_d).

Each direction can have its own smoothing parameter :math:`\lambda_i`.

For a 2-D tensor ``te(0, 1)`` with :math:`K_1 = K_2 = 6`, the design matrix
has :math:`n \times 36` columns and the penalty is :math:`36 \times 36`.


Huber IRLS (robust fitting)
---------------------------

Ordinary least squares is sensitive to outliers.  Robust fitting replaces the
squared loss with the **Huber loss**:

.. math::

   \rho(r) = \begin{cases}
   \tfrac12 r^2            & |r| \le c \\
   c(|r| - \tfrac12 c)     & |r| > c
   \end{cases}

where :math:`c = 1.345 \, \hat\sigma` and :math:`\hat\sigma = \text{MAD} / 0.6745`.
The constant 1.345 gives ~95% asymptotic efficiency under Gaussian errors.

The IRLS algorithm iterates:

1. Compute residuals :math:`r = y - X\hat\beta`.
2. Compute weights :math:`w_i = \min\bigl(1, c / |r_i|\bigr)`.
3. Solve the weighted penalised system with weights :math:`\sqrt{w_i}`.
4. Repeat until convergence (default 15 iterations for fit, 10 for grid
   search).

This is equivalent to M-estimation with the Huber ψ-function and is
implemented by weighting rows of :math:`X` and :math:`y` by :math:`\sqrt{w}`
before calling the standard PIRLS solver.


Confidence and prediction intervals
-----------------------------------

The variance of the fitted mean at a new point :math:`x^*` is:

.. math::

   \text{Var}\bigl(\hat y(x^*)\bigr)
   = x^{*\top} \, \text{Cov}(\hat\beta) \, x^*.

The covariance is:

.. math::

   \text{Cov}(\hat\beta) = B_{\text{solve}} B_{\text{solve}}^\top \, \hat\sigma^2,
   \qquad
   \hat\sigma^2 = \frac{\text{RSS}}{n - \text{EDF}}.

**Confidence intervals** (for the mean response) use:

.. math::

   \hat y(x^*) \pm t_{\alpha/2, \, n-\text{EDF}} \,
   \sqrt{\text{Var}(\hat y(x^*))}.

**Prediction intervals** (for a new observation) add the observation noise:

.. math::

   \hat y(x^*) \pm t_{\alpha/2, \, n-\text{EDF}} \,
   \sqrt{\text{Var}(\hat y(x^*)) + \hat\sigma^2}.

The t-distribution is used instead of the normal because :math:`\sigma` is
estimated from data.  Degrees of freedom are :math:`n - \text{EDF}` because
the penalty has already "consumed" EDF degrees of freedom.
