.. _math:

Mathematical Foundations
========================

This chapter explains the ideas that make GAMs work — not just the
equations, but *why* we chose each ingredient and what problem it solves.

If you're new to GAMs, start from the top.  If you already know penalised
B-splines, skip to :ref:`the solver<section-svd-solver>` or
:ref:`GCV<section-gcv>`.


.. _section-why-gam:

Why GAMs?
---------

Classical linear regression fits a straight line (or hyperplane):

.. math::

   y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \varepsilon_i.

The problem: the real world is rarely a straight line.  Temperature does not
degrade turbine blades linearly.  Dose-response curves saturate.  An
engine's sweet-spot RPM is a peak, not a slope.

**Generalised Additive Models** replace each linear term :math:`\beta_j
x_{ij}` with a *learned smooth function* :math:`f_j(x_{ij})`:

.. math::

   y_i = \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + \dots + \varepsilon_i.

Now the model can bend to match the data.  The word **additive** means the
contributions still *sum* together — there's no multiplicative or nested
structure, just a collection of understandable pieces you can inspect one at
a time.  The word **generalised** means we replaced "linear coefficient ×
predictor" with "learned function of predictor."

This is the central tradeoff of statistical modelling:

  **Simple models** (lines) don't capture reality.  **Complex models**
  (deep nets, random forests) capture it but you can't explain *how*.
  GAMs sit in the sweet spot: flexible enough to fit nonlinear patterns,
  interpretable enough that each term is a plot you can show to a colleague.

When you call ``model.plot_decomposition(x, y)``, what you see is the
**decomposition** of the prediction into independent pieces:

.. math::

   \hat y(\mathbf{x}) =
   \underset{\text{intercept}}{\beta_0}
   + \underset{\text{term 0}}{f_1(x_1)}
   + \underset{\text{term 1}}{f_2(x_2)}
   + \dots
   + \underset{\text{noise}}{\hat\varepsilon}.

Each subplot in the decomposition is one :math:`f_j` in isolation.  Because
the model is additive, the pieces are independent: changing :math:`x_1`
only affects :math:`f_1(x_1)`, not :math:`f_2(x_2)`.  This is the property
that makes GAMs **interpretable** where black-box models are not.


.. _section-bsplines:

B-spline bases: building curves from building blocks
-----------------------------------------------------

If we want :math:`f(x)` to be a *smooth, flexible curve*, we need a
way to represent it mathematically.

**Why not just use a high-degree polynomial?**  A degree-20 polynomial can
wiggle a lot, but when you fit it to data every point in one corner of the
plot affects the curve *everywhere*.  You get wild oscillations (Runge's
phenomenon).  Polynomials are globally coupled — change one coefficient and
everything changes.

**The key insight of B-splines**: build the curve from *localised* building
blocks.  Each block (basis function) is nonzero only in a small region.
Adjusting a block only changes the curve in its neighbourhood, not across
the whole domain.  This is called **compact support**.

Think of LEGO bricks.  A long, curved wall isn't built from one flexible
piece — it's built from many small, rigid bricks each responsible for a
tiny segment.  The bricks overlap slightly at the edges for smoothness.
B-spline basis functions are the mathematical equivalent: each one is a
smooth "bump" that covers a few neighbouring data points.  When you stack
them with different heights (coefficients :math:`\beta_j`), you get any
smooth shape you want:

.. math::

   f(x) = \sum_{j=1}^K \beta_j \, B_j(x).

Each :math:`B_j(x)` is one basis function; :math:`\beta_j` is how tall
you make it; :math:`K` is how many bricks you use (``n_splines`` in the
API).

**The De Boor recursion.**  A B-spline of order :math:`k` (polynomial
degree :math:`k-1`) is built recursively.  Given data
:math:`x \in [x_{\min}, x_{\max}]`, we map to :math:`[0, 1]`:

.. math::

   z = \frac{x - x_{\min}}{x_{\max} - x_{\min}}.

Uniform interior knots divide :math:`[0, 1]` into segments:

.. math::

   \tau_0 = 0, \; \tau_1, \dots, \tau_{K-k+1} = 1.

The augmented knot sequence extends :math:`k` knots beyond each boundary
to handle the edges.

Start with order-1 (piecewise constant) bases:

.. math::

   B_{i,1}(z) = \mathbf{1}_{[\tau_i, \tau_{i+1})}(z).

Then recursively blend for orders :math:`m = 2, \dots, k`:

.. math::

   B_{i,m}(z)
   = \frac{z - \tau_i}{\tau_{i+m-1} - \tau_i} \, B_{i,m-1}(z)
   + \frac{\tau_{i+m} - z}{\tau_{i+m} - \tau_{i+1}} \, B_{i+1,m-1}(z).

The result is an :math:`n \times K` design matrix :math:`B`.  Each row sums
to 1 (the bricks tile the space) and each column has compact support
(localised bumps).

**Why cubic?**  linGAM defaults to cubic splines (:math:`k = 3`,
``spline_order=3``), corresponding to polynomial degree 2.  Cubic is the
lowest order that gives *continuous second derivatives* — the curve looks
smooth to the human eye.  Lower orders look jagged; higher orders increase
computation without visible benefit.

**Linear extrapolation.**  Outside :math:`[0,1]`, B-splines can oscillate
or collapse to zero — not useful for prediction beyond the training range.
linGAM replaces the B-spline values with a linear extension using the
boundary derivative:

.. math::

   f(z) \approx f(0) + f'(0) \, z \quad (z < 0),
   \qquad
   f(z) \approx f(1) + f'(1) \, (z-1) \quad (z > 1).

The derivative :math:`f'(0)` is computed analytically from the
:math:`(k-1)`-order basis saved during recursion.

**Periodic basis.**  When ``constraint='periodic'``, the domain wraps
around.  The first :math:`k-1` columns fold into the last :math:`k-1`,
enforcing :math:`f(0) = f(1)` and reducing the coefficient count from
:math:`K` to :math:`K_{\text{eff}} = K - k + 1`.


.. _section-penalties:

Penalties: why flexibility needs a cost
---------------------------------------

Suppose we give the model :math:`K = 30` basis functions.  It could fit the
data almost *perfectly* — including every random noise wiggle.  When we then
predict on new data, the model falls apart.  This is **overfitting**.

The solution is to penalise complexity.  We don't forbid wiggles; we charge
a tax for them.  The model can still wiggle, but only if doing so improves
the fit enough to be worth the tax.

**The penalised least-squares objective** is:

.. math::

   \underbrace{\|\mathbf{y} - X\boldsymbol{\beta}\|^2}_{\text{fit to data}}
   \;+\;
   \underbrace{\boldsymbol{\beta}^\top P \boldsymbol{\beta}}_{\text{complexity penalty}}.

The first term rewards fitting the training data closely.  The second term
punishes certain patterns in the coefficients.  The smoothing parameter
:math:`\lambda` (``lam`` in the API) controls the tax rate:

* :math:`\lambda = 0`: no penalty — the curve can be as wiggly as the
  basis allows.  Equivalent to unpenalised regression.
* :math:`\lambda \to \infty`: infinite penalty — the curve is forced to
  be as smooth as possible (a straight line for second-difference penalties).
* :math:`0 < \lambda < \infty`: a balance.  linGAM uses grid search to
  find the :math:`\lambda` that minimises GCV.

**What each penalty does to the curve:**

*Second-difference penalty (default):*  Penalises the *curvature* (second
derivative) of :math:`f(x)`.  A straight line has zero curvature, so it
costs nothing.  Wiggles cost a lot.  This produces the visually "smooth"
curves we expect.

.. math::

   D_2 = \begin{pmatrix}
   1 & -2 & 1 &        &   \\
     &  1 & -2 & 1     &   \\
     &    & \ddots & \ddots & \ddots
   \end{pmatrix}_{(K-2) \times K},
   \qquad
   P = \lambda \, D_2^\top D_2.

*First-difference penalty (monotonicity):*  Penalises *changes* between
adjacent coefficients.  If all adjacent coefficients are similar, the
curve can't reverse direction — hence the curve is pushed toward being
monotonic.

.. math::

   D_1 = \begin{pmatrix}
   -1 & 1 &   &   \\
      & -1 & 1 &   \\
      &    & \ddots & \ddots
   \end{pmatrix}_{(K-1) \times K},
   \qquad
   P = \lambda \, D_1^\top D_1.

*Circular second-difference (periodicity):*  The second-difference operator
wraps around, connecting the last coefficient back to the first — the
mathematical equivalent of gluing the two ends of the curve together.

*Ridge penalty:*  For factor and linear terms, :math:`P = \lambda I_K`.
This is a "shrink-toward-zero" penalty — it pulls all coefficients toward
zero by the same amount.

**The role of** :math:`\boldsymbol{\lambda}` **cannot be overstated.**
:math:`\lambda` determines how much information the model extracts from the
data vs. how much it imposes from the smoothness prior.  Too small →
overfitting (the model chases noise).  Too large → underfitting (the model
misses real patterns).  Grid search finds the Goldilocks value
automatically.


.. _section-svd-solver:

The SVD data-augmentation solver
--------------------------------

Once we have the design matrix :math:`X` and the penalty :math:`P`, we
need to find the coefficients :math:`\hat{\boldsymbol{\beta}}` that
minimise the penalised objective.

**The naive approach** forms the normal equations:

.. math::

   (X^\top X + P)\hat{\boldsymbol{\beta}} = X^\top \mathbf{y}.

This is mathematically correct but numerically dangerous.  Forming
:math:`X^\top X` squares the condition number — if :math:`X` has a
condition number of :math:`10^4`, then :math:`X^\top X` has :math:`10^8`.
On a computer with 16-digit precision, this leaves very little room for
error.

**The data-augmentation trick** treats the penalty as "fake data":

.. math::

   \begin{pmatrix} X \\ E \end{pmatrix} \boldsymbol{\beta}
   \approx \begin{pmatrix} \mathbf{y} \\ \mathbf{0} \end{pmatrix},

where :math:`E` satisfies :math:`E^\top E = P + \varepsilon I` (a tiny
ridge :math:`\varepsilon` ensures :math:`P` is positive definite even
when :math:`\lambda = 0`).  The top block says "fit the actual data."
The bottom block says "keep the coefficients small in the penalised
directions."

Solving this augmented system is equivalent to the penalised normal
equations, but **never forms** :math:`X^\top X`.  Instead, linGAM uses:

1. **QR decomposition** of :math:`X = QR`.  :math:`R` is small
   (:math:`m \times m`, where :math:`m = \sum K_j + 1`), so subsequent
   operations are cheap even when :math:`n` is huge.

2. **SVD of the stacked matrix** :math:`[R; E] = U \Sigma V^\top`.  The
   singular value decomposition is the most numerically stable way to
   factor a matrix — it exposes the "directions" in which the data has
   information and the directions it doesn't.

3. **Solution operator** :math:`B_{\text{solve}} = V \Sigma^{-1} U_1^\top
   Q^\top` so that :math:`\hat{\boldsymbol{\beta}} = B_{\text{solve}}
   \mathbf{y}`.  This is saved and reused for covariance, EDF, and
   confidence intervals — no re-solving needed.

The SVD approach is especially important when :math:`\lambda` is very small
(the near-unpenalised case) or very large (near-zero coefficients), where
the normal equations become ill-conditioned.  By working with the
square-root of :math:`X^\top X + P` rather than the matrix itself, the SVD
solver stays accurate across the entire range of :math:`\lambda`.


.. _section-edf:

Effective Degrees of Freedom (EDF)
----------------------------------

In ordinary linear regression with :math:`p` predictors, the model uses
exactly :math:`p` degrees of freedom — one per coefficient.  You can think
of each degree of freedom as the model's ability to "move" in one
independent direction to fit the data.

**Penalised splines blur this picture.**  The model *has* :math:`K`
coefficients (basis functions), but the penalty restricts how freely each
one can move.  A heavily penalised coefficient is "shrunken" toward its
neighbours — it costs you an entire degree of freedom to include, but the
penalty means you only use a fraction of it.

The EDF measures **how many independent directions the fitted model actually
explores.**  It is always between 0 and :math:`K`, and usually much closer
to the number of "real features" in the data.

Mathematically, the EDF is the trace of the influence (hat) matrix:

.. math::

   \text{EDF} = \operatorname{tr}(H),
   \qquad
   H = X (X^\top X + P)^{-1} X^\top.

Using the SVD augmentation from the solver, this simplifies to:

.. math::

   \text{EDF} = \sum_{i} \sigma_i(U_1)^2 = \|U_1\|_F^2,

where :math:`U_1` is a block of the SVD's left singular vectors.  This
is computed in a single NumPy call with no matrix inversion.

**What EDF tells you in practice:**

* ``EDF ≈ 2``: the term is essentially a straight line (2 df = intercept +
  slope).
* ``EDF ≈ K``: the penalty is very weak; the term is using nearly all its
  basis functions.
* ``EDF << K`` with good fit: the penalty is working well — the data has a
  smooth trend that the model captures with few effective parameters.
* ``EDF`` growing while GCV stays flat: you're overfitting — the extra
  wiggle doesn't help prediction.

Per-coefficient EDF tells you *which* basis functions are doing the work:

.. math::

   \text{EDF}_j = \sum_i U_{1,ij}^2.

A coefficient with EDF :math:`\approx 0` is essentially ignored by the
model — its basis function could be removed without consequence.


.. _section-gcv:

Generalised Cross Validation (GCV)
----------------------------------

**The problem.**  How do we know if our model is good?  We could hold out
some data, fit on the rest, and check how well it predicts the held-out
points.  But we'd have to do this many times (cross-validation), which is
slow — especially during grid search where we evaluate thousands of models.

**The clever solution.**  Leave-one-out cross-validation (LOOCV) fits the
model :math:`n` times, each time dropping one data point and predicting it.
This is :math:`O(n)` fits, prohibitive for anything but tiny datasets.

GCV is a closed-form approximation of LOOCV that requires **only one fit**.
The formula:

.. math::

   \text{GCV} = \frac{n \cdot \text{RSS}}
                     {(n - \gamma \cdot \text{EDF})^2}.

Let's unpack each piece:

* :math:`n`: number of observations — a larger dataset gives a more stable
  estimate.
* :math:`\text{RSS} = \sum_i (y_i - \hat y_i)^2`: how far the predictions
  are from the actual values.  Lower is better (better fit).
* :math:`\text{EDF}`: effective degrees of freedom — the model's complexity.
  More EDF means the model can "chase" data points more aggressively.
* :math:`\gamma`: the GCV penalty multiplier (default 1.4).  Setting
  :math:`\gamma > 1` makes the criterion slightly conservative — it prefers
  simpler models, analogous to AIC's penalty term.

The denominator :math:`(n - \gamma \cdot \text{EDF})^2` is the key.  When
EDF is small (simple model), the denominator is large → GCV is small →
good.  When EDF grows, the denominator shrinks → GCV explodes → bad,
*unless* the extra complexity reduces RSS enough to compensate.

**Why GCV and not plain CV?**  GCV is asymptotically equivalent to LOOCV
but computes in :math:`O(1)` extra time after a single fit.  It's the
reason linGAM's grid search can evaluate thousands of hyperparameter
combinations in seconds rather than minutes.

**Interpreting GCV values:**

* GCV is in squared units of :math:`y`.  Comparing GCV between models on
  the same dataset tells you which is better; comparing across datasets
  does not.
* A GCV that drops dramatically when you add a term means the term is
  useful.
* A GCV that stays flat or increases when you add a term means the term is
  unnecessary noise.
* The absolute value matters less than the *relative* value across models.


.. _section-penalised-fit:

Putting it together: the penalised fit
---------------------------------------

Let's trace what happens when you call ``model.fit(x, y)``:

1. **Parse the formula.**  ``"s(0) + te(1,2) + f(3)"`` becomes a list of
   term objects, each knowing its feature column(s), basis count, and
   penalty type.

2. **Build design matrices.**  Each term constructs its B-spline basis
   (or one-hot encoding for factors, or just a column for linear terms).
   These are stacked side-by-side, plus an intercept column of ones, to
   form the full design matrix :math:`X` (size :math:`n \times m`).

3. **Build the penalty.**  Each term's penalty matrix is computed and
   assembled into a block-diagonal :math:`P` (size :math:`m \times m`).
   The intercept block is zero — we never penalise the overall level.

4. **Solve.**  The SVD data-augmentation solver finds
   :math:`\hat{\boldsymbol{\beta}}` and stores the solution operator
   :math:`B_{\text{solve}}` for later use.

5. **Compute diagnostics.**  RSS, EDF, GCV, scale estimate, covariance
   matrix, standard errors — all available via ``model.statistics_``.

**Why the intercept is special.**  If we penalised the intercept, the
model would be forced toward :math:`y = 0`, which is almost never what we
want.  The intercept absorbs the overall level, letting the penalised
terms focus on shape.  This is why ``fit_intercept=True`` is the default
and why the intercept block in :math:`P` is zero.


.. _section-tensor:

Tensor-product interactions
---------------------------

An additive model with only univariate terms cannot capture relationships
where the effect of one predictor depends on the value of another.  For
example, high temperature might be especially damaging at high vibration —
this is a **synergistic interaction**.

**The tensor product** ``te(f1, f2, ..., fd)`` models this by building a
multi-dimensional smooth surface.  The design matrix is the column-wise
Kronecker product of marginal B-spline bases:

.. math::

   B_{\text{tensor}} = B_1 \odot B_2 \odot \dots \odot B_d.

If :math:`B_1` is :math:`n \times K_1` and :math:`B_2` is :math:`n \times
K_2`, then :math:`B_{\text{tensor}}` is :math:`n \times (K_1 K_2)`.  This
grows quickly — a 2-D tensor with :math:`K_1 = K_2 = 6` creates 36
columns; with :math:`K_1 = K_2 = 10` it's 100 columns.  This is why
tensor terms use smaller default :math:`K` values (4-8) than univariate
splines (5-25).

The penalty is a **Kronecker sum** — each dimension gets its own
smoothing parameter:

.. math::

   P_{\text{tensor}} = \sum_{d=1}^D \lambda_d \,
   \bigl( I_{K_1} \otimes \dots \otimes P_d \otimes \dots
   \otimes I_{K_D} \bigr).

This means smoothness is enforced independently in each direction.  A
surface can be smooth in temperature but wiggly in vibration, or vice
versa.

**When to use a tensor vs. separate s() terms.**  If the interaction
exists, a model with only ``s(temperature) + s(vibration)`` will show
patterned residuals — the error at high-temp-high-vibration will be
systematically wrong.  Adding ``te(temperature, vibration)`` captures this
extra structure.  Check the decomposition plot: if the te() heatmap shows
a strong diagonal pattern, the interaction is real.


.. _section-formula-decomposition:

Decomposition: how the model explains its predictions
------------------------------------------------------

The defining feature of GAMs is that predictions decompose additively:

.. math::

   \hat y(\mathbf{x}_i) =
   \underbrace{\hat\beta_0}_{\text{intercept}}
   + \underbrace{\hat f_1(x_{i1})}_{\text{term 1}}
   + \underbrace{\hat f_2(x_{i2})}_{\text{term 2}}
   + \dots
   + \underbrace{\hat f_p(x_{ip})}_{\text{term p}}.

Each :math:`\hat f_j` is a *partial dependence function* — it shows how the
prediction changes when you vary :math:`x_j` while holding everything else
fixed (usually at the median).  This is the "all else equal" effect.

**The plot_decomposition grid.**  When you call
``model.plot_decomposition(x, y)``, you see:

* **Top row — overall fit:** The model's prediction when sweeping the first
  predictor across its range, with all others at their medians.  Gray dots
  are the actual data.  This gives you a quick sanity check: does the curve
  pass through the cloud of points?

* **Per-term panels:** One subplot for each term in the formula:

  - ``s()`` panel: a line plot.  The curve shows how blade life changes with
    temperature (or RPM, or vibration).  A flat line means the term
    contributes nothing — the predictor doesn't matter.  A steep curve means
    the predictor is important.

  - ``te()`` panel: a 2-D heatmap (or projection line for 3+D).  Blue =
    below-average contribution, red = above-average.  Look for diagonal
    patterns — they indicate synergy.  A uniform color means no interaction.

  - ``f()`` panel: a bar chart.  Each bar is one category's baseline shift
    relative to the reference level.  Tall bars mean big differences between
    categories.

  - ``l()`` panel: a straight line.  The slope (beta coefficient) is shown
    in the title.  The line passes through zero at the median of the
    predictor.

* **Red shaded regions:** Extrapolation — the model is predicting outside
  the range of the training data.  Treat these regions with caution; the
  linear extrapolation used there is a heuristic, not a guarantee.

* **Zero reference lines:** Each term's contribution is centred so that the
  sum of all contributions (plus intercept) equals the prediction.  The
  dashed horizontal line at zero shows where the term contributes nothing.

**How to read the decomposition as a story.**  Start from the leftmost
term and work right.  Each panel answers one question:

1. "What does temperature do to blade life, all else equal?"
2. "What does RPM do, all else equal?"
3. "Do temperature and vibration interact synergistically?"
4. "Does coating type A last longer than B?"
5. "Is cooling flow linearly protective?"

The answers are *independent* — they add up to the full prediction, but you
can understand each one without reference to the others.  This is the
superpower of additive models: complex predictions built from simple,
understandable pieces.

**Partial dependence vs. raw data.**  The partial dependence curves do NOT
show the raw data distribution; they show the *model's isolated view* of
each predictor.  A scatter of raw data against a single predictor mixes in
the effects of all other predictors, so it can look very different from the
partial dependence.  The partial dependence is the "clean" signal — the
model's best estimate of what that predictor alone contributes.


.. _section-intervals:

Confidence and prediction intervals
-----------------------------------

A point prediction says "the model thinks blade life is 42.3 cycles."  An
interval says "the model thinks blade life is between 40.1 and 44.5 cycles,
with 95% confidence."  Intervals quantify uncertainty.

**Confidence intervals** answer: *How precisely do we know the average
response at this point?*  They capture uncertainty in the coefficient
estimates.  Narrow where we have lots of data; wide where data is sparse.

**Prediction intervals** answer: *Where would a single new observation
likely fall?*  They add the residual variability :math:`\hat\sigma^2` to
the confidence interval width.  A prediction interval is always wider than
a confidence interval at the same point — individual observations are
noisier than averages.

The formulas:

.. math::

   \text{Var}(\hat y(\mathbf{x}^*))
   = \mathbf{x}^{*\top} \, \text{Cov}(\hat{\boldsymbol{\beta}}) \,
     \mathbf{x}^*,

.. math::

   \text{Cov}(\hat{\boldsymbol{\beta}})
   = B_{\text{solve}} B_{\text{solve}}^\top \, \hat\sigma^2,
   \qquad
   \hat\sigma^2 = \frac{\text{RSS}}{n - \text{EDF}}.

Confidence intervals:

.. math::

   \hat y(\mathbf{x}^*) \pm t_{\alpha/2, \, n-\text{EDF}} \,
   \sqrt{\text{Var}(\hat y(\mathbf{x}^*))}.

Prediction intervals:

.. math::

   \hat y(\mathbf{x}^*) \pm t_{\alpha/2, \, n-\text{EDF}} \,
   \sqrt{\text{Var}(\hat y(\mathbf{x}^*)) + \hat\sigma^2}.

We use the t-distribution (not the normal) because :math:`\sigma` is
estimated from the data.  Degrees of freedom are :math:`n - \text{EDF}`
because the penalty has already "used up" EDF worth of information.


.. _section-grid-search-mechanics:

Fast grid search: how it works
------------------------------

Grid search evaluates every combination of :math:`K` (``n_splines``) and
:math:`\lambda` (``lam``) candidates.  A naïve implementation would rebuild
:math:`X` and solve from scratch for each combination — prohibitively slow.

linGAM's fast path splits the work into two phases:

**Phase 1 — Precompute per** :math:`\boldsymbol{K}` **(threaded).**
For each candidate basis count :math:`K`:

1. Build :math:`X_K` and compute :math:`X_K = Q_K R_K` (QR).
2. Cache :math:`R_K^\top R_K` and :math:`R_K^\top Q_K^\top \mathbf{y}`.
   These are *independent of* :math:`\lambda`.
3. Precompute the unscaled penalty basis :math:`S = D^\top D` at
   :math:`\lambda = 1`.

All :math:`K`-configurations are processed in parallel via
``ThreadPoolExecutor``.  NumPy releases the GIL during QR decomposition,
so CPU-bound linear algebra genuinely runs on multiple cores.

**Phase 2 — Evaluate per** :math:`\boldsymbol{(K, \lambda)}` **(threaded).**
For each combination:

1. Assemble penalty: :math:`P = \lambda S`.
2. Form :math:`B = R_K^\top R_K + P`.
3. Cholesky solve for :math:`\hat{\boldsymbol{\beta}}` and EDF.
4. Compute RSS and GCV.

Cholesky costs :math:`O(m^3/3)` vs. :math:`O(m^3)` for SVD, and it's
numerically safe because :math:`R_K^\top R_K + P` is positive definite
(the penalty adds ridge stabilisation).  Both phases are fully parallel.

**Why this matters.**  For a model with :math:`m = 20` coefficients,
:math:`n = 500` observations, and a grid of :math:`8 \times 8 = 64`
combinations: a naïve sweep takes ~64 SVDs of :math:`500 \times 20`
matrices.  The fast path does ~8 QR decompositions (one per unique
:math:`K`) and ~64 Cholesky solves (one per :math:`(K, \lambda)` pair),
which is roughly 10× faster.  The gap widens with larger :math:`n`.


.. _section-robust:

Huber IRLS: handling outliers
-----------------------------

Real data has outliers — sensor glitches, data-entry errors, anomalous
operating conditions.  Ordinary least squares weights every point equally,
so a single extreme outlier can pull the entire curve toward it.

**The Huber loss** replaces squared error with a hybrid: small residuals
are penalised quadratically (like OLS), large residuals are penalised
linearly (less harshly).  The transition point :math:`c` is set
automatically from the data's median absolute deviation (MAD):

.. math::

   \rho(r) = \begin{cases}
   \tfrac12 r^2            & |r| \le c \\
   c(|r| - \tfrac12 c)     & |r| > c
   \end{cases},
   \qquad
   c = 1.345 \cdot \text{MAD} / 0.6745.

The constant 1.345 gives ~95% asymptotic efficiency under Gaussian errors
— when there are no outliers, the robust fit is almost as good as OLS.
When there are outliers, it's vastly better.

**IRLS iterates:**

1. Fit the model with current weights.
2. Compute residuals :math:`r_i = y_i - \hat y_i`.
3. Update weights: :math:`w_i = \min(1, \, c / |r_i|)`.  Points with
   large residuals get small weights — they're effectively ignored.
4. Re-fit with the new weights.  Repeat until convergence.

This is equivalent to M-estimation with the Huber ψ-function.  linGAM
implements it by scaling each row of :math:`X` and element of
:math:`\mathbf{y}` by :math:`\sqrt{w_i}` before calling the standard
PIRLS solver — no separate code path needed.

**When to use robust fitting:**

- You suspect outliers (always a good default in engineering data).
- Your residuals show a few extreme values.
- Standard and robust fits give very different GCV scores -- the standard fit is being distorted by outliers.

**When NOT to use it:**

- Your data is clean and you need maximum speed (robust is ~5x slower due to iterations).
- Your "outliers" are actually the most important data points (e.g., rare failure events you want to model accurately).


.. _section-summary:

Key takeaways
-------------

.. list-table::
   :header-rows: 1

   * - Concept
     - One-sentence summary
   * - B-spline basis
     - Build any smooth curve from localised LEGO-brick basis functions
   * - Penalty
     - Charge a tax for wiggles; let data pay it only when worthwhile
   * - :math:`\lambda`
     - The tax rate: 0 = unpenalised, infinity = straight line
   * - EDF
     - How many "effective parameters" the penalty allows the model to use
   * - GCV
     - Approximate leave-one-out CV in a single fit; your model-selection compass
   * - SVD solver
     - Numerically stable solution without squaring condition numbers
   * - Decomposition
     - Each term is an independent, interpretable piece of the prediction
   * - Grid search
     - Two-phase QR+Cholesky: precompute K-dim work, then evaluate all lambdas in parallel
   * - Robust fitting
     - Down-weight outliers so they don't hijack the curve
