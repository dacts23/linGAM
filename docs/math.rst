.. _math:

Mathematical Foundations
========================

This chapter explains the ideas that make GAMs work — not just the
equations, but *why* we chose each ingredient and what problem it solves.
linGAM ships two engines that solve the same problem with different
machinery, so the chapter is organised in four parts:

* :ref:`Part I<section-why-gam>` — the shared ideas both backends agree on:
  what a GAM is, and how its additive decomposition works.
* :ref:`Part II<section-pygam-design>` — the **pygam backend**: uniform
  B-splines, difference penalties, a discrete grid search over
  :math:`\lambda`, and the SVD solver.  Everything you need to read the
  code in ``lingam/pygam/``.
* :ref:`Part III<section-mgcv-design>` — the **mgcv backend**: the basis
  families of R's ``mgcv`` (thin plate, cubic, cyclic, P-splines...), the
  sum-to-zero identifiability constraint, and automatic smoothing-parameter
  selection by REML or GCV.  Everything you need to read ``lingam/mgcv/``.
* :ref:`Part IV<section-backends-compared>` — an explicit side-by-side
  comparison of the two, so you can pick the right engine for your problem.

If you're new to GAMs, start from the top.  If you already know penalised
B-splines, skip to :ref:`the SVD solver<section-svd-solver>` (pygam) or
:ref:`the mgcv bases<section-mgcv-bases>`, and finish with
:ref:`the comparison<section-backends-compared>`.


.. _part-shared:

Part I — Shared foundations
---------------------------

.. _section-why-gam:

Why GAMs?
~~~~~~~~~

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

Both linGAM backends fit exactly this model.  Where they differ is *how*:
how each :math:`f_j` is represented, how the smoothness penalty is built,
how the smoothing parameter is chosen, and how uncertainty is reported.
Part II and Part III cover those differences; the shared philosophy is the
equation above.


.. _section-formula-decomposition:

Decomposition: how the model explains its predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

One subtlety to remember later: the two backends *centre* these term plots
differently.  The mgcv backend enforces an explicit sum-to-zero constraint
on every smooth (so each panel is genuinely centred at zero), while the
pygam backend lets the intercept absorb each term's overall level.  We'll
make this precise in :ref:`Part III<section-mgcv-identifiability>`.


.. _part-pygam:

Part II — The PyGAM backend
---------------------------

.. _section-pygam-design:

How the pygam backend thinks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pygam backend (``backend='pygam'``) is the minimalist design: **one
basis family, one solver, one criterion, chosen well.**  Every smooth term
is a uniform cubic B-spline; smoothness is charged by a difference penalty;
the one free knob, the smoothing parameter :math:`\lambda`, is chosen by
evaluating a discrete grid of candidates and keeping the one with the best
Generalised Cross Validation (GCV) score.

The whole pipeline is:

.. math::

   \text{formula}
   \;\to\;
   \underbrace{B\text{-spline bases } X}_{\S 1}
   \;\to\;
   \underbrace{\text{penalty } P}_{\S 2}
   \;\to\;
   \underbrace{\text{solve } (X^\top X + P)\hat\beta = X^\top y}_{\S 3--4}
   \;\to\;
   \underbrace{\text{score with GCV, pick best } (K, \lambda)}_{\S 5--6}.

Each step is a section below.  The design bets on three ideas:

1. **Uniform B-splines are good enough.**  A cubic B-spline with
   10–20 basis functions can approximate almost any smooth engineering
   relationship; the exact knot placement rarely matters at that
   resolution.
2. **Grid search is honest.**  Instead of optimising :math:`\lambda`
   continuously (which requires derivatives of criteria that may have
   flat or multiple minima), evaluate a finite, user-visible grid and
   pick the winner.  Slow in theory, fast in practice once you precompute
   cleverly (:ref:`grid search<section-grid-search-mechanics>`).
3. **SVD is the safest solver.**  Never form :math:`X^\top X` explicitly;
   work with a square-root of the penalised system instead
   (:ref:`SVD solver<section-svd-solver>`).

Identifiability in this backend is *implicit*: the intercept column is
unpenalised and absorbs the overall level, and each penalty leaves its own
null space (constant and linear trends for the default penalty) uncharged,
so the decomposition into intercept + smooths is well defined in practice.
There is no explicit centering constraint — a deliberate simplification we
compare against mgcv's approach later.

.. _section-bsplines:

B-spline bases: building curves from building blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Then recursively blend for orders :math:`m = 2, \dots, k+1`:

.. math::

   B_{i,m}(z)
   = \frac{z - \tau_i}{\tau_{i+m-1} - \tau_i} \, B_{i,m-1}(z)
   + \frac{\tau_{i+m} - z}{\tau_{i+m} - \tau_{i+1}} \, B_{i+1,m-1}(z).

Each blend raises the order by one and shrinks the basis count by one, so
:math:`K` functions of order :math:`k+1` come out at the end.  The result
is an :math:`n \times K` design matrix :math:`B`.  Each row sums to 1 (the
bricks tile the space) and each column has compact support (localised
bumps).

**Why cubic?**  linGAM defaults to ``spline_order=3``, which the recursion
above turns into basis functions of order :math:`k+1 = 4` — cubic
polynomials, degree 3.  Cubic is the lowest degree that gives *continuous
second derivatives* — the curve looks smooth to the human eye.  Lower
orders look jagged; higher orders increase computation without visible
benefit.

**Linear extrapolation.**  Outside :math:`[0,1]`, B-splines can oscillate
or collapse to zero — not useful for prediction beyond the training range.
linGAM replaces the B-spline values with a linear extension using the
boundary derivative:

.. math::

   f(z) \approx f(0) + f'(0) \, z \quad (z < 0),
   \qquad
   f(z) \approx f(1) + f'(1) \, (z-1) \quad (z > 1).

The derivative :math:`f'(0)` is computed analytically from the order-:math:`k`
bases saved during the last step of the recursion, using the standard
B-spline derivative formula :math:`B'_{i,k+1} = k
\bigl( B_{i,k}/\Delta\tau - B_{i+1,k}/\Delta\tau \bigr)`.

**Periodic basis.**  When ``constraint='periodic'``, the domain wraps
around: the input is taken modulo 1 and the first :math:`k-1` columns fold
into the last :math:`k-1`, enforcing :math:`f(0) = f(1)` and reducing the
coefficient count from :math:`K` to :math:`K_{\text{eff}} = K - k + 1`.


.. _section-penalties:

Penalties: why flexibility needs a cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

*Ridge penalty:*  For factor terms (and linear terms, if penalised),
:math:`P = \lambda I_K`.  This is a "shrink-toward-zero" penalty — it pulls
all coefficients toward zero by the same amount.  (Linear ``l()`` terms
default to :math:`\lambda = 0`: unpenalised.)

**A honest caveat about "constraints".**  The ``constraint=`` argument maps
mono_inc/mono_dec to the first-difference penalty and convex/concave to
the second-difference penalty.  These are *soft* pushes, not hard
constraints: a heavily penalised first-difference fit is almost monotonic
but is never forced to be exactly monotonic.  For hard guarantees you would
need an inequality-constrained solver — a different mathematical problem
altogether (and one reason the mgcv backend does not offer these options
at all).

**The role of** :math:`\boldsymbol{\lambda}` **cannot be overstated.**
:math:`\lambda` determines how much information the model extracts from the
data vs. how much it imposes from the smoothness prior.  Too small →
overfitting (the model chases noise).  Too large → underfitting (the model
misses real patterns).  Grid search finds the Goldilocks value
automatically.


.. _section-svd-solver:

The SVD data-augmentation solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
when :math:`\lambda = 0`; in the code :math:`E` is the transposed
Cholesky factor of :math:`P + \varepsilon I`).  The top block says "fit
the actual data."  The bottom block says "keep the coefficients small in
the penalised directions."

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
   confidence intervals — no re-solving needed.  (:math:`U_1` is the top
   :math:`m \times m` block of :math:`U`, the part corresponding to the
   data block :math:`R`.)

The SVD approach is especially important when :math:`\lambda` is very small
(the near-unpenalised case) or very large (near-zero coefficients), where
the normal equations become ill-conditioned.  By working with the
square-root of :math:`X^\top X + P` rather than the matrix itself, the SVD
solver stays accurate across the entire range of :math:`\lambda`.


.. _section-edf:

Effective Degrees of Freedom (EDF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

where :math:`U_1` is the data block of the SVD's left singular vectors.
Why the equality holds: with :math:`X = QR` and
:math:`[R;E] = U\Sigma V^\top`, the hat matrix factors as
:math:`H = Q U_1 U_1^\top Q^\top`, so its trace is
:math:`\operatorname{tr}(U_1^\top Q^\top Q U_1) = \|U_1\|_F^2`.  This is
computed in a single NumPy call with no matrix inversion.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
* :math:`\gamma`: the GCV penalty multiplier (default 1.4 in the pygam
  backend).  Setting :math:`\gamma > 1` makes the criterion slightly
  conservative — it prefers simpler models, analogous to AIC's penalty term.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's trace what happens when you call ``model.fit(x, y)`` with the pygam
backend:

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

**A note on identifiability.**  The default second-difference penalty does
not charge constant or linear trends (they lie in :math:`D_2`'s null
space).  Together with the unpenalised intercept, this pins down the
decomposition: the intercept owns the level, and each smooth owns only its
*bending*.  The pygam backend relies on this implicit mechanism.  The mgcv
backend, as we'll see, makes the centering explicit with a sum-to-zero
constraint — same goal, different route.


.. _section-tensor:

Tensor-product interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
versa.  (Pass a list to ``lam=`` to give each dimension its own
:math:`\lambda_d`; a scalar is broadcast to all dimensions.)

**When to use a tensor vs. separate s() terms.**  If the interaction
exists, a model with only ``s(temperature) + s(vibration)`` will show
patterned residuals — the error at high-temp-high-vibration will be
systematically wrong.  Adding ``te(temperature, vibration)`` captures this
extra structure.  Check the decomposition plot: if the te() heatmap shows
a strong diagonal pattern, the interaction is real.


.. _section-intervals:

Confidence and prediction intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

This covariance is the **frequentist** one — it treats :math:`\lambda` as
fixed and propagates only the sampling variability of
:math:`\hat{\boldsymbol{\beta}}`.  The mgcv backend additionally computes
a *Bayesian* covariance (treating the penalty as a prior); the difference
is explained in :ref:`Part III<section-mgcv-inference>`.


.. _section-grid-search-mechanics:

Fast grid search: how it works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

All :math:`K`-configurations are processed in parallel when
``n_jobs != 1`` (``ThreadPoolExecutor``; NumPy releases the GIL during QR
decomposition, so CPU-bound linear algebra genuinely runs on multiple
cores).  With the default ``n_jobs=1`` everything runs serially.

**Phase 2 — Evaluate per** :math:`\boldsymbol{(K, \lambda)}` **(threaded).**
For each combination:

1. Assemble penalty: :math:`P = \lambda S`.
2. Form :math:`B = R_K^\top R_K + P`.
3. Cholesky solve for :math:`\hat{\boldsymbol{\beta}}` and EDF.
4. Compute RSS and GCV.

Cholesky costs :math:`O(m^3/3)` vs. :math:`O(m^3)` for SVD, and it's
numerically safe because :math:`R_K^\top R_K + P` is positive definite
(the penalty adds ridge stabilisation).  Both the K-precomputation and
the per-K evaluation loop use the same ``n_jobs``-gated threading.

**Why this matters.**  For a model with :math:`m = 20` coefficients,
:math:`n = 500` observations, and a grid of :math:`8 \times 8 = 64`
combinations: a naïve sweep takes ~64 SVDs of :math:`500 \times 20`
matrices.  The fast path does ~8 QR decompositions (one per unique
:math:`K`) and ~64 Cholesky solves (one per :math:`(K, \lambda)` pair),
which is roughly 10× faster.  The gap widens with larger :math:`n`.

Each smooth term (and each tensor dimension) owns its own
:math:`\lambda` slot in the grid, so the search optimises a vector of
smoothing parameters jointly.  A scalar ``lam=`` grid is broadcast to
every slot when you want the classic "one :math:`\lambda` for the whole
model" behaviour.


.. _section-robust:

Huber IRLS: handling outliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
implements it in ``irls_solve`` (used via ``fit(x, y, robust=True)``) by
scaling each row of :math:`X` and element of :math:`\mathbf{y}` by
:math:`\sqrt{w_i}` before calling the standard PIRLS solver, running up
to 15 such re-weighting iterations.  The reported RSS — and hence the
scale estimate and GCV — use the final weights,
:math:`\sum_i w_i r_i^2`, so outliers don't inflate the reported
uncertainty either.

**When to use robust fitting:**

- You suspect outliers (always a good default in engineering data).
- Your residuals show a few extreme values.
- Standard and robust fits give very different GCV scores -- the standard fit is being distorted by outliers.

**When NOT to use it:**

- Your data is clean and you need maximum speed (robust is ~5x slower due to iterations).
- Your "outliers" are actually the most important data points (e.g., rare failure events you want to model accurately).

This option is pygam-only; the mgcv backend currently fits by plain
penalised least squares.


.. _part-mgcv:

Part III — The mgcv backend
---------------------------

.. _section-mgcv-design:

How the mgcv backend thinks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mgcv backend (``backend='mgcv'``) is a faithful Python port of Simon
Wood's R package ``mgcv`` (Wood 2006, 2017).  Where the pygam backend bets
on one basis and a grid search, mgcv bets on **richness and automatic
inference**:

1. **A zoo of bases.**  Seven basis families (``'tp'``, ``'ts'``,
   ``'cr'``, ``'cs'``, ``'cc'``, ``'ps'``, ``'re'`` — plus ``'cp'``)
   with different knot placements, different penalties, and different
   shrinkage behaviour.  Each solves the smoothness problem a slightly
   different way.
2. **One smoothing parameter per penalty, estimated.**  Instead of
   sweeping a grid, the backend treats :math:`\lambda` as a free
   parameter and *optimises* a model-selection criterion — Restricted
   Maximum Likelihood (REML) by default, GCV as an option — in
   log-:math:`\lambda` space with standard optimisers.
3. **Explicit identifiability.**  Every smooth is centred by absorbing a
   sum-to-zero constraint into the basis itself, and every penalty is
   rescaled to a common scale (mgcv's ``S.scale``), so the :math:`\lambda`
   values are comparable across terms.
4. **Full inferential machinery.**  Bayesian and frequentist covariances,
   per-term approximate F-tests with p-values (Wood 2013), AIC/BIC,
   deviance explained.

The pipeline is:

.. math::

   \text{formula}
   \;\to\;
   \underbrace{\text{basis + penalty per term}}_{\S 1}
   \;\to\;
   \underbrace{\text{rescale, centre}}_{\S 2}
   \;\to\;
   \underbrace{\text{optimise } \lambda \text{ by REML/GCV}}_{\S 3}
   \;\to\;
   \underbrace{\text{inference}}_{\S 4}.

Sections 1–4 below follow that order.

.. _section-mgcv-bases:

The basis zoo
~~~~~~~~~~~~~

All of the mgcv bases share one objective: represent a smooth
:math:`f(x)` as a linear combination of basis functions with an attached
**wiggliness penalty** :math:`\boldsymbol{\beta}^\top S
\boldsymbol{\beta}`.  Three of the families (``tp``, ``cr``, ``cc``)
penalise the exact curvature integral :math:`\int f''(x)^2\,dx`; the
P-splines penalise a discrete difference approximation of it; the random
effect family penalises the coefficients directly.  Here's the map:

.. list-table::
   :header-rows: 1

   * - Code
     - Family
     - Basis functions
     - Penalises
     - Null space
   * - ``tp``
     - Thin plate regression spline
     - :math:`k` leading eigenfunctions of the curvature kernel
     - :math:`\int f''^2 dx`
     - constants + linear (dim 2)
   * - ``ts``
     - Thin plate, shrinkage
     - same as ``tp``
     - curvature + small charge on the null space
     - none (can shrink to zero)
   * - ``cr``
     - Cubic regression spline
     - natural cubics, knots at sample quantiles
     - :math:`\int f''^2 dx`
     - linear functions (dim 2)
   * - ``cs``
     - Cubic, shrinkage
     - same as ``cr``
     - curvature + small charge on the null space
     - none (can shrink to zero)
   * - ``cc``
     - Cyclic cubic
     - periodic natural cubics
     - :math:`\int f''^2 dx` on the circle
     - constants (dim 1)
   * - ``ps``
     - P-spline (Eilers & Marx)
     - cubic B-splines, equally spaced knots
     - :math:`m`-th order differences of coefficients
     - polynomials of degree :math:`< m`
   * - ``cp``
     - Cyclic P-spline
     - cubic B-splines on wrapped knots
     - wrapped :math:`m`-th order differences of coefficients
     - constants (dim 1)
   * - ``re``
     - Random effect
     - one column per factor level
     - :math:`\sum \beta_j^2` (ridge)
     - none (all levels shrunken)

The table's "null space" column matters more than it looks: those are the
function shapes that cost *nothing* to include.  A penalty with a null
space leaves the corresponding pattern unpenalised and lets the intercept
(or the model's other terms) absorb it.

.. _section-mgcv-tp:

Thin plate regression splines (``tp``, ``ts``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default in mgcv, and the most mathematically interesting family.  It
starts from a *continuous* problem, not a knot placement.

**The continuous problem.**  Among all functions with two derivatives,
find the :math:`f` minimising

.. math::

   \sum_{i=1}^n \bigl( y_i - f(x_i) \bigr)^2
   + \lambda \int f''(x)^2 \, dx.

The first term is ordinary least squares; the second charges the total
squared curvature.  It turns out (the *representer theorem* for splines)
that the minimiser has the form

.. math::

   f(x) = \sum_{i=1}^n \delta_i \, \eta(|x - x_i|) + \alpha_0 + \alpha_1 x,
   \qquad
   \eta(r) = \frac{|r|^3}{12},

subject to :math:`\sum_i \delta_i = \sum_i \delta_i x_i = 0` (the
:math:`T^\top\delta = 0` constraint).  The :math:`n` "radial" basis
functions sit one on each data point; the two polynomial terms form the
**null space** — they have zero curvature, so the penalty never touches
them.

**Why** :math:`\boldsymbol{\eta(r) = |r|^3/12}` **?**  It is the Green's
function of the fourth-derivative operator:

.. math::

   \frac{d^4}{dr^4}\left(\frac{|r|^3}{12}\right) = \delta(r),

where :math:`\delta` is the Dirac delta.  Consequence: for any function of
the form above, the curvature integral collapses to a quadratic form in
the coefficients,

.. math::

   \int f''(x)^2 \, dx \;=\; \boldsymbol{\delta}^\top E \boldsymbol{\delta},
   \qquad
   E_{ij} = \eta(|x_i - x_j|).

A one-line sanity check: for :math:`r > 0`,
:math:`\eta'''(r) = 1/2` (and :math:`-1/2` for :math:`r < 0`), so the
third derivative jumps by exactly 1 at the origin — which is what makes
the fourth derivative a unit Dirac spike.  The "thin plate" name comes
from the :math:`d=2` generalisation, where the penalty is the bending
energy of a physical plate.

**Conditional positive definiteness.**  The matrix :math:`E` has both
positive and negative eigenvalues, but it is *positive definite on the
subspace orthogonal to the null space* (:math:`T^\top\delta = 0`).
Solving the continuous problem exactly gives :math:`f` with one parameter
per data point (an :math:`O(n)` model) — beautiful, but too expensive and
over-flexible for practical use.

**The low-rank approximation.**  mgcv's trick (and linGAM's) is to keep
only the :math:`k` most important radial directions.  Eigen-decompose

.. math::

   E = U \operatorname{diag}(v) U^\top,

sort eigenpairs by :math:`|v_j|`, and truncate to the top :math:`k`:

.. math::

   E_k = U_k \operatorname{diag}(v_k) U_k^\top.

(The code uses ARPACK's ``eigsh`` for speed when there are many unique
points, and falls back to a full ``eigh`` otherwise.)  The retained
eigenfunctions are the smoothest, largest-scale modes — exactly the
structure you want in a smooth — while the discarded high-frequency modes
are the ones a penalty would crush anyway.

**Absorbing the constraint.**  We still need the retained directions to
be orthogonal to the null space, i.e. :math:`T^\top U_k \delta = 0`.
Compute a Householder rotation :math:`Q` such that

.. math::

   T^\top U_k \, Q = \bigl[\,0_{M \times (k-M)} \;\; R\,\bigr],
   \qquad M = 2.

Then the first :math:`k-M` columns of :math:`U_k Q` span exactly the
retained modes that are orthogonal to constants and linear terms.  The
basis and penalty become

.. math::

   X = \Bigl[\, U_k \operatorname{diag}(v_k) Q_{[:, :k-M]}
   \;\;\; T \,\Bigr],
   \qquad
   S = Q^\top \operatorname{diag}(v_k) Q
       \;\text{with rows/cols}\; k-M \dots k-1 \;\text{zeroed}.

The last :math:`M` columns of :math:`X` are :math:`[1, x-\bar x]`
(the null space, unpenalised); the first :math:`k-M` columns are the
penalised modes, each charged by its eigenvalue :math:`v_j`.

**Column rescaling.**  Finally each column of :math:`X` is scaled to unit
root-mean-square and :math:`S` is divided by the corresponding row/column
products (:math:`S_{ij} \leftarrow S_{ij} / (w_i w_j)`).  This keeps the
numerical scale of :math:`X` and :math:`S` comparable no matter what
units :math:`x` is measured in.

**Prediction at new points** follows the same construction without
re-truncating: :math:`X_{\text{new}} = \bigl[ E_{\text{new}} U_k Q_{[:, :k-M]},\;\;
[1, x_{\text{new}} - \bar x] \bigr]` (then divided by the same column
scales), where :math:`(E_{\text{new}})_{ij} = \eta(|x_i^{\text{new}} -
x_j^{\text{train}}|)`.

**Shrinkage (``ts``).**  The penalty :math:`S` above has a two-dimensional
null space (the :math:`T` part).  That means :math:`\lambda` can shrink
the wiggly part all it wants, but the linear trend in :math:`f` is never
penalised — a ``tp`` smooth can never be removed from the model.  The
``ts`` variant changes this by eigen-decomposing :math:`S` and giving the
null-space eigenvalues a small positive value:

.. math::

   \text{ev}_{0}, \text{ev}_{1} \leftarrow \text{ev}_2 \times 0.1,

which charges the linear trend just enough that a strong :math:`\lambda`
can shrink the whole smooth to zero.  (This is also how model selection
can effectively drop a term.)

.. _section-mgcv-cr:

Cubic regression splines (``cr``, ``cs``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If thin plate splines are the elegant continuous solution, cubic
regression splines are the workhorse discrete one.  The idea: place
:math:`k` knots at the sample **quantiles** of the unique :math:`x`
values, demand that :math:`f` be a cubic polynomial between knots, twice
continuously differentiable at each knot, and *linear outside the range of
the knots* (the "natural" boundary condition: :math:`f'' = 0` beyond the
first and last knot).

**The value–curvature parametrisation.**  A natural cubic spline is fully
determined by its values at the knots together with the natural boundary
condition.  Within an interval :math:`[x_j, x_{j+1}]` of width
:math:`h_j`, the spline interpolates between knot values
:math:`f_j, f_{j+1}` and second derivatives :math:`\gamma_j,
\gamma_{j+1}`:

.. math::

   f(x) = \frac{(x_{j+1}-x)^3 - h_j^2(x_{j+1}-x)}{6h_j}\,\gamma_j
        + \frac{(x-x_j)^3 - h_j^2(x-x_j)}{6h_j}\,\gamma_{j+1}
        + \frac{x_{j+1}-x}{h_j}\, f_j
        + \frac{x-x_j}{h_j}\, f_{j+1}.

(This is the formula the ``_eval_cr_basis`` code implements, with
:math:`\gamma = F\boldsymbol{\beta}` and the linear hat functions added
on top.)

**Two linear systems connect everything.**  For :math:`n_k` knots:

* The second derivatives at the knots are a linear function of the knot
  values, :math:`\boldsymbol{\gamma} = B^{-1} D \,\mathbf{f}`, where
  :math:`D` is the :math:`(n_k-2) \times n_k` divided-second-difference
  matrix

  .. math::

     D = \begin{pmatrix}
     \frac{1}{h_1} & -(\frac{1}{h_1}+\frac{1}{h_2}) & \frac{1}{h_2} & & \\
     & \frac{1}{h_2} & -(\frac{1}{h_2}+\frac{1}{h_3}) & \frac{1}{h_3} & \\
     & & \ddots & \ddots & \ddots
     \end{pmatrix},

  and :math:`B` is the symmetric tridiagonal matrix with diagonal
  :math:`(h_j + h_{j+1})/3` and off-diagonal :math:`h_{j+1}/6`.

* The curvature integral then factorises as

  .. math::

     \int f''(x)^2\,dx \;=\; \boldsymbol{\gamma}^\top B \boldsymbol{\gamma}
     \;=\; \mathbf{f}^\top D^\top B^{-1} D \,\mathbf{f}
     \;\Rightarrow\;
     S = D^\top B^{-1} D.

  Deriving it is a two-line calculus exercise: between knots :math:`f''`
  is a straight line, so its integral is exactly the quadratic form with
  the tridiagonal :math:`B`.

:math:`S` has rank :math:`k-2`: its null space is the two-dimensional
space of linear functions (straight lines have no curvature), which again
remain unpenalised.  Extrapolation beyond the boundary knots is linear by
construction — the natural boundary condition *is* linear extrapolation,
so ``cr`` behaves predictably outside the data range.

**Shrinkage (``cs``)** is the discrete twin of ``ts``: eigen-decompose
:math:`S`, and charge the two null-space eigenvalues at 0.1× and 0.01× the
smallest positive one:

.. math::

   \text{ev}_1 \leftarrow \text{ev}_2 \times 0.1, \qquad
   \text{ev}_0 \leftarrow \text{ev}_1 \times 0.1.

The linear trend is now weakly penalised, so a strong :math:`\lambda` can
shrink the whole term toward zero.

.. _section-mgcv-cc:

Cyclic cubic regression splines (``cc``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For quantities on a circle — time-of-day, wind direction, phase angles —
the boundary values should agree: :math:`f(x_0) = f(x_1)` and matching
derivatives.  The cyclic cubic spline (Lancaster & Salkauskas 1986) takes
the ``cr`` construction and *wraps* it: knots are placed evenly through
the sorted unique :math:`x` values, and the :math:`B` and :math:`D`
matrices become **circulant** — the last interval connects back to the
first, as in the code's ``build_cc_basis``:

.. math::

   B_{0,0} = \frac{h_{n-1} + h_0}{3}, \quad B_{0,1} = \frac{h_0}{6},
   \quad B_{0,n-1} = \frac{h_{n-1}}{6},

with the mirror entries on the last row, and similarly for :math:`D`.
The penalty is the same form, :math:`S = D^\top B^{-1} D`, now of rank
:math:`k-2` with a one-dimensional null space — constants only, since a
periodic function that is "straight" is exactly constant.  Evaluating the
basis wraps the covariate cyclically onto :math:`[x_0, x_1]` first.

.. _section-mgcv-ps:

P-splines (``ps``, ``cp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Eilers & Marx (1996) cut the problem even further: forget exact curvature
integrals; place a **uniformly spaced** grid of knots (extended 0.1%
beyond the data range, plus :math:`m_{\text{basis}}+1` knots past each
end so all :math:`k` basis functions are well defined), build cubic
B-splines on it, and penalise the *:math:`m`-th order differences* of
adjacent coefficients:

.. math::

   S = D^\top D, \qquad
   D = \underbrace{\Delta^m}_{\text{repeated differencing}},
   \qquad (k-m) \times k.

The default is :math:`m = 2`: second differences, the discrete
counterpart of curvature.  The penalty's null space is the space of
polynomials of degree :math:`< m` — for :math:`m=2`, linear trends are
free.  P-splines trade a little elegance for speed and simplicity (the
penalty needs no matrix inverse, unlike :math:`B^{-1}` in ``cr``), and
the equal knot spacing makes the basis numerically well behaved.

For cyclic P-splines (``'cp'``), both the B-spline basis (via mgcv's
``cSplineDes`` knot wrapping) and the :math:`m`-th order difference
penalty wrap around the boundaries, leaving only constants in the
one-dimensional null space.

.. _section-mgcv-re:

Random-effect smooths (``re``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a categorical predictor with many levels, ``re`` builds one column per
level (an identity/one-hot design matrix) and penalises the coefficients
with :math:`S = I` — pure ridge shrinkage:

.. math::

   \boldsymbol{\beta}^\top S \boldsymbol{\beta}
   = \sum_j \beta_j^2.

In the Bayesian view (see the REML section) this is exactly a
random-effects model: level effects drawn from :math:`N(0, \sigma^2 /
\lambda)`, and :math:`\lambda` estimated from the data is the
inverse-variance of the random effect.  Larger :math:`\lambda` shrinks
all level effects toward zero, pooling the levels together.

.. _section-mgcv-identifiability:

Identifiability: rescaling and the sum-to-zero constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A GAM with an intercept is over-parametrised: adding a constant to a
smooth :math:`f_j` and subtracting it from the intercept leaves every
prediction unchanged.  Fitting "works" regardless, but the decomposition
becomes arbitrary — you cannot say whether the average blade life came
from the intercept or from the temperature term.  mgcv resolves this with
two standard moves, both applied to *every* smooth term before fitting.

**Move 1 — ``S.scale``: rescale each penalty.**  A penalty's raw size
depends on its basis: a basis of large-magnitude functions produces a
large :math:`S`.  To make the smoothing parameter mean "amount of
smoothing" rather than "basis scale × amount of smoothing", each penalty
is normalised by

.. math::

   \text{S.scale}_i
   = \frac{\|S_i\|_{\text{O}}}{\|X\|_{\text{I}}^2},
   \qquad
   S_i \leftarrow \frac{S_i}{\text{S.scale}_i},

where :math:`\|\cdot\|_{\text{O}}` is the maximum column sum (the
operator norm of a symmetric matrix) and :math:`\|\cdot\|_{\text{I}}` is
the maximum row sum of :math:`X`, squared.  The scale is saved, so
reported :math:`\lambda` values are on this normalised scale — comparable
across terms and across basis families.

**Move 2 — ``absorb.cons``: absorb the sum-to-zero constraint.**  The
constraint is that each smooth's fitted values average to zero:

.. math::

   C \boldsymbol{\beta} = 0, \qquad C = \mathbf{1}^\top X
   \;\; \text{(the row of column sums)}.

Rather than solving a constrained problem, the constraint is *absorbed
into the basis*: compute the full QR of :math:`C^\top`
(:math:`k \times 1`), take :math:`Z = Q_{[:, 1:]}`
— an orthonormal basis of :math:`\ker C` — and project:

.. math::

   \tilde X = X Z, \qquad \tilde S_i = Z^\top S_i Z.

Now :math:`\mathbf{1}^\top \tilde X = 0` identically: the smooth is
centred *by construction*, in every fit, at every :math:`\lambda`, and in
every prediction.  The intercept owns the level; each smooth owns only
its shape.  This is why mgcv decomposition panels are genuinely centred at
zero, while the pygam panels rely on the intercept to absorb levels.

For tensor-product terms the same pipeline applies after assembling the
Kronecker-sum penalties; each marginal penalty is first normalised by its
largest eigenvalue (so the dimensions contribute to the penalty on equal
footing), then the usual ``S.scale`` + ``absorb.cons`` pass runs on the
full tensor basis.

.. _section-mgcv-reml:

Smoothing parameter selection: REML and GCV.Cp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both backends minimise the same penalised objective,

.. math::

   \| \mathbf{y} - X\boldsymbol{\beta} \|^2
   + \sum_i \lambda_i \boldsymbol{\beta}^\top S_i \boldsymbol{\beta}
   = \| \mathbf{y} - X\boldsymbol{\beta} \|^2
   + \boldsymbol{\beta}^\top S_{\boldsymbol{\lambda}} \boldsymbol{\beta},

but they choose :math:`\boldsymbol{\lambda}` differently.  The pygam
backend sweeps a grid and scores each point with GCV.  The mgcv backend
treats :math:`\boldsymbol{\lambda}` as continuous parameters of a
model-selection criterion and *optimises*.  For a fixed
:math:`\boldsymbol{\lambda}`, the fit is obtained by QR of :math:`X` and
a Cholesky solve of

.. math::

   H = R^\top R + S_{\boldsymbol{\lambda}} = L L^\top,
   \qquad
   \hat{\boldsymbol{\beta}} = H^{-1} X^\top \mathbf{y}.

(With the tiny ridge on :math:`H`'s diagonal guarding the
:math:`\lambda = 0` edge case.)

**GCV.Cp.**  The same criterion as the pygam backend, computed from the
Cholesky factor instead of an SVD.  The EDF is

.. math::

   \text{EDF} = \operatorname{tr}(H^{-1} R^\top R)
   = \| L^{-1} R^\top \|_F^2,

(a trace identity: :math:`\operatorname{tr}(ABA^\top) = \|AB^{1/2}\|_F^2`
with :math:`A = R`, :math:`B = H^{-1} = L^{-\top}L^{-1}`), and the score is

.. math::

   V_g(\boldsymbol{\lambda}) = \frac{n \cdot \text{RSS}}{(n - \gamma \cdot \text{EDF})^2},
   \qquad
   \hat\sigma^2 = \frac{\text{RSS}}{n - \text{EDF}}.

Here the default :math:`\gamma = 1` (mgcv's ``GCV.Cp``), versus 1.4 in
the pygam backend — a small but real difference between the two engines'
model-selection tastes.

**REML (the default).**  Give the coefficients a Gaussian prior whose
precision *is* the penalty,

.. math::

   \mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \quad
   \boldsymbol{\varepsilon} \sim N(0, \sigma^2 I), \qquad
   \boldsymbol{\beta} \sim N\bigl(0, \,\sigma^2 S_{\boldsymbol{\lambda}}^{-}\bigr),

where :math:`S_{\boldsymbol{\lambda}}^{-}` denotes the (generalised)
inverse — an *improper* prior that treats the penalty's null space as
infinite variance.  Integrating :math:`\boldsymbol{\beta}` out gives the
marginal (restricted) likelihood for :math:`\boldsymbol{\lambda}` and
:math:`\sigma^2`.  Profiling out :math:`\sigma^2` at

.. math::

   \hat\sigma^2 = \frac{D_p}{n - M_p}, \qquad
   D_p = \underbrace{\|\mathbf{y} - X\hat{\boldsymbol{\beta}}\|^2}_{\text{RSS}}
       + \underbrace{\hat{\boldsymbol{\beta}}^\top S_{\boldsymbol{\lambda}} \hat{\boldsymbol{\beta}}}_{\text{penalty cost}},

where :math:`M_p` is the dimension of the parametric part of the model
(the intercept, :math:`M_p = 1`), the criterion to minimise is

.. math::

   \ell_r(\boldsymbol{\lambda}) \;=\;
   \frac{D_p}{2\hat\sigma^2}
   + \frac{n - M_p}{2} \log(2\pi\hat\sigma^2)
   + \frac12 \log |H|
   - \frac12 \log |S_{\boldsymbol{\lambda}}|_+.

Reading it term by term:

* :math:`D_p / 2\hat\sigma^2`: how well the penalised fit explains the
  data, measured against the estimated noise.  The first two terms
  together are :math:`(n-M_p)(1 + \log 2\pi\hat\sigma^2)/2` after
  substituting :math:`\hat\sigma^2` — a pure function of the penalised
  deviance.
* :math:`\frac12 \log|H|`: a complexity tax.  The more the penalty is
  weakened, the larger :math:`H`'s determinant, the worse the score.
  This term is what prevents :math:`\lambda \to 0`.
* :math:`-\frac12 \log|S_{\boldsymbol{\lambda}}|_+`: a normalisation
  correction for the prior.  :math:`|S|_+` is the product of the
  *positive* eigenvalues only (the penalty is singular by design — its
  null space would otherwise contribute :math:`\log 0 = -\infty`).

Why the ":math:`_+`" matters can be seen by taking :math:`S_\lambda =
\lambda S` with :math:`\operatorname{rank}(S) = r`:
:math:`\log|S_\lambda|_+ = r\log\lambda + \log|S|_+`, which scales
correctly with the rank.  The implementation computes it by
eigendecomposition, summing logs of the strictly positive eigenvalues;
when the penalties of different terms act on *disjoint* blocks of
coefficients, it uses the factorisation

.. math::

   \log |S_{\boldsymbol{\lambda}}|_+ = \sum_i
   \Bigl[ \operatorname{rank}(S_i) \log \lambda_i
   + \sum_j \log \text{ev}^+_j(S_i) \Bigr],

and falls back to the general eigendecomposition otherwise (e.g. tensor
terms, whose per-dimension penalties share the same coefficient block).

:math:`\log|H|` needs no eigendecomposition: from the Cholesky factor,
:math:`\log|H| = 2\sum_i \log L_{ii}`.

**Why REML and not maximum likelihood?**  In the mixed-model view,
estimating :math:`\boldsymbol{\beta}` costs degrees of freedom; plain ML
ignores this and biases :math:`\sigma^2` (and thus :math:`\lambda`)
downward — it under-smoothes.  The restricted likelihood corrects for
exactly this loss, which is why the criterion uses :math:`n - M_p`
instead of :math:`n`.  REML also has better-behaved optima than GCV,
which is notorious for flat minima and occasional near-zero
:math:`\lambda` solutions.

**Optimising in log space.**  :math:`\lambda` must be positive, so the
backend optimises :math:`\rho = \log\lambda` (unconstrained, with
:math:`\lambda = e^\rho`) over the range :math:`\rho \in [-15, 15]` —
i.e. :math:`\lambda` between roughly :math:`3\times10^{-7}` and
:math:`3\times10^{6}`.  One free :math:`\lambda`: Brent's method
(minimise_scalar, bounded) on the 1-D criterion.  Several free
:math:`\lambda` (e.g. one per tensor dimension): L-BFGS-B starting from
:math:`\rho = 0` (:math:`\lambda = 1`).  Either way the fit is
re-evaluated a handful of times — each evaluation one Cholesky — rather
than at every point of a grid.  Individual :math:`\lambda` values can be
pinned with ``sp=`` if you want some terms fixed and others estimated.

The REML score itself is reported in ``statistics_`` along with the
estimated :math:`\lambda` values, so you can compare models on the
restricted likelihood exactly as R users do.

.. _section-mgcv-inference:

Inference: two covariances, one F-test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With :math:`\boldsymbol{\lambda}` estimated, the backend computes a full
inferential kit.  Two covariance matrices appear:

.. math::

   V_p = H^{-1}\hat\sigma^2, \qquad
   V_e = H^{-1} X^\top X H^{-1} \hat\sigma^2
       = H^{-1} R^\top R H^{-1} \hat\sigma^2.

* :math:`V_p` is the **Bayesian posterior covariance** — the posterior
  of :math:`\boldsymbol{\beta}` under the prior that generated REML, with
  :math:`\lambda` treated as known.  It is what mgcv reports in
  ``summary()`` tables and uses for confidence bands; standard errors are
  :math:`\operatorname{SE}_j = \sqrt{(V_p)_{jj}}`.
* :math:`V_e` is the **frequentist (sandwich) covariance** — the
  sampling covariance of :math:`\hat{\boldsymbol{\beta}}` treating the
  penalty as part of the estimator, again with :math:`\lambda` fixed.
  It ignores the uncertainty in :math:`\lambda` itself (a standard
  simplification in the mgcv world, though the literature notes it makes
  intervals slightly optimistic).

Interval widths for a new point :math:`\mathbf{x}^*` come from
:math:`\mathbf{x}^{*\top} V_p \,\mathbf{x}^*`, with
:math:`\hat\sigma^2` added for *prediction* intervals; the implementation
uses t quantiles with :math:`n - \text{EDF}` degrees of freedom.

**Per-coefficient and per-term EDF.**  From the influence matrix

.. math::

   F = H^{-1} R^\top R, \qquad
   \text{EDF} = \operatorname{tr}(F),
   \qquad
   \text{EDF}_j = F_{jj}.

Term-level EDF is the sum of :math:`F_{jj}` over that term's coefficient
block.  A term with EDF near 1 is essentially linear; a term whose EDF
barely exceeds 1 despite a large :math:`k` is doing very little.

**Wood's approximate F-tests.**  For each term, testing
:math:`H_0:\; f_j = 0` (in basis terms, :math:`\boldsymbol{\beta}_j =
0`), mgcv uses the test statistic (Wood 2013)

.. math::

   F_j = \frac{\hat{\boldsymbol{\beta}}_j^\top
              V_{p,j}^{-1} \hat{\boldsymbol{\beta}}_j}
              {\text{EDF}_j}
   \;\sim\; F(\text{EDF}_j,\; n - \text{EDF}_{\text{total}})
   \;\; \text{under } H_0,

with the p-value from the right tail of the F distribution.  The
justification is a moment-matching argument: under the null the
quadratic form has mean :math:`\text{EDF}_j` and variance
:math:`2\,\text{EDF}_j`, matching an :math:`F`-like distribution to
second order.  It is an *approximation* — exact for
:math:`\lambda = 0` (where it reduces to the classical F-test) and
well-calibrated for EDF comfortably above 1, but p-values for terms with
EDF barely above 1 should be read with suspicion (a known mgcv caveat).

**Model-level summaries.**  The backend also reports:

.. math::

   \text{deviance explained} = 1 - \frac{\text{RSS}}{\sum_i (y_i-\bar y)^2},
   \qquad
   R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n-\text{EDF})}
                                {\text{TSS}/(n-1)},

the Gaussian log-likelihood :math:`\ell = -\tfrac{n}{2}\log(2\pi\hat\sigma^2)
- \tfrac{\text{RSS}}{2\hat\sigma^2}`, and the information criteria
:math:`\text{AIC} = -2\ell + 2(\text{EDF}+1)`, :math:`\text{BIC} = -2\ell
+ \log(n)(\text{EDF}+1)` — the :math:`+1` counting the estimated
:math:`\sigma^2`.  These make the mgcv backend the choice when you need
p-values, AIC-based model comparison, or output that matches an R
``summary.gam`` table.


.. _part-comparison:

Part IV — Backends compared
---------------------------

.. _section-backends-compared:

PyGAM vs mgcv: the explicit comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both engines fit the same model to the same data with the same penalised
objective.  Everything else is a strategy choice.  The table summarises
the differences; the subsections after it explain the ones that matter.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Aspect
     - PyGAM backend
     - mgcv backend
   * - Basis families
     - One: uniform B-spline, cubic default, linear extrapolation, periodic folding
     - Seven+: ``tp``, ``ts``, ``cr``, ``cs``, ``cc``, ``ps``, ``re`` (+ ``cp`` stub), mixable per tensor margin
   * - Knot placement
     - Uniform over :math:`[\min x, \max x]`
     - Quantiles (``cr``/``cs``), even spread through unique values (``cc``), equally spaced (``ps``), knot-free eigen-basis (``tp``)
   * - Smoothing parameters
     - One :math:`\lambda` per term (per tensor dimension); grid can be broadcast to all
     - One :math:`\lambda` per penalty matrix; each estimated independently
   * - Selection method
     - Discrete grid search over :math:`(K, \lambda)`, scored by GCV with :math:`\gamma = 1.4`
     - Continuous optimisation of :math:`\log\lambda`; REML (default) or GCV.Cp with :math:`\gamma = 1`
   * - Solver
     - SVD data-augmentation of :math:`[R; E]` (never forms :math:`X^\top X`)
     - QR + Cholesky of :math:`R^\top R + S_\lambda`
   * - EDF computation
     - :math:`\|U_1\|_F^2` from the augmented SVD
     - :math:`\operatorname{tr}(H^{-1}R^\top R) = \|L^{-1}R^\top\|_F^2` from Cholesky
   * - Identifiability
     - Implicit: unpenalised intercept + penalty null spaces
     - Explicit: ``S.scale`` rescaling + absorbed sum-to-zero constraint
   * - Covariance
     - Frequentist :math:`B_{\text{solve}} B_{\text{solve}}^\top \hat\sigma^2`
     - Bayesian :math:`V_p = H^{-1}\hat\sigma^2` *and* frequentist :math:`V_e = H^{-1}R^\top R H^{-1}\hat\sigma^2`
   * - Intervals
     - t-based CI/PI, df :math:`= n - \text{EDF}`
     - t-based on :math:`V_p` (plus :math:`\hat\sigma^2` for PI), df :math:`= n - \text{EDF}`
   * - Term significance
     - None — compare GCV scores between models
     - Wood (2013) approximate F-tests with p-values
   * - Model metrics
     - GCV
     - REML/GCV score, AIC, BIC, deviance explained, adjusted :math:`R^2`
   * - Shape constraints
     - Soft: mono inc/dec, convex, concave via :math:`D_1`/:math:`D_2` penalties; periodic via wrapped basis
     - Periodic via ``cc`` (``cp`` pending); no monotonic/convex options
   * - Robust fitting
     - Huber IRLS, MAD-scaled
     - Not available (plain penalised least squares)
   * - Cost profile
     - Many cheap evaluations (grid); threaded with ``n_jobs != 1``; QR cached per :math:`K`
     - Few, heavier evaluations (optimiser iterations); one QR, repeated Cholesky
   * - Best fit for
     - Fast sweeps, shape constraints, outlier-heavy data, teaching the machinery
     - p-values & R parity, automatic :math:`\lambda`, basis choice, tensor margin mixing

**The same objective, two strategies.**  Both engines minimise
:math:`\|\mathbf{y} - X\boldsymbol{\beta}\|^2 +
\boldsymbol{\beta}^\top S_\lambda \boldsymbol{\beta}`.  The difference is
only *which* :math:`\lambda` you evaluate.  Grid search is deterministic,
embarrassingly parallel, and gives you the whole GCV surface to inspect —
but its resolution is limited to the grid, and cost grows multiplicatively
with the number of terms (each term adds a dimension to the grid).
Optimisation is adaptive (it spends evaluations near the minimum), handles
many :math:`\lambda` values gracefully, and lands on the continuous
optimum — but it converges to *local* minima, and REML's surface is
usually (though not always) nicer than GCV's.

**Numerics.**  The SVD solver is the more robust of the two across the
full :math:`\lambda` range — it never squares a condition number and it
degrades gracefully at :math:`\lambda \to 0`.  The Cholesky path is
faster per solve (:math:`m^3/3` vs :math:`\sim m^3`) and is protected by
the penalty itself, which keeps :math:`R^\top R + S_\lambda` positive
definite in practice.  Both are exact up to floating point; on
well-scaled data they agree to machine precision, and on ill-conditioned
data the SVD path is the safer bet.

**Inference.**  This is where the backends diverge most for the user.
The pygam backend reports GCV and (t-based, frequentist) intervals and
stops there — model selection is by comparing GCV between fits.  The mgcv
backend adds a Bayesian covariance, per-term p-values, AIC/BIC and an
R-style summary.  If you need "is this term significant?" or "which of
these models should I report?", the mgcv backend answers directly.

**Robustness and constraints.**  Conversely, the pygam backend owns two
practical features mgcv does not: Huber-robust fitting (outlier-tolerant
by design) and soft shape constraints (monotonic/convex/concave pushes).
If your data has sensor glitches or physics demands a monotone
dose–response, the pygam backend is the one with the knob.  Both handle
periodicity, differently: the pygam backend wraps its basis *and* uses a
circular difference penalty; the mgcv backend uses a purpose-built cyclic
cubic basis.

**Speed and scaling.**  With few terms and a small grid the pygam backend
is effectively instant; its cost grows with the product of grid sizes and
the number of :math:`\lambda` slots, which is why the two-phase
QR-caching matters.  The mgcv backend's cost is dominated by the
optimiser: typically 10–100 objective evaluations, each one Cholesky —
flat in the number of grid points, but sequential.  In practice: pygam
wins for sweeping many configurations; mgcv wins for a single
many-termed fit with automatically tuned :math:`\lambda` per term.

**The deep difference in one sentence.**  The pygam backend asks
*"which of these :math:`\lambda` candidates predicts best out of
sample?"* (GCV over a grid); the mgcv backend asks *"which
:math:`\lambda` makes the data most likely under a smoothness prior?"*
(REML over a continuous parameter).  Everything else — bases, solvers,
covariances — follows from that choice of question.

.. _section-backends-choose:

Which backend should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **You want p-values, an R-style summary, or parity with R mgcv** →
  ``backend='mgcv'`` with default REML.
* **You need automatic per-term smoothing with many terms** → mgcv's
  optimisation scales better than a per-term grid.
* **You want monotonic/convex/concave pushes or Huber robustness** →
  ``backend='pygam'``; these features don't exist in the mgcv backend.
* **You want to sweep basis counts and smoothing parameters and inspect
  the GCV surface** → pygam's grid search, threaded.
* **You want the simplest possible mental model** (one basis, one
  penalty, one criterion) → pygam.
* **You're teaching GAMs** → pygam for the machinery, then mgcv for the
  statistics; this chapter is laid out in that order.

When in doubt, fit both and compare: the decomposition plots and the
fitted curves should agree on the data's shape even where the p-values and
:math:`\lambda` philosophies differ.  Disagreement between the two engines
on the same data is a signal worth investigating — usually it means
under-smoothing by GCV, an outlier distorting the unrobust mgcv fit, or a
basis count too small for the grid being searched.


.. _section-summary:

Key takeaways
-------------

.. list-table::
   :header-rows: 1

   * - Concept
     - One-sentence summary
   * - Additive structure
     - Every prediction is a sum of independent, plottable pieces (both backends)
   * - B-spline basis (pygam)
     - Build any smooth curve from localised LEGO-brick basis functions
   * - Penalty (both)
     - Charge a tax for wiggles; let data pay it only when worthwhile
   * - :math:`\lambda` (both)
     - The tax rate: 0 = unpenalised, infinity = straight line
   * - EDF (both)
     - How many "effective parameters" the penalty allows the model to use
   * - GCV (pygam; mgcv option)
     - Approximate leave-one-out CV in a single fit; the pygam model-selection compass
   * - REML (mgcv default)
     - The restricted likelihood with the penalty as a prior; the mgcv model-selection compass
   * - SVD solver (pygam)
     - Numerically stable solution without squaring condition numbers
   * - QR + Cholesky (mgcv)
     - Compact, fast solves; :math:`\log|H|` and EDF fall out of the Cholesky factor
   * - Basis zoo (mgcv)
     - tp/ts, cr/cs, cc, ps, re — different penalties and null spaces for different jobs
   * - Sum-to-zero (mgcv)
     - Absorbed into the basis, so every smooth is centred by construction
   * - Wood F-tests (mgcv)
     - Approximate per-term significance tests with p-values
   * - Shape constraints (pygam)
     - Soft monotonic/convex/periodic pushes via the choice of penalty
   * - Huber IRLS (pygam)
     - Down-weight outliers so they don't hijack the curve
   * - Grid search (pygam)
     - Two-phase QR+Cholesky: precompute K-dim work, then evaluate all lambdas in parallel
   * - Comparison rule of thumb
     - mgcv for inference and automatic :math:`\lambda`; pygam for constraints, robustness, and fast sweeps
