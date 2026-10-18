.. _part-shared:

Part I: Shared Foundations
===========================

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
contributions still *sum* together; there is no multiplicative or nested
structure, just a collection of understandable pieces you can inspect one at
a time.  The word **generalised** means we replaced "linear coefficient ×
predictor" with "learned function of predictor."

This is the central tradeoff of statistical modelling:

  **Simple models** (lines) don't capture reality.  **Complex models**
  (deep nets, random forests) capture it but you can't explain *how*.
  GAMs sit in the sweet spot: flexible enough to fit nonlinear patterns,
  interpretable enough that each term is a plot you can show to a colleague.

.. figure:: ../_static/images/why_gam_sweet_spot.png
   :alt: Underfitting linear regression vs overfitting unpenalized spline vs penalized GAM
   :align: center
   :width: 100%

   **(a)** Ordinary linear regression underfits nonlinear curves. **(b)** An unpenalised
   high-capacity spline chases noise (:math:`\text{EDF} = 28`). **(c)** A penalised GAM
   recovers the underlying curve with calibrated 95% confidence and prediction intervals.

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

Both magli backends fit exactly this model.  Where they differ is *how*:
how each :math:`f_j` is represented, how the smoothness penalty is built,
how the smoothing parameter is chosen, and how uncertainty is reported.
Part II and Part III cover those differences; the shared philosophy is the
equation above.


.. _section-formula-decomposition:

Decomposition: how the model explains its predictions
-----------------------------------------------------

The defining feature of GAMs is that predictions decompose additively:

.. math::

   \hat y(\mathbf{x}_i) =
   \underbrace{\hat\beta_0}_{\text{intercept}}
   + \underbrace{\hat f_1(x_{i1})}_{\text{term 1}}
   + \underbrace{\hat f_2(x_{i2})}_{\text{term 2}}
   + \dots
   + \underbrace{\hat f_p(x_{ip})}_{\text{term p}}.

Each :math:`\hat f_j` is a *partial dependence function*, showing how the
prediction changes when you vary :math:`x_j` while holding everything else
fixed (usually at the median).  This is the "all else equal" effect.

.. figure:: ../_static/images/gamcore_decomposition.png
   :alt: GAMCore additive decomposition across s(), te(), f(), and l() terms
   :align: center
   :width: 100%

   Example ``GAMCore.plot_decomposition`` grid for ``s(0) + te(0, 1) + f(2) + l(3)``:
   overall fit at the top, followed by each term's isolated partial dependence.

**The plot_decomposition grid.**  When you call
``model.plot_decomposition(x, y)``, you see:

* **Top row (overall fit):** The model's prediction when sweeping the first
  predictor across its range, with all others at their medians.  Gray dots
  are the actual data.  This gives you a quick sanity check: does the curve
  pass through the cloud of points?

* **Per-term panels:** One subplot for each term in the formula:

  - ``s()`` panel: a line plot.  The curve shows how blade life changes with
    temperature (or RPM, or vibration).  A flat line means the term
    contributes nothing, meaning the predictor doesn't matter.  A steep curve means
    the predictor is important.

  - ``te()`` panel: a 2-D heatmap (or projection line for 3+D).  Blue =
    below-average contribution, red = above-average.  Look for diagonal
    patterns, as they indicate synergy.  A uniform color means no interaction.

  - ``f()`` panel: a bar chart.  Each bar is one category's baseline shift
    relative to the reference level.  Tall bars mean big differences between
    categories.

  - ``l()`` panel: a straight line :math:`\hat\beta_j x_j`.  The slope
    (:math:`\hat\beta_j` coefficient) is shown in the title, and the line
    passes through zero at :math:`x_j = 0`.

* **Red shaded regions (Extrapolation):** the model is predicting outside
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

The answers are *independent*: they add up to the full prediction, but you
can understand each one without reference to the others.  This is the
superpower of additive models: complex predictions built from simple,
understandable pieces.

**Partial dependence vs. raw data.**  The partial dependence curves do NOT
show the raw data distribution; they show the *model's isolated view* of
each predictor.  A scatter of raw data against a single predictor mixes in
the effects of all other predictors, so it can look very different from the
partial dependence.  The partial dependence is the "clean" signal, while the
model's best estimate of what that predictor alone contributes.

One subtlety to remember later: the two backends *centre* these term plots
differently.  The Wood backend enforces an explicit sum-to-zero constraint
on every smooth (so each panel is genuinely centred at zero), while the
Eilers-Marx backend lets the intercept absorb each term's overall level.  We'll
make this precise in :ref:`Part III<section-wood-identifiability>`.


.. _section-core-intuitions:

Core Intuitions & Mental Models
-------------------------------

Before diving into the matrix algebra of the two backends, having the right physical
and geometric mental models makes GAMs intuitive.

The Elastic Spring Analogy for Penalties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A standard polynomial or unpenalised spline treats each coefficient as a free, independent
lever.  If noise moves one data point, the unpenalised model will twist the curve wildly
to chase it.

In a GAM, think of the spline coefficients :math:`\beta_1, \beta_2, \dots, \beta_K`
as heights of pegs, connected to their neighbours by **elastic springs**:

* **Second-difference penalty (:math:`D_2`):** The springs penalise sharp changes in slope
  (curvature).  If one peg is pulled out of line with its two neighbours, the spring
  resists with tension proportional to :math:`\lambda`.
* **Zero penalty (:math:`\lambda \to 0`):** The springs are completely slack.  Every peg moves
  freely, overfitting local noise.
* **Infinite penalty (:math:`\lambda \to \infty`):** The springs become rigid steel rods.  The
  pegs are forced into a perfectly straight line, collapsing the spline to its linear null space.
* **The sweet spot (:math:`0 < \lambda < \infty`):** The springs allow the curve to bend where
  multiple data points agree on a genuine signal, while damping out high-frequency noise.

Demystifying Effective Degrees of Freedom (EDF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ordinary regression, degrees of freedom are integers: an unpenalised model with 10 basis
functions has exactly 10 parameters (10 degrees of freedom).

In a GAM, you will routinely see outputs like:

.. code-block:: text

   s(0):  EDF = 3.42,  k = 10

What does **3.42 degrees of freedom** mean?

Because the penalty matrix :math:`S_\lambda` ties the coefficients together, they cannot
move independently.  The mathematical definition is the trace of the influence (hat) matrix:

.. math::

   \text{EDF} = \operatorname{tr}\bigl( (X^\top X + S_\lambda)^{-1} X^\top X \bigr).

Intuitively:

* An EDF of **1.0** means the term has collapsed to a simple straight line (1 linear slope parameter).
* An EDF of **2.0** represents a pure quadratic curve (parabola).
* An EDF of **3.42** means the curve has the flexibility of approximately a cubic polynomial,
  but with the localized smoothness and stability of a spline.
* An EDF close to the basis dimension (:math:`\text{EDF} \approx k - 1`) is a warning sign that
  the basis has **saturated**: the true underlying function may be wigglier than :math:`k` allows,
  and you should increase ``k``.

Confidence Intervals vs. Prediction Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

magli models provide two distinct uncertainty bands via ``model.confidence_intervals(x)``
and ``model.prediction_intervals(x)``:

1. **Confidence Interval (Credible Band on :math:`f(x)`):**
   Reflects the uncertainty in estimating the *true underlying mean curve* :math:`f(x)`.
   If you gathered infinite data points, this band would shrink to zero width.
   Mathematically, it depends only on the parameter covariance:
   :math:`\operatorname{Var}(\hat{f}(x)) = \mathbf{b}(x)^\top V_{\boldsymbol{\beta}} \mathbf{b}(x)`.

2. **Prediction Interval (Spread of New Data Points :math:`y_{\text{new}}`):**
   Answers: *"Where will a future individual measurement fall?"*
   Even if the mean curve :math:`f(x)` were known with 100% certainty, individual observations
   still vary with residual noise :math:`\sigma^2`.
   The prediction variance includes this observation noise:
   :math:`\operatorname{Var}(y_{\text{new}} \mid x) = \operatorname{Var}(\hat{f}(x)) + \hat\sigma^2`.
   Consequently, prediction intervals never collapse to zero width, regardless of sample size.

Multidimensional Geometry: How Tensor Products Work
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When two variables interact (e.g., turbine blade temperature :math:`x_1` and vibration amplitude :math:`x_2`),
an additive model :math:`f_1(x_1) + f_2(x_2)` cannot capture the synergy: high temperature
might be benign at low vibration, but catastrophic at high vibration.

``te(0, 1)`` solves this by taking the **tensor product** of the marginal bases:

.. math::

   B_{i, j}(x_1, x_2) = B_i(x_1) \times B_j(x_2).

.. figure:: ../_static/images/tensor_product_interaction.png
   :alt: Tensor-product splines from 1-D bases to 2-D interaction surfaces
   :align: center
   :width: 100%

   Constructing a 2-D interaction surface with ``te()``: **(1)** Marginal 1-D basis functions
   along :math:`x_1` and :math:`x_2`; **(2)** A single 2-D tensor basis bump :math:`B_{3,3}(x_1, x_2)`
   localized at the intersection of both intervals; **(3)** Fitted 2-D surface :math:`\hat{f}(x_1, x_2)`
   capturing non-linear interaction with data points.

Crucially, tensor products use **Kronecker sum penalties**:

.. math::

   S_{\text{total}} = \lambda_1 \, (S_1 \otimes I) + \lambda_2 \, (I \otimes S_2).

This allows the model to have different amounts of smoothness along each coordinate axis:
if :math:`x_1` varies rapidly while :math:`x_2` changes slowly, REML estimates
:math:`\lambda_1 \ne \lambda_2` automatically.


.. _section-diagnostic-guide:

A Practitioner's Diagnostic & Troubleshooting Guide
---------------------------------------------------

Once you fit a GAM, how do you verify whether the model is trustworthy?

1. How to Read `model.summary()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Wood backend summary provides:

* **Parametric coefficients:** Linear effects and intercept, reported with standard errors,
  t-statistics, and p-values matching standard regression.

* **Smooth terms table:**

  * **EDF:** Effective degrees of freedom. Higher EDF = wigglier curve.
  * **Ref.df:** Reference degrees of freedom used in the denominator of the F-test.
  * **F-statistic & p-value (Wood 2013):** Tests the null hypothesis :math:`H_0: f(x) = 0`
    (i.e. the predictor has *zero effect* across its entire range). A small p-value
    (:math:`p < 0.05`) indicates significant non-linear evidence.

* **Model-level metrics:**

  * **R-sq.(adj):** Adjusted :math:`R^2` accounting for EDF.
  * **Deviance explained:** Percentage of variation captured by the model relative to the null (mean-only) model.
  * **REML / GCV score:** Model-selection criterion (lower is better).

2. Checking for Basis Saturation (Is k large enough?)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The basis dimension :math:`k` sets the **maximum possible flexibility** of the term.
The penalty shrinks that flexibility down to the optimal EDF.

* **Rule of thumb:** If the estimated EDF is substantially lower than :math:`k`
  (e.g., :math:`\text{EDF} = 3.2` with :math:`k = 10`), :math:`k` is sufficient.
* **Saturation warning:** If :math:`\text{EDF} \approx k - 1` (or :math:`k - 2`),
  the penalty wanted more freedom than the basis could supply.
  **Action:** Re-fit with a larger basis (e.g. ``k=20`` or ``k=30``).

3. Boundary Flares and Data Sparsity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you notice the 95% confidence bands flare out widely at the left or right edges of a
decomposition plot, check your predictor distribution:

* Splines have fewer basis functions overlapping near boundaries.
* If data is sparse near extremes, there are fewer observations to constrain the curve,
  causing parameter variance :math:`V_{\boldsymbol{\beta}}` to inflate.
* **Action:** Avoid interpreting or extrapolating in regions where confidence bands flare widely.

4. Four-Panel Diagnostic Checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calling ``model.plot_diagnostics(x, y)`` renders four essential diagnostic checks:

1. **Observed vs. Fitted:** Points should cluster tightly along the identity line :math:`y = \hat{y}`.
2. **Residuals vs. Fitted:** Points should scatter symmetrically around zero with no curvature.
   Any curve or bow indicates an unmodeled non-linear pattern or missing interaction.
3. **Residual Histogram:** Should be roughly bell-shaped, centered at zero.
4. **Normal Q-Q Plot:** Residual quantiles should follow the diagonal red reference line.
   Departures in the tails indicate heavy-tailed errors or outliers (for which observation
   weights ``weights=...`` can be used).



