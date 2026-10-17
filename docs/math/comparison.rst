.. _part-comparison:

Part IV — Backends compared
===========================

.. _section-backends-compared:

Eilers-Marx vs Wood: the explicit comparison
--------------------------------------------

Both engines fit the same model to the same data with the same penalised
objective.  Everything else is a strategy choice.  The table summarises
the differences; the subsections after it explain the ones that matter.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Aspect
     - Eilers-Marx backend
     - Wood backend
   * - Basis families
     - One: uniform B-spline, cubic default, linear extrapolation, periodic folding
     - Nine: ``tp``, ``ts``, ``cr``, ``cs``, ``cc``, ``ps``, ``cp``, ``bs``, ``re``, mixable per tensor margin
   * - Knot placement
     - Uniform over :math:`[\min x, \max x]`
     - Quantiles (``cr``/``cs``), even spread through unique values (``cc``), equally spaced (``ps``/``cp``/``bs``), knot-free eigen-basis (``tp``/``ts``)
   * - Smoothing parameters
     - One :math:`\lambda` per term (per tensor dimension); grid can be broadcast to all
     - One :math:`\lambda` per penalty matrix; each estimated independently
   * - Selection method
     - Discrete grid search over :math:`(K, \lambda)`, scored by GCV with :math:`\gamma = 1.4`
     - Continuous optimisation of :math:`\log\lambda` via REML (default) or GCV.Cp (:math:`\gamma = 1`), plus optional ``gridsearch()`` over :math:`k`
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
     - GCV, AIC, BIC, log-likelihood, deviance explained
     - REML/GCV score, AIC, BIC, log-likelihood, deviance explained, adjusted :math:`R^2`
   * - Shape constraints
     - Soft: ``mono_inc``/``mono_dec`` (:math:`D_1` penalty), ``convex``/``concave`` (:math:`D_2` penalty), ``periodic`` (wrapped basis + circular :math:`D_2`)
     - Periodic via ``cc`` and ``cp`` (or ``constraint='periodic'``); no monotonic/convex options
   * - Observation weights
     - Supported in ``fit()`` and ``gridsearch()`` via :math:`\sqrt{w_i}` scaling
     - Supported in ``fit()`` and ``gridsearch()`` via :math:`\sqrt{w_i}` scaling
   * - Cost profile
     - Many cheap evaluations (grid); threaded with ``n_jobs != 1``; QR cached per :math:`K`
     - Few, heavier evaluations (optimiser iterations with analytic gradients); one QR, repeated Cholesky
   * - Best fit for
     - Fast sweeps, shape constraints, teaching the machinery
     - p-values & rigorous summary, automatic :math:`\lambda`, basis choice, tensor margin mixing

.. figure:: ../_static/images/eilers_vs_wood_comparison.png
   :alt: Six-panel side-by-side comparison of Eilers-Marx and Wood backends across basis functions, smoothing criteria, shape constraints, observation weights, and 2-D tensor interactions
   :align: center
   :width: 98%

   Head-to-head comparison of the Eilers-Marx and Wood backends across six
   scenarios (from ``examples/13_eilers_vs_wood_comparison.py``): (1) 1-D
   basis comparison, (2) discrete GCV grid search vs. continuous REML/GCV.Cp,
   (3) monotonic shape constraint, (4) cyclic periodic smooths, (5) observation
   weights on outlier-contaminated data, and (6) 2-D tensor-product
   decomposition with per-axis smoothing.

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
The Eilers-Marx backend reports GCV and (t-based, frequentist) intervals and
stops there — model selection is by comparing GCV between fits.  The Wood
backend adds a Bayesian covariance, per-term p-values, AIC/BIC and an
interpretable summary.  If you need "is this term significant?" or "which of
these models should I report?", the Wood backend answers directly.

**Constraints.**  Conversely, the Eilers-Marx backend owns soft shape
constraints (monotonic/convex/concave pushes) that the Wood backend does
not.  If physics demands a monotone dose–response, the Eilers-Marx backend
is the one with the knob.  Both handle periodicity, differently: the
Eilers-Marx backend wraps its basis *and* uses a circular difference
penalty; the Wood backend uses a purpose-built cyclic cubic or cyclic
P-spline basis.

**Speed and scaling.**  With few terms and a small grid the Eilers-Marx backend
is effectively instant; its cost grows with the product of grid sizes and
the number of :math:`\lambda` slots, which is why the two-phase
QR-caching matters.  The Wood backend's cost is dominated by the
optimiser: typically 10–100 objective evaluations, each one Cholesky —
flat in the number of grid points, but sequential.  In practice: Eilers-Marx
wins for sweeping many configurations; Wood wins for a single
many-termed fit with automatically tuned :math:`\lambda` per term.

**The deep difference in one sentence.**  The Eilers-Marx backend asks
*"which of these :math:`\lambda` candidates predicts best out of
sample?"* (GCV over a grid); the Wood backend asks *"which
:math:`\lambda` makes the data most likely under a smoothness prior?"*
(REML over a continuous parameter).  Everything else — bases, solvers,
covariances — follows from that choice of question.

.. _section-backends-choose:

Which backend should I use?
---------------------------

* **You want p-values, a detailed summary table, or parity with Wood's methods** →
  ``backend='wood'`` with default REML.
* **You need automatic per-term smoothing with many terms** → Wood's
  optimisation scales better than a per-term grid.
* **You want monotonic/convex/concave pushes** →
  ``backend='eilers'``; these features don't exist in the Wood backend.
* **You want to sweep basis counts and smoothing parameters and inspect
  the GCV surface** → Eilers-Marx's grid search, threaded.
* **You want the simplest possible mental model** (one basis, one
  penalty, one criterion) → Eilers-Marx.
* **You're teaching GAMs** → Eilers-Marx for the machinery, then Wood for the
  statistics; this chapter is laid out in that order.

When in doubt, fit both and compare: the decomposition plots and the
fitted curves should agree on the data's shape even where the p-values and
:math:`\lambda` philosophies differ.  Disagreement between the two engines
on the same data is a signal worth investigating — usually it means
under-smoothing by GCV or a basis count too small for the grid being searched.


.. _section-summary:

Key takeaways
-------------

.. list-table::
   :header-rows: 1

   * - Concept
     - One-sentence summary
   * - Additive structure
     - Every prediction is a sum of independent, plottable pieces (both backends)
   * - B-spline basis (Eilers-Marx)
     - Build any smooth curve from localised LEGO-brick basis functions
   * - Penalty (both)
     - Charge a tax for wiggles; let data pay it only when worthwhile
   * - :math:`\lambda` (both)
     - The tax rate: 0 = unpenalised, infinity = straight line
   * - EDF (both)
     - How many "effective parameters" the penalty allows the model to use
   * - GCV (Eilers-Marx; Wood option)
     - Approximate leave-one-out CV in a single fit; the Eilers-Marx model-selection compass
   * - REML (Wood default)
     - The restricted likelihood with the penalty as a prior; the Wood model-selection compass
   * - SVD solver (Eilers-Marx)
     - Numerically stable solution without squaring condition numbers
   * - QR + Cholesky (Wood)
     - Compact, fast solves; :math:`\log|H|` and EDF fall out of the Cholesky factor
   * - Basis zoo (Wood)
     - tp/ts, cr/cs, cc, ps, re — different penalties and null spaces for different jobs
   * - Sum-to-zero (Wood)
     - Absorbed into the basis, so every smooth is centred by construction
   * - Wood F-tests (Wood)
     - Approximate per-term significance tests with p-values
   * - Shape constraints (Eilers-Marx)
     - Soft monotonic/convex/periodic pushes via the choice of penalty
   * - Observation weights (both)
     - Scale observations by precision or down-weight known outliers via ``weights``
   * - Grid search (Eilers-Marx)
     - Two-phase QR+Cholesky: precompute K-dim work, then evaluate all lambdas in parallel
   * - Comparison rule of thumb
     - Wood for inference and automatic :math:`\lambda`; Eilers-Marx for constraints and fast sweeps

