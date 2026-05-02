Shape Constraints
=================

linGAM supports soft shape constraints on univariate spline terms (``s()``)
and on ``LinGAM``.  These are not hard constraints (the solver does not use
quadratic programming or active-set methods).  Instead, they change the
penalty matrix so that the *natural* smooth solution tends toward the desired
shape.

Available constraints
---------------------

+----------------+-------------------+----------------------------------------+
| Constraint     | Penalty           | Effect                                 |
+================+===================+========================================+
| ``None``       | :math:`D_2^\top   | Standard smoothness (default)          |
|                | D_2`              |                                        |
+----------------+-------------------+----------------------------------------+
| ``mono_inc``   | :math:`D_1^\top   | Encourages monotonically increasing    |
|                | D_1`              | curve                                  |
+----------------+-------------------+----------------------------------------+
| ``mono_dec``   | :math:`D_1^\top   | Encourages monotonically decreasing    |
|                | D_1`              | curve                                  |
+----------------+-------------------+----------------------------------------+
| ``convex``     | :math:`D_2^\top   | Encourages convexity (U-shaped)        |
|                | D_2`              |                                        |
+----------------+-------------------+----------------------------------------+
| ``concave``    | :math:`D_2^\top   | Encourages concavity (∩-shaped)        |
|                | D_2`              |                                        |
+----------------+-------------------+----------------------------------------+
| ``periodic``   | Circular          | Enforces :math:`f(0) = f(1)` with      |
|                | :math:`D_2^\top   | wrapped basis and penalty              |
|                | D_2`              |                                        |
+----------------+-------------------+----------------------------------------+

How they work
-------------

**Monotonicity** uses a first-difference penalty:

.. math::

   P = \lambda \, D_1^\top D_1,
   \qquad
   D_1 = \begin{pmatrix}
   -1 & 1 &        &   \\
      & -1 & 1     &   \\
      &    & \ddots & \ddots
   \end{pmatrix}.

Minimising :math:`\beta^\top P \beta = \lambda \sum_i (\beta_{i+1} - \beta_i)^2`
forces adjacent B-spline coefficients to be similar.  Because B-spline basis
functions are ordered left-to-right across the domain, similar coefficients
produce a curve that does not reverse direction — hence monotonic.

* ``mono_inc`` applies the penalty as-is.
* ``mono_dec`` applies the same penalty; the data + negative trend
  naturally produces a decreasing fit because the unpenalised OLS direction
  is downward.

**Periodicity** wraps the basis and penalty:

* The B-spline basis is evaluated on :math:`[0,1]` with :math:`z = z \bmod 1`,
  so the domain is a circle.
* The last :math:`k-1` columns are folded into the first :math:`k-1` columns,
  reducing the coefficient count from :math:`K` to :math:`K - k + 1`.
* The second-difference penalty connects index :math:`K_{\text{eff}}` back
  to index 1, so :math:`\beta_{K_{\text{eff}}} \approx \beta_1`.

**Convexity / concavity** use the standard second-difference penalty.  The
difference is subtle: convexity is a property of the *second derivative*,
which is already penalised by :math:`D_2^\top D_2`.  The constraint keyword
is primarily a signal to the user and may be combined with stronger
:math:`\lambda` values to more aggressively penalise inflection points.

When to use which
-----------------

* **Monotonic increasing** — dose-response curves, learning curves,
  cumulative distributions, prices vs. demand.
* **Monotonic decreasing** — diminishing returns, survival curves,
  price elasticity.
* **Convex** — cost functions, loss landscapes, risk curves.
* **Concave** — utility functions, production functions.
* **Periodic** — time-of-day effects, seasonal patterns, angular
  coordinates (compass directions).

Caveats
-------

1. **Soft, not hard.**  With very noisy data or weak :math:`\lambda`, a
   ``mono_inc`` term can still locally decrease.  Increase :math:`\lambda`
   or use grid search to tighten the constraint.

2. **Intercept is free.**  The intercept column is never penalised, so the
   overall vertical shift is unrestricted.

3. **Periodicity reduces capacity.**  A periodic spline with :math:`K=10`
   and :math:`k=3` has only :math:`K_{\text{eff}} = 8` free coefficients.
   You may need slightly larger ``n_splines`` to capture the same complexity
   as a non-periodic spline.

4. **Not available for ``te()``.**  Tensor-product terms do not support
   shape constraints because the multi-dimensional penalty structure does not
   admit a simple ordered-difference formulation.  Apply constraints to the
   marginal ``s()`` terms instead.

Examples
--------

Monotonically increasing spline::

   model = GAMCore("s(0, constraint='mono_inc') + l(1)")
   model.fit(x, y)

Periodic seasonal effect::

   model = GAMCore("s(0, n_splines=12, constraint='periodic') + l(1)")
   model.fit(x, y)

Single-term with monotonicity::

   model = LinGAM(n_splines=12, constraint='mono_inc')
   model.fit(x, y)
