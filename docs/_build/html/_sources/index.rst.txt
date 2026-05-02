linGAM
======

**Simple Fast Linear GAM** — a minimal, high-performance implementation of
Generalized Additive Models using penalised B-splines.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   math
   api
   constraints
   gridsearch
   examples
   changelog


What is linGAM?
---------------

linGAM fits smooth, additive regression models of the form

.. math::

   y_i = \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + \dots + f_p(x_{ip}) + \varepsilon_i

where each :math:`f_j` is a flexible smooth function *learned from data*,
penalised to avoid overfitting.

In plain language: instead of assuming "y goes up by 2.3 for every unit of x"
(a straight line), the model figures out the *shape* of the relationship on
its own.  It can find saturation curves, optimal operating points,
diminishing returns, and interactions — all while keeping each effect
separate and interpretable.

The package supports:

* **Univariate splines** ``s()`` — smooth nonlinear trends (e.g., "how does
  temperature affect blade life?")
* **Tensor-product interactions** ``te()`` — multidimensional smooths (e.g.,
  "does high temperature + high vibration make things worse than either
  alone?")
* **Categorical factors** ``f()`` — one-hot / dummy encoding (e.g., "does
  coating type A last longer than B?")
* **Linear terms** ``l()`` — plain linear coefficients (e.g., "more cooling
  flow → linearly longer life")
* **Shape constraints** — monotonicity, convexity, concavity, periodicity
  (encode what physics tells you)
* **Robust fitting** — Huber IRLS for resistance to sensor outliers
* **Fast grid search** — QR+Cholesky GCV optimisation with selective term
  search (automatically find the right amount of smoothing)

Key design goals
^^^^^^^^^^^^^^^^

* **Speed**:  All grid search uses QR precomputation + threaded Cholesky
  evaluation.  NumPy releases the GIL, so ``ThreadPoolExecutor`` parallelises
  CPU-bound linear algebra.
* **Correctness**:  SVD data-augmentation solver avoids squaring condition
  numbers.  No :math:`X^\top X` is ever formed explicitly.
* **Simplicity**:  Formula interface ``s(0) + te(1,2) + f(3)`` mirrors R/mgcv
  and pyGAM conventions.
* **Flexibility**:  Choose which terms participate in grid search, apply
  shape constraints, or mix standard and robust fitting.

When to use linGAM
^^^^^^^^^^^^^^^^^^

* You need more flexibility than linear regression but more interpretability
  than a neural network.
* You have domain knowledge about shapes (monotonic, concave, periodic) that
  you want to bake into the model.
* You need to *explain* predictions to colleagues — each term produces a
  plot, not a black-box score.
* Your data has outliers and you want the fit to ignore them automatically.
* You want automatic hyperparameter tuning via grid search over smoothing
  parameters.


Quick example
-------------

.. code-block:: python

   import numpy as np
   from lingam import GAMCore

   # Synthetic data
   np.random.seed(42)
   x = np.random.uniform(0, 10, (200, 2))
   y = 2 * x[:, 0] + 0.5 * x[:, 1] ** 2 + np.random.normal(0, 0.5, 200)

   # Fit a GAM: smooth for x0, linear for x1
   model = GAMCore("s(0, n_splines=10) + l(1)")
   model.fit(x, y)

   # Predict & confidence intervals
   y_hat = model.predict(x)
   ci = model.confidence_intervals(x, width=0.95)

   # Grid search (fast QR+Cholesky path)
   model.gridsearch(x, y)
   print(model.statistics_)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
