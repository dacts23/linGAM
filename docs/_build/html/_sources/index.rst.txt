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

   y_i = \beta_0 + \sum_{j=1}^{p} f_j(x_{ij}) + \varepsilon_i

where each :math:`f_j` is a flexible smooth function learned from data,
penalised to avoid overfitting.  The package supports:

* **Univariate splines** ``s()`` — smooth nonlinear trends
* **Tensor-product interactions** ``te()`` — multidimensional smooths
* **Categorical factors** ``f()`` — one-hot / dummy encoding
* **Linear terms** ``l()`` — plain linear coefficients
* **Shape constraints** — monotonicity, convexity, concavity, periodicity
* **Robust fitting** — Huber IRLS for outlier resistance
* **Fast grid search** — QR+Cholesky GCV optimisation with selective term search

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
