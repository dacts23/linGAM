gamkit
======

**Simple Fast Linear GAM** — a high-performance Python implementation of
Generalized Additive Models supporting both **Eilers & Marx (1996)** and **Wood (2011)** backends.

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


What is gamkit?
---------------

gamkit fits smooth, additive regression models of the form

.. math::

   y_i = \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + \dots + f_p(x_{ip}) + \varepsilon_i

where each :math:`f_j` is a flexible smooth function *learned from data*,
penalised to avoid overfitting.

The package provides two complementary engines accessible through a single, unified interface:

1. **Wood (2011) Backend** (``backend='wood'``):
   Python implementation of Simon Wood's GAM methods (Wood 2011, 2017).
   Supports multiple basis types (``'tp'``, ``'ts'``, ``'cr'``, ``'cs'``, ``'cc'``,
   ``'ps'``, ``'cp'``, ``'bs'``, ``'re'``, ``'te'``), null-space constraint absorption,
   and **automatic REML / GCV smoothing parameter optimization** (no grid search required).
   Outputs formatted summary tables with approximate F-tests and p-values (Wood 2013).

2. **Eilers & Marx (1996) Backend** (``backend='eilers'``):
   Clean, fast B-splines with discrete GCV grid search and soft shape constraints
   (monotonic increasing/decreasing, convex, concave, periodic).

The package supports:

* **Univariate splines** ``s()`` — smooth nonlinear trends
* **Tensor-product interactions** ``te()`` — multidimensional interactions
* **Categorical factors** ``f()`` — dummy / one-hot coding
* **Linear terms** ``l()`` — parametric linear effects
* **Shape constraints** — monotonicity, convexity, concavity, periodicity
* **Fast evaluation** — QR decomposition with SVD data-augmentation to avoid squaring condition numbers
* **Confidence & prediction intervals** — Bayesian posterior covariance and frequentist intervals

When to use gamkit
^^^^^^^^^^^^^^^^^^

* You need more flexibility than linear regression but full interpretability without black-box complexity.
* You want the exact statistical rigor and bases of Simon Wood's GAM formulation in pure Python.
* You have domain constraints (monotonic, concave, periodic) to enforce on model components.
* You need to *explain* predictions to domain experts — each term produces a component plot.


Quick example
-------------

Using the **Wood backend** with automatic REML smoothing parameter estimation:

.. code-block:: python

   import numpy as np
   from gamkit import LinGAM, GAMCore

   np.random.seed(42)
   x = np.linspace(0, 1, 200)
   y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, 200)

   # Single-term model with cubic regression splines and REML estimation
   model = LinGAM(k=10, bs="cr", backend="wood")
   model.fit(x, y)

   # Formatted summary table
   print(model.summary())

   # Predictions & Bayesian confidence intervals
   y_hat = model.predict(x)
   ci = model.confidence_intervals(x, width=0.95)

Multi-term formula with tensor products:

.. code-block:: python

   gam = GAMCore("te(0, 1, k=5) + l(2)", backend="wood")
   gam.fit(X, y)
   print(gam.summary())


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
