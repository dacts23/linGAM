Quick Start
===========

Installation
------------

gamkit requires Python ≥3.8 and the following packages:

* ``numpy >= 1.20``
* ``scipy >= 1.7``
* ``matplotlib >= 3.5``

Install from the repository root::

   pip install -e .

For building documentation locally::

   pip install sphinx furo sphinx-copybutton
   cd docs
   make html


Unified API & Backends
----------------------

gamkit provides two backends via the unified :class:`LinGAM` and :class:`GAMCore` classes:

* ``backend='wood'`` (default) — Simon Wood's GAM algorithm (continuous REML or GCV optimization, multiple basis families, and Wood 2013 summary tables).
* ``backend='eilers'`` — B-spline basis with discrete GCV grid search and soft shape constraints (Eilers & Marx 1996).


1. Wood (2011) Backend (Automatic REML)
---------------------------------------

In the Wood engine, smoothing parameters are estimated automatically during fit via REML
(Restricted Maximum Likelihood) or GCV, so no discrete grid search is necessary.

.. code-block:: python

   import numpy as np
   from gamkit import LinGAM, GAMCore

   np.random.seed(42)
   n = 200
   x = np.linspace(0, 1, n)
   y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, n)

   # Single-term model with cubic regression splines (cr)
   model = LinGAM(k=10, bs="cr", backend="wood")
   model.fit(x, y)

   # Print summary table with F-tests & p-values (Wood 2013)
   print(model.summary())

   # Predictions & Bayesian confidence intervals
   y_hat = model.predict(x)
   ci = model.confidence_intervals(x, width=0.95)

Multi-term formula with tensor products:

.. code-block:: python

   X = np.random.uniform(0, 10, (200, 3))
   y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 1.2 * X[:, 2] + np.random.normal(0, 0.2, 200)

   gam = GAMCore("te(0, 1, k=5, bs='cr') + l(2)", backend="wood")
   gam.fit(X, y)
   print(gam.summary())


2. Eilers & Marx (1996) Backend (GCV Grid Search)
-------------------------------------------------

The Eilers-Marx engine uses B-splines and GCV grid search over candidate lambda and spline count grids:

.. code-block:: python

   from gamkit import LinGAM, GAMCore

   model = LinGAM(n_splines=12, backend="eilers")
   model.gridsearch(x, y)          # Fast QR+Cholesky GCV evaluation
   model.plot_decomposition(x, y)  # Visualise basis decomposition

Shape constraints:

.. code-block:: python

   model = LinGAM(n_splines=12, constraint='mono_inc', backend="eilers")
   model.fit(x, y)

Observation weights (precision weights or down-weighted outliers):

.. code-block:: python

   model = LinGAM(n_splines=12, backend="eilers")
   model.fit(x, y, weights=w)


Formula syntax
--------------

Terms are separated by ``+``. Each term has the form ``type(args, kwargs)``.

``s(feature, ...)`` — univariate spline
   Accepts both Wood syntax (``k=10``, ``bs='cr'``, ``sp=0.5``) and Eilers-Marx syntax
   (``n_splines=10``, ``lam=1.0``, ``constraint='mono_inc'``).

``te(f1, f2, ...)`` — tensor-product interaction
   ``te(0, 1, k=5, bs='cr')`` or ``te(0, 1, n_splines=5)``.

``f(feature, ...)`` — categorical factor
   ``f(2, coding='dummy')`` or ``f(2, coding='one-hot')``.

``l(feature, ...)`` — linear term
   ``l(3)`` (parametric unpenalized or ridge-penalized linear effect).


Partial dependence & Decomposition
----------------------------------

Inspect the additive contribution of individual terms:

.. code-block:: python

   pdep_0 = model.partial_dependence(0, x)  # Component effect of term 0
   pdep_1 = model.partial_dependence(1, x)  # Component effect of term 1

   # Multi-panel visualization (one panel per term)
   fig, axes = model.plot_decomposition(x, y)
