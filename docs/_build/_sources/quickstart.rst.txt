Quick Start
===========

Installation
------------

linGAM requires Python ≥3.8 and the following packages:

* ``numpy >= 1.20``
* ``scipy >= 1.7``
* ``matplotlib >= 3.5``

Install from the repository root::

   pip install -e .

For building documentation locally::

   pip install sphinx sphinx-rtd-theme
   cd docs
   make html


Your first model
----------------

.. code-block:: python

   import numpy as np
   from lingam import GAMCore

   np.random.seed(42)
   n = 200
   x = np.random.uniform(0, 10, (n, 2))
   y = np.sin(x[:, 0]) + 0.3 * x[:, 1] + np.random.normal(0, 0.2, n)

   model = GAMCore("s(0, n_splines=12) + l(1)")
   model.fit(x, y)

   print(f"GCV = {model.statistics_['GCV']:.3f}")
   print(f"EDF = {model.statistics_['edof']:.2f}")

   y_hat = model.predict(x)
   ci = model.confidence_intervals(x, width=0.95)


Formula syntax
--------------

Terms are separated by ``+``.  Each term has the form ``type(args, kwargs)``.

``s(feature, ...)`` — univariate spline
   ``s(0, n_splines=10, lam=1.0, spline_order=3, constraint=None)``

``te(f1, f2, ...)`` — tensor-product interaction
   ``te(0, 1, n_splines=5, lam=1.0, spline_order=3)``
   Per-dimension grids can be lists: ``n_splines=[5, 8]``

``f(feature, ...)`` — categorical factor
   ``f(2, lam=1.0, coding='one-hot')``

``l(feature, ...)`` — linear term
   ``l(3, lam=0.0)``

Example formula::

   "s(0, n_splines=10, constraint='mono_inc') + te(0, 1, n_splines=[6, 8]) + f(2) + l(3)"


Single-term model (LinGAM)
--------------------------

For a single smooth :math:`y = f(x) + \varepsilon`, use ``LinGAM``:

.. code-block:: python

   from lingam import LinGAM

   model = LinGAM(n_splines=12, lam=1.0, spline_order=3)
   model.fit(x, y)

   model.gridsearch(x, y)          # fast QR+Cholesky path
   model.plot_decomposition(x, y)  # visualise basis functions

Shape constraints work too::

   model = LinGAM(n_splines=12, constraint='mono_inc')
   model.fit(x, y)


Robust fitting
--------------

Pass ``robust=True`` to down-weight outliers with Huber IRLS:

.. code-block:: python

   model = GAMCore("s(0) + l(1)")
   model.fit(x, y, robust=True)          # Huber-weighted IRLS

   model.gridsearch(x, y, robust=True)   # robust grid search

Internally this iteratively reweights observations using the Huber loss
with threshold :math:`c = 1.345 \, \text{MAD} / 0.6745`.


Partial dependence
------------------

Inspect the contribution of each term:

.. code-block:: python

   pdep_0 = model.partial_dependence(0, x)  # s(0) contribution
   pdep_1 = model.partial_dependence(1, x)  # l(1) contribution


Plotting
--------

.. code-block:: python

   fig, axes = model.plot_decomposition(x, y, n_grid=200, extrap_frac=0.15)

This shows the overall fit plus one panel per term.  Extrapolated regions
are shaded red.  Tensor terms produce 2-D heatmaps when ``d == 2``.
