Examples
========

This chapter collects end-to-end examples demonstrating common use cases.

Basic regression
----------------

.. code-block:: python

   import numpy as np
   from lingam import GAMCore

   np.random.seed(42)
   n = 300
   x = np.column_stack([
       np.random.uniform(0, 10, n),
       np.random.uniform(0, 5, n),
   ])
   y = np.sin(x[:, 0]) + 0.3 * x[:, 1] + np.random.normal(0, 0.2, n)

   model = GAMCore("s(0, n_splines=12) + l(1)")
   model.fit(x, y)

   print("GCV  =", model.statistics_['GCV'])
   print("EDF  =", model.statistics_['edof'])
   print("AIC  =", model.aic())
   print("BIC  =", model.bic())
   print("R²   =", model.deviance_explained())

   y_hat = model.predict(x)
   ci = model.confidence_intervals(x)

   fig, axes = model.plot_decomposition(x, y)


Grid search with selective terms
--------------------------------

.. code-block:: python

   model = GAMCore("s(0) + s(1) + te(0, 1)")
   model.gridsearch(
       x, y,
       lam_grids=[np.logspace(-2, 2, 5)],
       n_splines_grids=[np.arange(5, 13, 2)],
       search_terms=['s'],  # only tune the univariate splines
   )

   print("Best s(0): K =", model._terms[0].n_splines,
         "lam =", model._terms[0].lam)


Categorical + interaction
-------------------------

.. code-block:: python

   np.random.seed(42)
   n = 400
   x = np.column_stack([
       np.random.uniform(0, 10, n),                    # continuous
       np.random.uniform(0, 5, n),                      # continuous
       np.random.choice([0, 1, 2], n).astype(float),   # categorical
   ])
   y = (
       np.sin(x[:, 0])
       + 0.3 * x[:, 1]
       + np.where(x[:, 2] == 0, -1, np.where(x[:, 2] == 1, 0, 2))
       + np.random.normal(0, 0.3, n)
   )

   model = GAMCore("s(0) + l(1) + f(2)")
   model.fit(x, y)
   model.gridsearch(x, y)

   # Partial dependence for each term
   pdep_s = model.partial_dependence(0, x)
   pdep_l = model.partial_dependence(1, x)
   pdep_f = model.partial_dependence(2, x)


Robust fitting with outliers
----------------------------

.. code-block:: python

   np.random.seed(42)
   x = np.random.uniform(0, 10, 200)
   y = np.sin(x) + np.random.normal(0, 0.2, 200)
   y[10] += 8   # massive outlier
   y[50] -= 6   # another outlier

   # Standard fit — outlier pulls the curve
   model_std = LinGAM(n_splines=12)
   model_std.fit(x, y)

   # Robust fit — outlier is down-weighted
   model_rob = LinGAM(n_splines=12)
   model_rob.fit(x, y, robust=True)

   print("Standard RSS:", model_std.statistics_['rss'])
   print("Robust RSS:  ", model_rob.statistics_['rss'])


Monotonic dose-response
-----------------------

.. code-block:: python

   np.random.seed(42)
   dose = np.random.uniform(0, 10, 150)
   response = 5 * (1 - np.exp(-0.4 * dose)) + np.random.normal(0, 0.3, 150)

   model = LinGAM(n_splines=12, constraint='mono_inc')
   model.fit(dose, response)

   # Grid search respects the monotonic constraint
   model.gridsearch(dose, response)

   fig, axes = model.plot_decomposition(dose, response)


Periodic seasonal pattern
-------------------------

.. code-block:: python

   np.random.seed(42)
   hour = np.random.uniform(0, 24, 300)
   temp = (
       20
       + 8 * np.sin(2 * np.pi * hour / 24)
       + 3 * np.random.normal(0, 1, 300)
   )

   # Periodic constraint encloses the daily cycle
   model = LinGAM(n_splines=12, constraint='periodic')
   model.fit(hour, temp)

   # Predict across a full day
   hour_grid = np.linspace(0, 24, 200)
   temp_hat = model.predict(hour_grid)

   # 6 AM and 6 PM should have the same prediction as 30 h and 42 h
   assert np.isclose(model.predict([6]), model.predict([30]), atol=1e-3)


2-D tensor interaction
----------------------

.. code-block:: python

   np.random.seed(42)
   n = 500
   x = np.column_stack([
       np.random.uniform(0, 10, n),
       np.random.uniform(0, 5, n),
   ])
   y = (
       np.sin(x[:, 0]) * np.cos(x[:, 1])
       + 0.5 * x[:, 0]
       + np.random.normal(0, 0.3, n)
   )

   model = GAMCore("s(0) + s(1) + te(0, 1, n_splines=6)")
   model.fit(x, y)
   model.gridsearch(x, y)

   # te(0,1) captures the interaction that s(0)+s(1) cannot
   pdep_te = model.partial_dependence(2, x)

   fig, axes = model.plot_decomposition(x, y)
