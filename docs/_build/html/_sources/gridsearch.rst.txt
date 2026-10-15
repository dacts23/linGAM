Smoothing Parameter Optimization & Tuning
===========================================

A central question in Generalized Additive Models is choosing the smoothing parameter :math:`\lambda`
(or :math:`sp`) to achieve the optimal balance between fidelity to data and smoothness.

linGAM supports two distinct mathematical approaches:

1. **Continuous REML Optimization (R mgcv engine)** — Solves for optimal :math:`\lambda` continuously during ``fit()`` (Wood 2011).
2. **Discrete GCV Grid Search (pyGAM engine)** — Evaluates a Cartesian grid of candidate :math:`(K, \lambda)` values via cached QR+Cholesky.


1. Continuous REML Optimization (R mgcv)
----------------------------------------

In the R ``mgcv`` backend, smoothing parameters are estimated automatically during model fitting:

.. code-block:: python

   from lingam import LinGAM, GAMCore

   # Single-term: REML estimates lambda automatically
   m = LinGAM(k=10, bs="cr", backend="mgcv")
   m.fit(x, y)
   print(f"Optimal estimated lambda: {m.lam:.4f}")

   # Multi-term: Simultaneous multi-dimensional REML optimization
   g = GAMCore("s(0, k=10) + s(1, k=10) + l(2)", backend="mgcv")
   g.fit(X, y)
   print(g.summary())

How it works:
* Uses Newton-Raphson / BFGS optimization of the Restricted Maximum Likelihood (REML) or Generalized Cross Validation (``method='GCV.Cp'``) criterion in continuous log-lambda space.
* Outer iteration updates :math:`\log \boldsymbol{\lambda}`; inner step computes penalized least-squares coefficients :math:`\boldsymbol{\beta}` and hat matrix traces.
* Avoids the discretization error, manual grid configuration, and combinatorial scaling of discrete grids.


2. Discrete GCV Grid Search (pyGAM)
-----------------------------------

The pyGAM backend evaluates a user-defined or default grid of :math:`(K, \lambda)` candidates by minimising
the Generalised Cross Validation (GCV) criterion.

.. code-block:: python

   from lingam import LinGAM, GAMCore

   m = LinGAM(n_splines=10, backend="pygam")
   m.gridsearch(x, y)

Two-phase fast QR+Cholesky evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Phase 1 (K precomputation)** — For each candidate spline dimension :math:`K`, construct design matrix :math:`X`,
   compute its QR decomposition :math:`X = QR`, and cache :math:`R^\top R` and :math:`R^\top Q^\top y`.
   These depend only on :math:`K` and are computed once.

2. **Phase 2 (Lambda evaluation)** — For each candidate :math:`\lambda`, assemble penalty matrix :math:`P = \lambda S`,
   solve :math:`(R^\top R + P) \boldsymbol{\beta} = R^\top Q^\top y` via Cholesky factorization, compute EDF from the
   eigen-structure, and record GCV.

Default search grids
^^^^^^^^^^^^^^^^^^^^

* **Lambda** — ``np.logspace(-3, 3, 11)`` per slot (11 values from 0.001 to 1000).
* **n_splines** — ``np.arange(5, 25, 2)`` for ``s()`` (10 values), and ``np.arange(4, 8)`` per dimension for ``te()``.

Selective search (``search_terms``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When confident in the parameters for some terms, restrict the grid search to a subset:

.. code-block:: python

   model = GAMCore("s(0) + s(1) + te(0, 1)", backend="pygam")

   # Search only univariate splines (s)
   model.gridsearch(x, y, search_terms=['s'])

   # Search only term 0 and factor terms
   model.gridsearch(x, y, search_terms=[0, 'f'])

Custom grids per term slot
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model.gridsearch(
       x, y,
       lam_grids=[
           np.logspace(-2, 2, 5),   # s(0)
           np.logspace(-1, 1, 3),   # te dim 1
           np.logspace(-1, 1, 3),   # te dim 2
       ],
       n_splines_grids=[
           np.arange(5, 15, 2),     # s(0)
           np.arange(4, 8),          # te dim 1
           np.arange(4, 8),          # te dim 2
       ],
   )
