Grid Search & Hyperparameter Tuning
===================================

linGAM's grid search automatically selects the optimal number of basis
functions ``n_splines`` and smoothing parameter ``lam`` for each term by
minimising Generalised Cross Validation (GCV).  The implementation is fully
parallel and uses a fast QR+Cholesky path.

How it works
------------

1. **Cartesian product search.**  For each candidate ``n_splines`` (or
tensor-product ``n_splines_per``) and each candidate ``lam`` (or
``lam_per``), the model is fit and its GCV score is recorded.

2. **Two-phase evaluation.**

   *Phase 1* — For each unique ``n_splines`` combination, build the design
   matrix :math:`X`, compute its QR decomposition, and cache
   :math:`R^\top R` and :math:`R^\top Q^\top y`.  These are independent of
   :math:`\lambda` and are computed once.

   *Phase 2* — For every :math:`(K, \lambda)` pair, assemble the penalty
   from precomputed bases, Cholesky-solve for coefficients and EDF, and
   compute GCV.  Both phases are threaded via
   ``concurrent.futures.ThreadPoolExecutor``.

3. **Best candidate wins.**  The hyperparameters with the lowest GCV are
   applied to the model, which is then re-fit.

Default search grids
--------------------

* **Lambda** — ``np.logspace(-3, 3, 11)`` per slot (11 values from 0.001 to
  1000).  Log spacing is natural because :math:`\lambda` has a multiplicative
  effect on wiggliness.
* **n_splines** — ``np.arange(5, 25, 2)`` for ``s()`` (10 values), and
  ``np.arange(4, 8)`` per dimension for ``te()``.

These defaults are chosen to cover a wide range of flexibility without being
prohibitively expensive.

Selective search (search_terms)
-------------------------------

When you are confident about some terms, you can restrict the search to a
subset.  This dramatically reduces the Cartesian product size and speeds up
tuning.

.. code-block:: python

   # Search only the first term
   model.gridsearch(x, y, search_terms=[0])

   # Search all spline and tensor terms
   model.gridsearch(x, y, search_terms=['s', 'te'])

   # Search term 0 and all factor terms
   model.gridsearch(x, y, search_terms=[0, 'f'])

   # Search everything (default)
   model.gridsearch(x, y, search_terms=None)

Non-searched terms keep their current ``lam`` and ``n_splines`` values.  Their
grids are pinned to a single candidate, so they contribute no extra
combinations.

Custom grids
------------

You can override defaults per slot.  A "slot" is one scalar hyperparameter
position:

* ``s()`` — 1 lam slot, 1 n_splines slot
* ``te()`` — ``d`` lam slots, ``d`` n_splines slots
* ``f()`` — 1 lam slot, 0 n_splines slots
* ``l()`` — 1 lam slot, 0 n_splines slots

.. code-block:: python

   # Custom lambda grids for a 3-term model: s(0) + te(1,2) + f(3)
   # Slots: [lam_s0, lam_te1, lam_te2, lam_f3]
   model.gridsearch(
       x, y,
       lam_grids=[
           np.logspace(-2, 2, 5),   # s(0)
           np.logspace(-1, 1, 3),   # te dim 1
           np.logspace(-1, 1, 3),   # te dim 2
           np.array([1.0]),          # f(3) — fixed
       ],
       n_splines_grids=[
           np.arange(5, 15, 2),     # s(0)
           np.arange(4, 8),          # te dim 1
           np.arange(4, 8),          # te dim 2
       ],
   )

If you pass a single grid, it is broadcast to all slots.

Robust grid search
------------------

When ``robust=True``, each candidate is evaluated with Huber IRLS (10
iterations per candidate).  This is slower than the standard path because
IRLS is iterative, but it protects against outliers distorting the GCV
score.  The search is still threaded over lambda combinations.

.. code-block:: python

   model.gridsearch(x, y, robust=True)

**Note:** The robust path does not use the two-phase QR cache because the
weights change every IRLS iteration.  Each candidate is evaluated with a
fresh weighted solve.

LinGAM grid search
------------------

``LinGAM`` has the same interface but with scalar grids:

.. code-block:: python

   model = LinGAM(n_splines=10)
   model.gridsearch(
       x, y,
       lam=np.logspace(-3, 3, 11),
       n_splines=np.arange(5, 25, 2),
   )

This is always the fast QR+Cholesky path.  Pass ``robust=True`` for Huber
IRLS evaluation.

Tips for faster search
----------------------

1. **Start coarse, then refine.**  Use a sparse grid (e.g. 5 lambdas, 5
   n_splines) to locate the rough optimum, then zoom in.
2. **Fix confident terms.**  Use ``search_terms`` to skip terms you already
   know.
3. **Reduce ``gamma``** if the model is underfitting (default 1.4 is
   conservative).
4. **Avoid very large ``n_splines``.**  The Cholesky cost is
   :math:`O(m^3)` where :math:`m = \sum K_j + 1`.  Keeping
   :math:`K \le 25` is usually sufficient.
