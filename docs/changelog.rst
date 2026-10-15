Changelog
=========

Version 0.3.0
-------------
*Released 2026-10-15*

New features
^^^^^^^^^^^^^

* **R mgcv backend**: Native implementation of Simon Wood's R ``mgcv`` package
  supporting thin plate splines (``'tp'``, ``'ts'``), cubic regression splines
  (``'cr'``, ``'cs'``), cyclic cubic splines (``'cc'``), P-splines (``'ps'``,
  ``'cp'``), B-splines (``'bs'``), random effects / ridge (``'re'``), and tensor
  products (``'te'``).
* **Automatic REML smoothing optimization**: Smoothing parameters are solved
  continuously via Newton-Raphson / BFGS minimization of REML or GCV criteria
  without needing discrete grid searches.
* **R-style summary tables**: ``model.summary()`` provides parametric coefficients,
  effective degrees of freedom (EDF), approximate F-tests, and p-values matching
  R's ``summary.gam`` (Wood 2013).
* **Symmetric dual-engine architecture**: Symmetrically organized into
  ``lingam.pygam`` and ``lingam.mgcv`` with unified ``LinGAM`` and ``GAMCore``
  factory dispatchers.

Improvements
^^^^^^^^^^^^

* **Simplified and unbloated**: Removed thread management overhead and unused
  Huber IRLS iterations.
* **Fixed R² computation**: Corrected deviance explained calculation against the
  sample mean null deviance.
* **Input validation**: Enforced NaN and Inf validation across all fit methods.
* **Extrapolation**: Explicit linear extrapolation for predictions beyond boundary knots.

Version 0.2.0
-------------
*Released 2026-05-01*

New features
^^^^^^^^^^^^^

* **Shape constraints** for ``s()`` terms and ``LinGAM``:
  ``mono_inc``, ``mono_dec``, ``convex``, ``concave``, ``periodic``.
* **Fast path is now the only path** — removed ``fast`` parameter from
  ``gridsearch()``.  All grid searches use QR+Cholesky by default.
* **Robust grid search** uses dedicated IRLS path with threaded evaluation.
* **Statistics methods** added to both ``GAMCore`` and ``LinGAM``:
  ``loglikelihood()``, ``aic()``, ``bic()``, ``deviance_explained()``.
* **Selective grid search** via ``search_terms`` parameter: restrict tuning
  to specific term indices or types.

Improvements
^^^^^^^^^^^^^

* Modular ``src/lingam/`` package layout with ``__slots__`` for memory
  efficiency.
* PEP8 naming: ``b_spline_basis``, ``solve_pirls``, ``irls_solve``, etc.
* ``b_spline_basis`` supports periodic wrapping.
* Penalty builders split into dedicated module with new
  ``first_diff_penalty`` and ``circular_second_diff_penalty``.

Version 0.1.0
-------------
*Released earlier*

* Initial release with ``LinGAM`` (single-term) and ``GAMCore`` (multi-term
  formula interface).
* B-spline basis via vectorised De Boor recursion.
* SVD data-augmentation solver (PIRLS).
* Huber IRLS robust fitting.
* GCV grid search with QR+Cholesky fast path and SVD basic path.
* Partial dependence, confidence/prediction intervals, and decomposition
  plotting.
