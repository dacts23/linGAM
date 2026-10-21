Changelog
=========

Unreleased
----------

* **Self-checking teaching notebooks** (documentation only, no library changes):
  two executed notebooks now walk through the mathematics and the workflow of
  each backend, with every derivation followed by an assertion and a closing
  pass/fail audit:
  `examples/wood_backend_math_tutorial.ipynb
  <https://github.com/dacts23/magli/blob/main/examples/wood_backend_math_tutorial.ipynb>`_
  and
  `examples/eilers_backend_math_tutorial.ipynb
  <https://github.com/dacts23/magli/blob/main/examples/eilers_backend_math_tutorial.ipynb>`_.
  They are linked from
  :doc:`quickstart`, :doc:`examples`, :doc:`math` and the backend math chapters.

Version 1.0.0
-------------
*Released 2026-10-18*

* **Reference Parity with R ``mgcv`` & ``pyGAM``**:
  Full numerical and mathematical alignment against reference implementations.
* **Exact Integrated Derivative B-Spline Penalties**:
  Upgraded Wood backend ``bs`` basis penalty to Gauss-Legendre quadrature of
  integrated squared derivatives matching R's ``smooth.construct.bs.smooth.spec``.
* **REML Null Space Dimension**:
  Corrected REML calculation of unpenalised null space dimension
  :math:`M_p = p - \operatorname{rank}(S_{\text{total}})` and added coarse-grid
  seeding before L-BFGS-B optimization.
* **Active-Set Shape Constraints**:
  Enforced ``pyGAM`` active-set asymmetric penalty formulation with
  :math:`\lambda_{\text{con}} = 10^9`, :math:`\lambda_{L2} = 10^{-3} I`, and
  exact periodic basis column wrapping.
* **Verification & Test Suite**:
  55 automated tests verifying numerical tolerances against R ``mgcv`` and mathematical invariants.

Version 0.3.1
-------------
*Released 2026-10-17*

* Updated documentation styling and replaced em dashes across all guides and examples.
* Added comparative overview with R mgcv and pygam.
* Enhanced package keywords, canonical metadata, and repository discoverability.

Version 0.3.0
-------------
*Released 2026-10-15*

New features
^^^^^^^^^^^^^

* **Wood (2011) backend**: Native implementation of Simon Wood's GAM methods
  supporting thin plate splines (``'tp'``, ``'ts'``), cubic regression splines
  (``'cr'``, ``'cs'``), cyclic cubic splines (``'cc'``), P-splines (``'ps'``,
  ``'cp'``), B-splines (``'bs'``), random effects / ridge (``'re'``), and tensor
  products (``'te'``).
* **Automatic REML smoothing optimization**: Smoothing parameters are solved
  continuously via Newton-Raphson / BFGS minimization of REML or GCV criteria
  without needing discrete grid searches.
* **Summary tables**: ``model.summary()`` provides parametric coefficients,
  effective degrees of freedom (EDF), approximate F-tests, and p-values matching
  Wood (2013).
* **Symmetric dual-engine architecture**: Symmetrically organized into
  ``magli.eilers`` and ``magli.wood`` with unified ``LinGAM`` and ``GAMCore``
  factory dispatchers.

Improvements
^^^^^^^^^^^^

* **Shared ``_GAMBase`` lifecycle**: ``magli._base`` now holds the
  code both engines inherit (``predict``, ``partial_dependence``,
  confidence/prediction intervals, ``loglikelihood``/``aic``/``bic``/
  ``deviance_explained``, and all decomposition/diagnostic plotting).
  Note the scale convention it abstracts, where Eilers-Marx stores ``scale`` as
  σ (uses ``s**2`` as variance) and Wood stores it as σ² directly.
* **Observation weights across both backends**: ``fit(x, y, weights=w)`` and
  ``gridsearch(x, y, weights=w)`` scale observations by :math:`\sqrt{w_i}` in
  both the Eilers-Marx and Wood backends (replacing the earlier ``robust=True``
  IRLS option).
* **Cyclic P-splines (``'cp'``)**: dedicated basis with wrapped knots and a wrapped difference penalty.
* **Thin-plate shrinkage (``'ts'``)**: now charges the null-space
  (smallest) eigenvalues as in Wood (2017), instead of the largest ones.
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
* **Fast path is now the only path**: removed ``fast`` parameter from
  ``gridsearch()``.  All grid searches use QR+Cholesky by default.
* **Robust grid search** uses dedicated IRLS path with threaded evaluation.
* **Statistics methods** added to both ``GAMCore`` and ``LinGAM``:
  ``loglikelihood()``, ``aic()``, ``bic()``, ``deviance_explained()``.
* **Selective grid search** (via ``search_terms`` parameter): restrict tuning
  to specific term indices or types.

Improvements
^^^^^^^^^^^^^

* Modular ``src/magli/`` package layout with ``__slots__`` for memory
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
