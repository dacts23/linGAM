API Reference
=============

This page documents the public API and internal engines in magli.

.. currentmodule:: magli

Public Unified API
------------------

The top-level :class:`GAM` (aliased as :class:`LinGAM`) and :class:`GAMCore` classes dynamically dispatch
to either the Wood (``backend='wood'``, default) or Eilers-Marx (``backend='eilers'``) engine.
Backend-specific classes are also directly importable from :mod:`magli` as
``WoodLinGAM``, ``WoodGAMCore``, ``EilersLinGAM``, and ``EilersGAMCore``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMCore
   LinGAM


Shared Architecture
-------------------

.. currentmodule:: magli._base

magli._base
^^^^^^^^^^^^

Shared ``_GAMBase`` lifecycle inherited by both engines, handling prediction, partial
dependence, confidence/prediction intervals, information criteria, and all
decomposition/diagnostic plotting.  (``scale`` is σ for Eilers-Marx, σ² for Wood.)

.. automodule:: magli._base
    :members:

.. currentmodule:: magli._formula

magli._formula
^^^^^^^^^^^^^^^

Shared additive model formula parser for ``s()``, ``te()``, ``f()``, and ``l()`` terms.

.. automodule:: magli._formula
   :members:

.. literalinclude:: ../src/magli/_formula.py
   :language: python
   :linenos:


Eilers & Marx Engine (``magli.eilers``)
---------------------------------------------

B-spline additive models with discrete GCV grid search and soft shape constraints
following Eilers & Marx (1996).

.. currentmodule:: magli.eilers

Classes
^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LinGAM
   GAMCore

Modules
^^^^^^^

.. currentmodule:: magli.eilers._bspline

magli.eilers._bspline
"""""""""""""""""""""""""""
.. automodule:: magli.eilers._bspline
   :members:

.. literalinclude:: ../src/magli/eilers/_bspline.py
   :language: python
   :linenos:

.. currentmodule:: magli.eilers._penalty

magli.eilers._penalty
"""""""""""""""""""""""""""
.. automodule:: magli.eilers._penalty
   :members:

.. literalinclude:: ../src/magli/eilers/_penalty.py
   :language: python
   :linenos:

.. currentmodule:: magli.eilers._solver

magli.eilers._solver
""""""""""""""""""""""""""
.. automodule:: magli.eilers._solver
   :members:

.. literalinclude:: ../src/magli/eilers/_solver.py
   :language: python
   :linenos:

.. currentmodule:: magli.eilers._terms

magli.eilers._terms
"""""""""""""""""""""""""
.. automodule:: magli.eilers._terms
   :members:

.. literalinclude:: ../src/magli/eilers/_terms.py
   :language: python
   :linenos:

.. currentmodule:: magli.eilers._lingam

magli.eilers._lingam
""""""""""""""""""""""""""
.. automodule:: magli.eilers._lingam
   :members:

.. literalinclude:: ../src/magli/eilers/_lingam.py
   :language: python
   :linenos:


.. currentmodule:: magli.eilers._gam

magli.eilers._gam
"""""""""""""""""""""""
.. automodule:: magli.eilers._gam
   :members:
   :exclude-members: _set_lams_from_combo, _assemble_penalty_from_bases,
                     _precompute_penalty_bases, _build_matrix_from_terms

.. literalinclude:: ../src/magli/eilers/_gam.py
   :language: python
   :linenos:


Wood Engine (``magli.wood``)
-----------------------------

Python implementation of Simon Wood's GAM methods (Wood 2011, 2017) with multiple basis types,
null-space constraint absorption, and continuous REML / GCV smoothing optimization.

.. currentmodule:: magli.wood

Classes
^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LinGAM
   GAMCore

Modules
^^^^^^^

.. currentmodule:: magli.wood._smooths

magli.wood._smooths
""""""""""""""""""""
.. automodule:: magli.wood._smooths
   :members:

.. literalinclude:: ../src/magli/wood/_smooths.py
   :language: python
   :linenos:

.. currentmodule:: magli.wood._optimizer

magli.wood._optimizer
""""""""""""""""""""""
.. automodule:: magli.wood._optimizer
   :members:

.. literalinclude:: ../src/magli/wood/_optimizer.py
   :language: python
   :linenos:

.. currentmodule:: magli.wood._terms

magli.wood._terms
""""""""""""""""""
.. automodule:: magli.wood._terms
   :members:

.. literalinclude:: ../src/magli/wood/_terms.py
   :language: python
   :linenos:

.. currentmodule:: magli.wood._lingam

magli.wood._lingam
"""""""""""""""""""
.. automodule:: magli.wood._lingam
   :members:

.. literalinclude:: ../src/magli/wood/_lingam.py
   :language: python
   :linenos:


.. currentmodule:: magli.wood._gam

magli.wood._gam
""""""""""""""""
.. automodule:: magli.wood._gam
   :members:

.. literalinclude:: ../src/magli/wood/_gam.py
   :language: python
   :linenos:
