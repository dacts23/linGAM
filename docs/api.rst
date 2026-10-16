API Reference
=============

This page documents the public API and internal engines in gamkit.

.. currentmodule:: gamkit

Public Unified API
------------------

The top-level :class:`LinGAM` and :class:`GAMCore` classes dynamically dispatch
to either the Eilers-Marx or Wood engine based on the ``backend`` argument.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMCore
   LinGAM


Shared Architecture
-------------------

.. currentmodule:: gamkit._base

gamkit._base
^^^^^^^^^^^^

Shared ``_GAMBase`` lifecycle inherited by both engines: prediction, partial
dependence, confidence/prediction intervals, information criteria, and all
decomposition/diagnostic plotting.  (``scale`` is σ for Eilers-Marx, σ² for Wood.)

.. automodule:: gamkit._base
    :members:

.. currentmodule:: gamkit._formula

gamkit._formula
^^^^^^^^^^^^^^^

Shared additive model formula parser for ``s()``, ``te()``, ``f()``, and ``l()`` terms.

.. automodule:: gamkit._formula
   :members:

.. literalinclude:: ../src/gamkit/_formula.py
   :language: python
   :linenos:


Eilers & Marx Engine (``gamkit.eilers``)
---------------------------------------------

B-spline additive models with discrete GCV grid search and soft shape constraints
following Eilers & Marx (1996).

.. currentmodule:: gamkit.eilers

Classes
^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LinGAM
   GAMCore

Modules
^^^^^^^

.. currentmodule:: gamkit.eilers._bspline

gamkit.eilers._bspline
"""""""""""""""""""""""""""
.. automodule:: gamkit.eilers._bspline
   :members:

.. literalinclude:: ../src/gamkit/eilers/_bspline.py
   :language: python
   :linenos:

.. currentmodule:: gamkit.eilers._penalty

gamkit.eilers._penalty
"""""""""""""""""""""""""""
.. automodule:: gamkit.eilers._penalty
   :members:

.. literalinclude:: ../src/gamkit/eilers/_penalty.py
   :language: python
   :linenos:

.. currentmodule:: gamkit.eilers._solver

gamkit.eilers._solver
""""""""""""""""""""""""""
.. automodule:: gamkit.eilers._solver
   :members:

.. literalinclude:: ../src/gamkit/eilers/_solver.py
   :language: python
   :linenos:

.. currentmodule:: gamkit.eilers._terms

gamkit.eilers._terms
"""""""""""""""""""""""""
.. automodule:: gamkit.eilers._terms
   :members:

.. literalinclude:: ../src/gamkit/eilers/_terms.py
   :language: python
   :linenos:

.. currentmodule:: gamkit.eilers._lingam

gamkit.eilers._lingam
""""""""""""""""""""""""""
.. automodule:: gamkit.eilers._lingam
   :members:

.. literalinclude:: ../src/gamkit/eilers/_lingam.py
   :language: python
   :linenos:


.. currentmodule:: gamkit.eilers._gam

gamkit.eilers._gam
"""""""""""""""""""""""
.. automodule:: gamkit.eilers._gam
   :members:
   :exclude-members: _set_lams_from_combo, _assemble_penalty_from_bases,
                     _precompute_penalty_bases, _build_matrix_from_terms

.. literalinclude:: ../src/gamkit/eilers/_gam.py
   :language: python
   :linenos:


Wood Engine (``gamkit.wood``)
-----------------------------

Python implementation of Simon Wood's GAM methods (Wood 2011, 2017): multiple basis types,
null-space constraint absorption, and continuous REML / GCV smoothing optimization.

.. currentmodule:: gamkit.wood

Classes
^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LinGAM
   GAMCore

Modules
^^^^^^^

.. currentmodule:: gamkit.wood._smooths

gamkit.wood._smooths
""""""""""""""""""""
.. automodule:: gamkit.wood._smooths
   :members:

.. literalinclude:: ../src/gamkit/wood/_smooths.py
   :language: python
   :linenos:

.. currentmodule:: gamkit.wood._optimizer

gamkit.wood._optimizer
""""""""""""""""""""""
.. automodule:: gamkit.wood._optimizer
   :members:

.. literalinclude:: ../src/gamkit/wood/_optimizer.py
   :language: python
   :linenos:

.. currentmodule:: gamkit.wood._terms

gamkit.wood._terms
""""""""""""""""""
.. automodule:: gamkit.wood._terms
   :members:

.. literalinclude:: ../src/gamkit/wood/_terms.py
   :language: python
   :linenos:

.. currentmodule:: gamkit.wood._lingam

gamkit.wood._lingam
"""""""""""""""""""
.. automodule:: gamkit.wood._lingam
   :members:

.. literalinclude:: ../src/gamkit/wood/_lingam.py
   :language: python
   :linenos:


.. currentmodule:: gamkit.wood._gam

gamkit.wood._gam
""""""""""""""""
.. automodule:: gamkit.wood._gam
   :members:

.. literalinclude:: ../src/gamkit/wood/_gam.py
   :language: python
   :linenos:
