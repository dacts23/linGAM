API Reference
=============

This page documents the public API and internal engines in linGAM.

.. currentmodule:: lingam

Public Unified API
------------------

The top-level :class:`LinGAM` and :class:`GAMCore` classes dynamically dispatch
to either the pyGAM or R mgcv engine based on the ``backend`` argument.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMCore
   LinGAM


Shared Architecture
-------------------

.. currentmodule:: lingam._base

lingam._base
^^^^^^^^^^^^

Shared ``_GAMBase`` lifecycle inherited by both engines: prediction, partial
dependence, confidence/prediction intervals, information criteria, and all
decomposition/diagnostic plotting.  (``scale`` is σ for pygam, σ² for mgcv.)

.. automodule:: lingam._base
    :members:

.. currentmodule:: lingam._formula

lingam._formula
^^^^^^^^^^^^^^^

Shared additive model formula parser for ``s()``, ``te()``, ``f()``, and ``l()`` terms.

.. automodule:: lingam._formula
   :members:

.. literalinclude:: ../src/lingam/_formula.py
   :language: python
   :linenos:


pyGAM Engine (``lingam.pygam``)
-------------------------------

B-spline additive models with discrete GCV grid search and soft shape constraints.

.. currentmodule:: lingam.pygam

Classes
^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LinGAM
   GAMCore

Modules
^^^^^^^

.. currentmodule:: lingam.pygam._bspline

lingam.pygam._bspline
"""""""""""""""""""""
.. automodule:: lingam.pygam._bspline
   :members:

.. literalinclude:: ../src/lingam/pygam/_bspline.py
   :language: python
   :linenos:

.. currentmodule:: lingam.pygam._penalty

lingam.pygam._penalty
"""""""""""""""""""""
.. automodule:: lingam.pygam._penalty
   :members:

.. literalinclude:: ../src/lingam/pygam/_penalty.py
   :language: python
   :linenos:

.. currentmodule:: lingam.pygam._solver

lingam.pygam._solver
""""""""""""""""""""
.. automodule:: lingam.pygam._solver
   :members:

.. literalinclude:: ../src/lingam/pygam/_solver.py
   :language: python
   :linenos:

.. currentmodule:: lingam.pygam._terms

lingam.pygam._terms
"""""""""""""""""""
.. automodule:: lingam.pygam._terms
   :members:

.. literalinclude:: ../src/lingam/pygam/_terms.py
   :language: python
   :linenos:

.. currentmodule:: lingam.pygam._lingam

lingam.pygam._lingam
""""""""""""""""""""
.. automodule:: lingam.pygam._lingam
   :members:

.. literalinclude:: ../src/lingam/pygam/_lingam.py
   :language: python
   :linenos:

.. currentmodule:: lingam.pygam._gam

lingam.pygam._gam
"""""""""""""""""
.. automodule:: lingam.pygam._gam
   :members:
   :exclude-members: _set_lams_from_combo, _assemble_penalty_from_bases,
                     _precompute_penalty_bases, _build_matrix_from_terms

.. literalinclude:: ../src/lingam/pygam/_gam.py
   :language: python
   :linenos:


R mgcv Engine (``lingam.mgcv``)
-------------------------------

Exact Python implementation of Wood's R ``mgcv`` package: multiple basis types,
null-space constraint absorption, and outer REML / GCV smoothing optimization.

.. currentmodule:: lingam.mgcv

Classes
^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LinGAM
   GAMCore

Modules
^^^^^^^

.. currentmodule:: lingam.mgcv._smooths

lingam.mgcv._smooths
""""""""""""""""""""
.. automodule:: lingam.mgcv._smooths
   :members:

.. literalinclude:: ../src/lingam/mgcv/_smooths.py
   :language: python
   :linenos:

.. currentmodule:: lingam.mgcv._optimizer

lingam.mgcv._optimizer
""""""""""""""""""""""
.. automodule:: lingam.mgcv._optimizer
   :members:

.. literalinclude:: ../src/lingam/mgcv/_optimizer.py
   :language: python
   :linenos:

.. currentmodule:: lingam.mgcv._terms

lingam.mgcv._terms
""""""""""""""""""
.. automodule:: lingam.mgcv._terms
   :members:

.. literalinclude:: ../src/lingam/mgcv/_terms.py
   :language: python
   :linenos:

.. currentmodule:: lingam.mgcv._lingam

lingam.mgcv._lingam
"""""""""""""""""""
.. automodule:: lingam.mgcv._lingam
   :members:

.. literalinclude:: ../src/lingam/mgcv/_lingam.py
   :language: python
   :linenos:

.. currentmodule:: lingam.mgcv._gam

lingam.mgcv._gam
""""""""""""""""
.. automodule:: lingam.mgcv._gam
   :members:

.. literalinclude:: ../src/lingam/mgcv/_gam.py
   :language: python
   :linenos:
