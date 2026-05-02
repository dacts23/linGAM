API Reference
=============

This page documents every public class, method, and function in linGAM.
Source code for each module is shown at the bottom of its section.

.. currentmodule:: lingam

Public classes
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMCore
   LinGAM


Internal modules
----------------

These modules are private (single leading underscore) but are documented here
for contributors and advanced users.

.. currentmodule:: lingam._bspline

lingam._bspline
^^^^^^^^^^^^^^^

.. automodule:: lingam._bspline
   :members:

.. literalinclude:: ../src/lingam/_bspline.py
   :language: python
   :linenos:


.. currentmodule:: lingam._formula

lingam._formula
^^^^^^^^^^^^^^^

.. automodule:: lingam._formula
   :members:

.. literalinclude:: ../src/lingam/_formula.py
   :language: python
   :linenos:


.. currentmodule:: lingam._gam

lingam._gam
^^^^^^^^^^^

.. automodule:: lingam._gam
   :members:
   :exclude-members: _set_lams_from_combo, _assemble_penalty_from_bases,
                     _precompute_penalty_bases, _build_matrix_from_terms

.. literalinclude:: ../src/lingam/_gam.py
   :language: python
   :linenos:


.. currentmodule:: lingam._lingam

lingam._lingam
^^^^^^^^^^^^^^

.. automodule:: lingam._lingam
   :members:

.. literalinclude:: ../src/lingam/_lingam.py
   :language: python
   :linenos:


.. currentmodule:: lingam._penalty

lingam._penalty
^^^^^^^^^^^^^^^

.. automodule:: lingam._penalty
   :members:

.. literalinclude:: ../src/lingam/_penalty.py
   :language: python
   :linenos:


.. currentmodule:: lingam._solver

lingam._solver
^^^^^^^^^^^^^^

.. automodule:: lingam._solver
   :members:

.. literalinclude:: ../src/lingam/_solver.py
   :language: python
   :linenos:


.. currentmodule:: lingam._terms

lingam._terms
^^^^^^^^^^^^^

.. automodule:: lingam._terms
   :members:
   :exclude-members: _khatri_rao, _kron_sum

.. literalinclude:: ../src/lingam/_terms.py
   :language: python
   :linenos:
