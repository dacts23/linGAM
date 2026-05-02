"""linGAM — Simple Fast Linear GAM.

A minimal implementation of Generalized Additive Models using
penalised B-splines. Supports single-term (``LinGAM``) and multi-term
with a formula interface (``GAMCore``).

Shape constraints
-----------------
Spline terms (``s()``) and ``LinGAM`` accept a ``constraint`` parameter:

- ``None``              — standard smoothness (second-difference penalty)
- ``'mono_inc'``        — encourages monotonically increasing trend
- ``'mono_dec'``        — encourages monotonically decreasing trend
- ``'convex'``          — encourages convexity (soft constraint)
- ``'concave'``         — encourages concavity (soft constraint)
- ``'periodic'``        — periodic B-spline with wrapped penalty

Example::

    model = GAMCore("s(0, constraint='mono_inc') + s(1, constraint='periodic')")
    lingam = LinGAM(n_splines=12, constraint='mono_inc')
"""

from ._gam import GAMCore
from ._lingam import LinGAM

__version__ = "0.2.0"
__all__ = ["GAMCore", "LinGAM", "__version__"]