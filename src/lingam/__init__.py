"""linGAM — Simple Fast Linear GAM with PyGAM and R mgcv backends."""

from . import mgcv, pygam
from .mgcv import (
    GAMCore as MgcvGAMCore,
    LinGAM as MgcvLinGAM,
)
from .pygam import (
    GAMCore as PygamGAMCore,
    LinGAM as PygamLinGAM,
)

__version__ = "0.3.0"


class LinGAM:
    """Single-term GAM supporting both pyGAM and R mgcv backends.

    Parameters
    ----------
    backend : {'pygam', 'mgcv'}, default 'pygam'
        Which engine to use:

        - 'pygam': Fast B-splines with GCV grid search.
        - 'mgcv': Exact R mgcv implementation with REML (or GCV) smoothing
          parameter optimization, multiple basis types ('tp', 'cr',
          'ps', 'cc', etc.), and exact penalty rescaling.
    """

    def __new__(cls, *args, backend: str = "pygam", **kwargs):
        backend = str(backend).lower()
        if backend == "mgcv":
            return MgcvLinGAM(*args, **kwargs)
        elif backend == "pygam":
            return PygamLinGAM(*args, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'pygam' or 'mgcv'.")


class GAMCore:
    """Multi-term GAM supporting both pyGAM and R mgcv backends.

    Parameters
    ----------
    formula : str
        Model specification.
    backend : {'pygam', 'mgcv'}, default 'pygam'
        Which engine to use ('pygam' or 'mgcv').
    """

    def __new__(cls, *args, backend: str = "pygam", **kwargs):
        backend = str(backend).lower()
        if backend == "mgcv":
            return MgcvGAMCore(*args, **kwargs)
        elif backend == "pygam":
            return PygamGAMCore(*args, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'pygam' or 'mgcv'.")


__all__ = [
    "LinGAM",
    "GAMCore",
    "PygamLinGAM",
    "PygamGAMCore",
    "MgcvLinGAM",
    "MgcvGAMCore",
    "pygam",
    "mgcv",
    "__version__",
]