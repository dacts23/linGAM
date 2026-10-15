"""linGAM — Simple Fast Linear GAM with PyGAM and R mgcv backends."""

from . import mgcv, pygam
from .mgcv import GAMCore as MgcvGAMCore, LinGAM as MgcvLinGAM
from .pygam import GAMCore as PygamGAMCore, LinGAM as PygamLinGAM

__version__ = "0.3.0"

_BACKENDS = {
    "pygam": (PygamLinGAM, PygamGAMCore),
    "mgcv": (MgcvLinGAM, MgcvGAMCore),
}


def _dispatch(idx: int, backend: str, *args, **kwargs):
    key = str(backend).lower()
    if key not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'pygam' or 'mgcv'.")
    return _BACKENDS[key][idx](*args, **kwargs)


class LinGAM:
    """Single-term GAM supporting both pyGAM and R mgcv backends."""

    def __new__(cls, *args, backend: str = "pygam", **kwargs):
        return _dispatch(0, backend, *args, **kwargs)


class GAMCore:
    """Multi-term GAM supporting both pyGAM and R mgcv backends."""

    def __new__(cls, *args, backend: str = "pygam", **kwargs):
        return _dispatch(1, backend, *args, **kwargs)


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
