"""gamkit — Pythonic Generalized Additive Models (GAMs) with Wood and Eilers backends."""

from . import eilers, wood
from .eilers import GAMCore as EilersGAMCore, LinGAM as EilersLinGAM
from .wood import GAMCore as WoodGAMCore, LinGAM as WoodLinGAM

__version__ = "0.3.0"

_BACKENDS = {
    "wood": (WoodLinGAM, WoodGAMCore),
    "eilers": (EilersLinGAM, EilersGAMCore),
    "pygam": (EilersLinGAM, EilersGAMCore),
}


def _dispatch(idx: int, backend: str, *args, **kwargs):
    key = str(backend).lower()
    if key not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'wood' or 'eilers'.")
    return _BACKENDS[key][idx](*args, **kwargs)


class GAM:
    """Single-term GAM supporting Wood and Eilers backends."""

    def __new__(cls, *args, backend: str = "wood", **kwargs):
        return _dispatch(0, backend, *args, **kwargs)


LinGAM = GAM


class GAMCore:
    """Multi-term GAM supporting Wood and Eilers backends."""

    def __new__(cls, *args, backend: str = "wood", **kwargs):
        return _dispatch(1, backend, *args, **kwargs)


__all__ = [
    "GAM",
    "LinGAM",
    "GAMCore",
    "EilersLinGAM",
    "EilersGAMCore",
    "WoodLinGAM",
    "WoodGAMCore",
    "eilers",
    "wood",
    "__version__",
]

