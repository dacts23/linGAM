"""magli (Modelo Aditivo Generalizado Lineal): Pythonic Generalized Additive Models (GAMs) with Wood and Eilers backends."""

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
    """Single-term Generalized Additive Model dispatcher.

    Parameters
    ----------
    backend : {'wood', 'eilers'}, default='wood'
        Engine backend: ``'wood'`` for Simon Wood's REML/GCV spline engine
        (:class:`magli.wood.LinGAM`) or ``'eilers'`` for Eilers & Marx
        B-splines with discrete GCV grid search (:class:`magli.eilers.LinGAM`).
    *args, **kwargs
        Forwarded to the selected backend's ``LinGAM`` constructor.
    """

    def __new__(cls, *args, backend: str = "wood", **kwargs):
        return _dispatch(0, backend, *args, **kwargs)


LinGAM = GAM


class GAMCore:
    """Multi-term Generalized Additive Model formula dispatcher.

    Parameters
    ----------
    formula : str
        Additive model formula with ``s()``, ``te()``, ``f()``, and ``l()`` terms.
    backend : {'wood', 'eilers'}, default='wood'
        Engine backend: ``'wood'`` (:class:`magli.wood.GAMCore`) or
        ``'eilers'`` (:class:`magli.eilers.GAMCore`).
    *args, **kwargs
        Forwarded to the selected backend's ``GAMCore`` constructor.
    """

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

