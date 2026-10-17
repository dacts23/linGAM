"""Eilers & Marx (1996) P-spline GAM backend for magli."""

from ._gam import GAMCore
from ._lingam import LinGAM
from ._solver import solve_pirls
from ._terms import (
    _FactorTerm,
    _LinearTerm,
    _SplineTerm,
    _TensorTerm,
    _Term,
    instantiate_terms,
)

GAM = LinGAM

__all__ = [
    "GAM",
    "LinGAM",
    "GAMCore",
    "solve_pirls",
    "instantiate_terms",
    "_Term",
    "_SplineTerm",
    "_TensorTerm",
    "_FactorTerm",
    "_LinearTerm",
]

