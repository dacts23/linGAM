"""Eilers & Marx (1996) P-spline GAM backend for gamkit."""

from ._gam import GAMCore
from ._lingam import LinGAM
from ._solver import irls_solve, solve_pirls
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
    "irls_solve",
    "instantiate_terms",
    "_Term",
    "_SplineTerm",
    "_TensorTerm",
    "_FactorTerm",
    "_LinearTerm",
]

