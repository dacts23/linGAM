"""Wood (2011) GAM backend for gamkit."""

from ._gam import GAMCore
from ._lingam import LinGAM
from ._optimizer import fit_wood_engine
from ._smooths import (
    absorb_sum_to_zero_constraint,
    build_smooth_basis,
    predict_smooth_basis,
    rescale_penalty,
)
from ._terms import (
    _WoodFactorTerm,
    _WoodLinearTerm,
    _WoodSplineTerm,
    _WoodTensorTerm,
    instantiate_wood_terms,
)

GAM = LinGAM

__all__ = [
    "GAM",
    "LinGAM",
    "GAMCore",
    "fit_wood_engine",
    "build_smooth_basis",
    "predict_smooth_basis",
    "rescale_penalty",
    "absorb_sum_to_zero_constraint",
    "instantiate_wood_terms",
    "_WoodSplineTerm",
    "_WoodTensorTerm",
    "_WoodFactorTerm",
    "_WoodLinearTerm",
]

