"""R mgcv-compatible GAM backend for linGAM."""

from ._gam import GAMCore
from ._lingam import LinGAM
from ._optimizer import fit_mgcv_engine
from ._smooths import (
    absorb_sum_to_zero_constraint,
    build_smooth_basis,
    predict_smooth_basis,
    rescale_penalty,
)
from ._terms import (
    _MgcvFactorTerm,
    _MgcvLinearTerm,
    _MgcvSplineTerm,
    _MgcvTensorTerm,
    instantiate_mgcv_terms,
)

__all__ = [
    "LinGAM",
    "GAMCore",
    "fit_mgcv_engine",
    "build_smooth_basis",
    "predict_smooth_basis",
    "rescale_penalty",
    "absorb_sum_to_zero_constraint",
    "instantiate_mgcv_terms",
    "_MgcvSplineTerm",
    "_MgcvTensorTerm",
    "_MgcvFactorTerm",
    "_MgcvLinearTerm",
]
