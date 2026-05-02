"""Vectorized De Boor B-spline basis construction."""

import numpy as np
from numpy.typing import ArrayLike


def b_spline_basis(
    x_in: ArrayLike,
    n_splines: int,
    spline_order: int,
    edge_knots: np.ndarray,
    periodic: bool = False,
) -> np.ndarray:
    """Construct B-spline basis matrix using vectorized De Boor recursion.

    Parameters
    ----------
    x_in : array-like, shape (n,)
        Predictor values at which to evaluate the basis.
    n_splines : int
        Number of basis functions K.  For periodic splines this is the
        *full* count before wrapping; the returned matrix has
        ``K - spline_order + 1`` columns.
    spline_order : int
        Order k of the B-spline (polynomial degree).
    edge_knots : np.ndarray, shape (2,)
        Domain boundaries [min, max] for mapping data to [0, 1].
    periodic : bool, default False
        If True, wrap the first ``spline_order - 1`` basis functions
        to enforce f(0) = f(1), returning a periodic basis.

    Returns
    -------
    bases : np.ndarray, shape (n, K) or (n, K_eff) for periodic
        B-spline basis matrix with linear extrapolation beyond edge knots.
    """
    K = n_splines
    k = spline_order

    offset = edge_knots[0]
    scale = edge_knots[1] - edge_knots[0]
    if scale == 0.0:
        scale = 1.0

    boundary_knots = np.linspace(0.0, 1.0, 1 + K - k)
    diff = boundary_knots[1] - boundary_knots[0]

    x = (np.asarray(x_in).ravel() - offset) / scale
    x = np.r_[x, 0.0, 1.0]

    if periodic:
        x = x % 1.0
        x[-2] = 0.0
        x[-1] = 1.0

    x_extrap_l = x < 0.0
    x_extrap_r = x > 1.0
    x_interp = ~(x_extrap_l | x_extrap_r)
    x = np.atleast_2d(x).T

    aug = np.arange(1, k + 1) * diff
    aug_knots = np.r_[-aug[::-1], boundary_knots, 1.0 + aug]
    aug_knots[-1] += 1e-9

    bases = ((x >= aug_knots[:-1]) & (x < aug_knots[1:])).astype(float)
    bases[-1] = bases[-2][::-1]

    n_active = len(aug_knots) - 1
    prev_bases = None

    for m in range(2, k + 2):
        n_active -= 1
        num_l = (x - aug_knots[:n_active]) * bases[:, :n_active]
        denom_l = aug_knots[m - 1:n_active + m - 1] - aug_knots[:n_active]
        left = num_l / denom_l

        num_r = (aug_knots[m:n_active + m] - x) * bases[:, 1:n_active + 1]
        denom_r = aug_knots[m:n_active + m] - aug_knots[1:n_active + 1]
        right = num_r / denom_r

        prev_bases = bases[-2:].copy()
        bases = left + right

    if (np.any(x_extrap_l) or np.any(x_extrap_r)) and k > 0 and not periodic:
        bases[~x_interp] = 0.0
        denom_l = aug_knots[k:-1] - aug_knots[:-k - 1]
        left_g = prev_bases[:, :-1] / denom_l
        denom_r = aug_knots[k + 1:] - aug_knots[1:-k]
        right_g = prev_bases[:, 1:] / denom_r
        grads = k * (left_g - right_g)

        if np.any(x_extrap_l):
            bases[x_extrap_l] = grads[0] * x[x_extrap_l] + bases[-2]
        if np.any(x_extrap_r):
            bases[x_extrap_r] = grads[1] * (x[x_extrap_r] - 1.0) + bases[-1]

    result = bases[:-2]

    if periodic:
        n_wrap = k - 1
        if n_wrap > 0 and n_wrap < K:
            K_eff = K - n_wrap
            wrapped = result[:, :K_eff].copy()
            for j in range(n_wrap):
                wrapped[:, j] += result[:, K_eff + j]
            return wrapped

    return result