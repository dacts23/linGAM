"""Penalty matrix builders for GAM regularisation."""

import numpy as np


def second_diff_penalty(k: int, lam: float) -> np.ndarray:
    """Second-difference penalty D2^T D2 scaled by lam."""
    D2 = np.diff(np.eye(k), n=2, axis=0)
    return lam * (D2.T @ D2)


def first_diff_penalty(k: int, lam: float) -> np.ndarray:
    """First-difference penalty D1^T D1 scaled by lam."""
    D1 = np.diff(np.eye(k), n=1, axis=0)
    return lam * (D1.T @ D1)


def circular_second_diff_penalty(k: int, lam: float) -> np.ndarray:
    """Circular second-difference penalty for periodic splines."""
    I = np.eye(k)
    D2 = np.roll(I, -1, axis=1) - 2.0 * I + np.roll(I, 1, axis=1)
    return lam * (D2.T @ D2)


def ridge_penalty(k: int, lam: float) -> np.ndarray:
    """L2 ridge penalty lam * I_k."""
    return lam * np.eye(k)


def shape_constraint_penalty(
    coef: np.ndarray,
    constraint: str,
    lam_con: float = 1e9,
    l2_con: float = 1e-3,
) -> np.ndarray:
    """Asymmetric active-set penalty for monotonicity and convexity constraints."""
    k = len(coef)
    if k <= 1:
        return np.zeros((k, k))
    if constraint in ("mono_inc", "mono_dec"):
        D = np.diff(np.eye(k), n=1, axis=0)
        d1 = np.diff(coef)
        mask = (d1 < 0) if constraint == "mono_inc" else (d1 > 0)
    elif constraint in ("convex", "concave"):
        if k <= 2:
            return np.zeros((k, k))
        D = np.diff(np.eye(k), n=2, axis=0)
        d2 = np.diff(coef, n=2)
        mask = (d2 < 0) if constraint == "convex" else (d2 > 0)
    else:
        return np.zeros((k, k))

    if not np.any(mask):
        return np.zeros((k, k))
    Dm = D[mask]
    C = lam_con * (Dm.T @ Dm)
    C.flat[:: k + 1] += l2_con
    return C

