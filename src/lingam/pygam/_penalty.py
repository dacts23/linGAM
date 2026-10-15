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
