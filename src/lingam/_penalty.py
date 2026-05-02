"""Penalty matrix builders for GAM regularisation."""

import numpy as np


def second_diff_penalty(k: int, lam: float) -> np.ndarray:
    """Second-difference penalty D2^T D2 scaled by lam.

    Parameters
    ----------
    k : int
        Number of basis functions (penalty size).
    lam : float
        Smoothing parameter.

    Returns
    -------
    np.ndarray, shape (k, k)
    """
    D2 = np.zeros((k - 2, k))
    idx = np.arange(k - 2)
    D2[idx, idx] = 1.0
    D2[idx, idx + 1] = -2.0
    D2[idx, idx + 2] = 1.0
    return lam * (D2.T @ D2)


def first_diff_penalty(k: int, lam: float) -> np.ndarray:
    """First-difference penalty D1^T D1 scaled by lam.

    Encourages constant-slope (monotonic) behaviour by penalising
    variation in adjacent coefficients.

    Parameters
    ----------
    k : int
        Number of basis functions (penalty size).
    lam : float
        Smoothing parameter.

    Returns
    -------
    np.ndarray, shape (k, k)
    """
    D1 = np.zeros((k - 1, k))
    idx = np.arange(k - 1)
    D1[idx, idx] = -1.0
    D1[idx, idx + 1] = 1.0
    return lam * (D1.T @ D1)


def circular_second_diff_penalty(k: int, lam: float) -> np.ndarray:
    """Circular second-difference penalty for periodic splines.

    Like ``second_diff_penalty`` but wraps at the boundaries so that
    the first and last coefficients are connected, enforcing periodicity.

    Parameters
    ----------
    k : int
        Number of basis functions (penalty size).
    lam : float
        Smoothing parameter.

    Returns
    -------
    np.ndarray, shape (k, k)
    """
    D2 = np.zeros((k, k))
    for i in range(k):
        D2[i, i] = -2
        D2[i, (i + 1) % k] += 1
        D2[i, (i - 1) % k] += 1
    return lam * (D2.T @ D2)


def ridge_penalty(k: int, lam: float) -> np.ndarray:
    """L2 ridge penalty lam * I_k.

    Parameters
    ----------
    k : int
        Number of coefficients (penalty size).
    lam : float
        Ridge strength.

    Returns
    -------
    np.ndarray, shape (k, k)
    """
    return lam * np.eye(k)