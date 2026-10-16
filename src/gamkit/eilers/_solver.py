"""Penalized least-squares solver via SVD data augmentation."""

from typing import Optional, Tuple

import numpy as np


def solve_pirls(
    X: np.ndarray,
    y: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve penalized least squares: (X^T X + P) beta = X^T y."""
    n, m = X.shape
    eps = np.finfo(float).eps
    E = np.linalg.cholesky(np.diag(np.ones(m) * np.sqrt(eps)) + P).T
    Q, R = np.linalg.qr(X)
    min_nm = min(m, n)
    U, d, Vt = np.linalg.svd(np.vstack([R, E]), full_matrices=False)
    U1 = U[:min_nm, :min_nm]
    d_inv = 1.0 / d[:min_nm]
    B_solve = (Vt[:min_nm].T * d_inv) @ U1.T @ Q.T
    return B_solve @ y, U1, B_solve


def irls_solve(
    X: np.ndarray,
    y: np.ndarray,
    P: np.ndarray,
    n_iter: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Huber-weighted Iteratively Reweighted Least Squares."""
    residuals = y - np.median(y)
    w = np.ones(len(y))
    coef: Optional[np.ndarray] = None
    U1: Optional[np.ndarray] = None
    B_solve: Optional[np.ndarray] = None

    for _ in range(n_iter):
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale_est = mad / 0.6745 if mad > 1e-6 else 1e-6
        c = 1.345 * scale_est
        abs_r = np.abs(residuals)
        w = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-6))

        W_sqrt = np.sqrt(w)
        coef, U1, B_solve = solve_pirls(X * W_sqrt[:, None], y * W_sqrt, P)
        residuals = y - X @ coef

    return coef, U1, B_solve, w  # type: ignore[return-value]
