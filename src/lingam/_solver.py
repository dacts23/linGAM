"""Penalized least-squares solvers via SVD data augmentation and IRLS."""

from typing import Optional, Tuple

import numpy as np


def solve_pirls(
    X: np.ndarray,
    y: np.ndarray,
    P: np.ndarray,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve penalized least squares: (X^T X + P) beta = X^T y.

    Uses SVD data-augmentation: stacks QR-compressed X on top of Cholesky(P)
    to avoid squaring the condition number.

    Returns (coef, U1, B_solve) where B_solve is the solution operator for
    covariance and EDF computation.
    """
    n, m = X.shape
    eps = np.finfo(float).eps

    S_ridge = np.diag(np.ones(m) * np.sqrt(eps))
    E = np.linalg.cholesky(S_ridge + P).T

    if Q is None or R is None:
        Q, R_ = np.linalg.qr(X)
    else:
        R_ = R

    min_nm = min(m, n)
    U, d, Vt = np.linalg.svd(np.vstack([R_, E]), full_matrices=False)

    U1 = U[:min_nm, :min_nm]
    Vt = Vt[:min_nm]
    d_inv = 1.0 / d[:min_nm]

    B_solve = (Vt.T * d_inv) @ U1.T @ Q.T
    coef = B_solve @ y

    return coef, U1, B_solve


def irls_solve(
    X: np.ndarray,
    y: np.ndarray,
    P: np.ndarray,
    n_iter: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Huber-weighted Iteratively Reweighted Least Squares.

    Iteratively down-weights outliers using the Huber loss function.
    Returns (coef, U1, B_solve, weights, residuals).
    """
    n = len(y)
    med_y = np.median(y)
    residuals = y - med_y
    w = np.ones(n)
    coef: Optional[np.ndarray] = None
    U1: Optional[np.ndarray] = None
    B_solve: Optional[np.ndarray] = None

    for _ in range(n_iter):
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale_est = mad / 0.6745 if mad > 1e-6 else 1e-6
        c = 1.345 * scale_est
        abs_r = np.abs(residuals)
        w = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-6))

        W_sqrt = np.sqrt(w).reshape(-1, 1)
        X_w = X * W_sqrt
        y_w = y * np.sqrt(w)
        Qw, Rw = np.linalg.qr(X_w)
        coef, U1, B_solve = solve_pirls(X_w, y_w, P, Q=Qw, R=Rw)
        residuals = y - X @ coef

    return coef, U1, B_solve, w, residuals  # type: ignore[return-value]
