"""Penalized least-squares solver via SVD data augmentation."""

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
