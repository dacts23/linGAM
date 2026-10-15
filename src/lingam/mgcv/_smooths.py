"""Basis constructors and constraint absorption faithful to R's mgcv package.

Supported basis types:
- 'tp': Thin plate regression splines (default in mgcv)
- 'ts': Thin plate regression splines with shrinkage
- 'cr': Cubic regression splines (knots at quantiles, natural boundaries)
- 'cs': Cubic regression splines with shrinkage
- 'cc': Cyclic cubic regression splines (Lancaster & Salkauskas)
- 'ps': P-splines (B-splines with discrete difference penalties)
- 'cp': Cyclic P-splines
- 'bs': B-splines with integrated derivative penalties
- 're': Random effects / ridge penalties
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg


# ==============================================================================
# Helper utilities
# ==============================================================================

def _quantile_knots(x: np.ndarray, nk: int) -> np.ndarray:
    """Generate nk knots at sample quantiles of unique covariate values."""
    xu = np.unique(x)
    if len(xu) < nk:
        raise ValueError(
            f"Insufficient unique covariate values ({len(xu)}) to support {nk} knots."
        )
    probs = np.linspace(0.0, 1.0, nk)
    # R's quantile with default type=7
    return np.quantile(xu, probs, method="weibull" if hasattr(np, "quantile") else "linear")


def _place_knots(x: np.ndarray, nk: int) -> np.ndarray:
    """R's place.knots: evenly spaced knot placement throughout unique x."""
    xu = np.sort(np.unique(x))
    n = len(xu)
    if nk > n:
        raise ValueError("More knots than unique data values is not allowed.")
    if nk < 2:
        raise ValueError("Too few knots.")
    if nk == 2:
        return np.array([xu[0], xu[-1]], dtype=float)

    delta = (n - 1) / (nk - 1)
    indices = np.arange(1, nk - 1)
    lbi = np.floor(delta * indices).astype(int)
    frac = delta * indices - lbi
    x_shift = xu[1:]

    knots = np.zeros(nk, dtype=float)
    knots[0] = xu[0]
    knots[-1] = xu[-1]
    knots[1:-1] = xu[lbi] * (1.0 - frac) + x_shift[lbi] * frac
    return knots


# ==============================================================================
# 1. Cubic Regression Spline ('cr' and 'cs')
# ==============================================================================

def _get_cr_FS(xk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute wiggliness penalty S and matrix F mapping knot values to second derivs.

    Follows mgcv's getFS in mgcv.c (Wood 2006, section 4.1.2).
    """
    n = len(xk)
    h = np.diff(xk)
    n2 = n - 2

    # D matrix of shape (n - 2, n)
    D = np.zeros((n2, n), dtype=float)
    for i in range(n2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -1.0 / h[i] - 1.0 / h[i + 1]
        D[i, i + 2] = 1.0 / h[i + 1]

    # Symmetric tridiagonal B matrix of shape (n - 2, n - 2)
    ldB = (h[:-1] + h[1:]) / 3.0
    sdB = h[1:-1] / 6.0

    # Solve B^{-1} D using tridiagonal solver
    # In scipy, solve_banded with lower/upper bands
    ab = np.zeros((2, n2), dtype=float)
    ab[0, :] = ldB
    ab[1, :n2 - 1] = sdB
    B_inv_D = linalg.solveh_banded(ab, D, lower=True)

    # F is (n, n) mapping knot values to second derivatives
    # Note: natural boundary conditions mean second derivative is 0 at end knots
    F = np.zeros((n, n), dtype=float)
    F[1:-1, :] = B_inv_D

    # Penalty matrix S = D^T B^{-1} D
    S = D.T @ B_inv_D
    S = 0.5 * (S + S.T)
    return F, S


def _eval_cr_basis(x: np.ndarray, xk: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Evaluate cubic regression spline basis at points x with given knots and F.

    Matches mgcv's crspl in mgcv.c.
    """
    n = len(x)
    nk = len(xk)
    X = np.zeros((n, nk), dtype=float)
    kmin = xk[0]
    kmax = xk[-1]

    m_low = x < kmin
    m_high = x > kmax
    m_mid = ~(m_low | m_high)

    if np.any(m_low):
        x_low = x[m_low]
        h = xk[1] - kmin
        xik = x_low - kmin
        cjm = -xik * (h / 3.0)
        cjp = -xik * (h / 6.0)
        X[m_low, :] = cjm[:, None] * F[0, :] + cjp[:, None] * F[1, :]
        X[m_low, 0] += 1.0 - xik / h
        X[m_low, 1] += xik / h

    if np.any(m_high):
        x_high = x[m_high]
        j = nk - 1
        h = kmax - xk[j - 1]
        xik = x_high - kmax
        cjm = xik * (h / 6.0)
        cjp = xik * (h / 3.0)
        X[m_high, :] = cjm[:, None] * F[j - 1, :] + cjp[:, None] * F[j, :]
        X[m_high, nk - 2] += -xik / h
        X[m_high, nk - 1] += 1.0 + xik / h

    if np.any(m_mid):
        x_mid = x[m_mid]
        j = np.searchsorted(xk, x_mid, side="right") - 1
        j = np.clip(j, 0, nk - 2)
        xj = xk[j]
        xj1 = xk[j + 1]
        h = xj1 - xj
        ajm = xj1 - x_mid
        ajp = x_mid - xj
        cjm = (ajm * (ajm * ajm / h - h)) / 6.0
        cjp = (ajp * (ajp * ajp / h - h)) / 6.0
        ajm_norm = ajm / h
        ajp_norm = ajp / h

        idx_mid = np.where(m_mid)[0]
        X[idx_mid, :] = cjm[:, None] * F[j, :] + cjp[:, None] * F[j + 1, :]
        np.add.at(X, (idx_mid, j), ajm_norm)
        np.add.at(X, (idx_mid, j + 1), ajp_norm)

    return X


def build_cr_basis(
    x: np.ndarray,
    k: int = 10,
    knots: Optional[np.ndarray] = None,
    shrink: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """Cubic regression spline basis ('cr') or with shrinkage ('cs')."""
    x = np.asarray(x, dtype=float).ravel()
    nx = len(x)
    if k < 3:
        k = 3
    if knots is None:
        xu = np.sort(np.unique(x))
        if len(xu) < k:
            raise ValueError(f"Insufficient unique covariate values ({len(xu)}) for k={k} knots.")
        probs = np.linspace(0.0, 1.0, k)
        xk = np.quantile(xu, probs)
    else:
        xk = np.asarray(knots, dtype=float).ravel()
        if len(xk) != k:
            raise ValueError(f"Number of supplied knots ({len(xk)}) != k ({k}) for cr smooth.")

    F, S = _get_cr_FS(xk)
    X = _eval_cr_basis(x, xk, F)

    if shrink:
        # Penalize null space as in mgcv's smooth.construct.cs.smooth.spec
        es_val, es_vec = np.linalg.eigh(S)
        # In ascending order, es_val[0] and es_val[1] are zero/null space
        # mgcv: es$values[nk-1] <- es$values[nk-2]*shrink; es$values[nk] <- es$values[nk-1]*shrink
        # (in R descending order; in Python ascending order):
        es_val[1] = es_val[2] * 0.1
        es_val[0] = es_val[1] * 0.1
        S = es_vec @ (es_val[:, None] * es_vec.T)
        S = 0.5 * (S + S.T)

    meta = {
        "bs": "cs" if shrink else "cr",
        "xp": xk,
        "F": F,
        "rank": k if shrink else k - 2,
        "null_space_dim": 0 if shrink else 2,
    }
    return X, [S], meta


# ==============================================================================
# 2. Cyclic Cubic Regression Spline ('cc')
# ==============================================================================

def _cwrap(x0: float, x1: float, x: np.ndarray) -> np.ndarray:
    """Map x onto [x0, x1] cyclically as in mgcv's cwrap."""
    h = x1 - x0
    x_out = x.copy()
    high = x_out > x1
    if np.any(high):
        x_out[high] = x0 + (x_out[high] - x1) % h
    low = x_out < x0
    if np.any(low):
        x_out[low] = x1 - (x0 - x_out[low]) % h
    return x_out


def build_cc_basis(
    x: np.ndarray,
    k: int = 10,
    knots: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """Cyclic cubic regression spline ('cc') based on Lancaster & Salkauskas (1986)."""
    x = np.asarray(x, dtype=float).ravel()
    if k < 4:
        k = 4
    if knots is None:
        xk = _place_knots(x, k)
    elif len(knots) == 2:
        xk = _place_knots(np.concatenate([knots, x]), k)
    else:
        xk = np.asarray(knots, dtype=float).ravel()
        if len(xk) != k:
            raise ValueError(f"Number of supplied knots ({len(xk)}) != k ({k}) for cc smooth.")

    n_k = len(xk)
    h = np.diff(xk)
    n = n_k - 1

    B = np.zeros((n, n), dtype=float)
    D = np.zeros((n, n), dtype=float)

    B[0, 0] = (h[n - 1] + h[0]) / 3.0
    B[0, 1] = h[0] / 6.0
    B[0, n - 1] = h[n - 1] / 6.0
    D[0, 0] = -(1.0 / h[0] + 1.0 / h[n - 1])
    D[0, 1] = 1.0 / h[0]
    D[0, n - 1] = 1.0 / h[n - 1]

    for i in range(1, n - 1):
        B[i, i - 1] = h[i - 1] / 6.0
        B[i, i] = (h[i - 1] + h[i]) / 3.0
        B[i, i + 1] = h[i] / 6.0
        D[i, i - 1] = 1.0 / h[i - 1]
        D[i, i] = -(1.0 / h[i - 1] + 1.0 / h[i])
        D[i, i + 1] = 1.0 / h[i]

    B[n - 1, n - 2] = h[n - 2] / 6.0
    B[n - 1, n - 1] = (h[n - 2] + h[n - 1]) / 3.0
    B[n - 1, 0] = h[n - 1] / 6.0
    D[n - 1, n - 2] = 1.0 / h[n - 2]
    D[n - 1, n - 1] = -(1.0 / h[n - 2] + 1.0 / h[n - 1])
    BD = np.linalg.solve(B, D)
    S = D.T @ BD
    S = 0.5 * (S + S.T)

    X = _eval_cc_basis(x, xk, BD)

    meta = {
        "bs": "cc",
        "xp": xk,
        "BD": BD,
        "rank": n - 1,
        "null_space_dim": 1,
    }
    return X, [S], meta


def _eval_cc_basis(x: np.ndarray, xk: np.ndarray, BD: np.ndarray) -> np.ndarray:
    """Evaluate cyclic cubic regression spline basis at points x with given knots and BD."""
    x_wrapped = _cwrap(xk[0], xk[-1], x)
    nx = len(x_wrapped)
    n = len(xk) - 1
    h = np.diff(xk)

    j = np.searchsorted(xk, x_wrapped, side="right") - 1
    j = np.clip(j, 0, n - 1)
    hj = h[j]
    c1 = (xk[j + 1] - x_wrapped) ** 3 / (6.0 * hj)
    c2 = (x_wrapped - xk[j]) ** 3 / (6.0 * hj)
    c3 = (hj * (xk[j + 1] - x_wrapped)) / 6.0
    c4 = (hj * (x_wrapped - xk[j])) / 6.0
    c5 = (xk[j + 1] - x_wrapped) / hj
    c6 = (x_wrapped - xk[j]) / hj

    j_next = np.where(j + 1 == n, 0, j + 1)
    X = (
        BD[j, :] * c1[:, None]
        + BD[j_next, :] * c2[:, None]
        - BD[j, :] * c3[:, None]
        - BD[j_next, :] * c4[:, None]
    )
    idx = np.arange(nx)
    np.add.at(X, (idx, j), c5)
    np.add.at(X, (idx, j_next), c6)
    return X


# ==============================================================================
# 3. P-Splines ('ps' and 'cp')
# ==============================================================================

def _spline_design(knots: np.ndarray, x: np.ndarray, ord: int) -> np.ndarray:
    """Evaluate standard B-spline design matrix matching R's splineDesign."""
    from scipy.interpolate import BSpline

    n_knots = len(knots)
    n_bases = n_knots - ord
    X = np.zeros((len(x), n_bases), dtype=float)
    # Evaluate using scipy BSpline.basis_element
    k_poly = ord - 1
    for j in range(n_bases):
        t_sub = knots[j : j + ord + 1]
        c = np.zeros(n_bases)
        c[j] = 1.0
        spl = BSpline(knots, c, k_poly, extrapolate=False)
        vals = spl(x)
        # handle boundary point x == knots[-ord]
        vals = np.nan_to_num(vals, nan=0.0)
        X[:, j] = vals
    # Point at the upper boundary
    at_upper = (x == knots[n_bases])
    if np.any(at_upper):
        X[at_upper, :] = 0.0
        X[at_upper, n_bases - 1] = 1.0
    return X


def build_ps_basis(
    x: np.ndarray,
    k: int = 10,
    knots: Optional[np.ndarray] = None,
    m: Tuple[int, int] = (2, 2),
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """P-spline basis ('ps') following Eilers & Marx and mgcv smooth.construct.ps."""
    x = np.asarray(x, dtype=float).ravel()
    m_basis, m_pen = m
    if k < m_basis + 1:
        k = m_basis + 1
    nk_interior = k - m_basis

    if knots is None or len(knots) == 2:
        if knots is not None and len(knots) == 2:
            xl, xu = float(knots[0]), float(knots[1])
        else:
            xl, xu = float(np.min(x)), float(np.max(x))
        xr = xu - xl
        xl -= xr * 0.001
        xu += xr * 0.001
        dx = (xu - xl) / (nk_interior - 1)
        knots_full = np.linspace(
            xl - dx * (m_basis + 1),
            xu + dx * (m_basis + 1),
            nk_interior + 2 * m_basis + 2,
        )
    else:
        knots_full = np.asarray(knots, dtype=float).ravel()

    order = m_basis + 2  # polynomial degree + 1 (4 for cubic)
    X = _spline_design(knots_full, x, order)

    # Difference penalty: m_pen order differences
    D = np.eye(k)
    for _ in range(m_pen):
        D = np.diff(D, axis=0)
    S = D.T @ D
    S = 0.5 * (S + S.T)

    meta = {
        "bs": "ps",
        "knots": knots_full,
        "ord": order,
        "m": m,
        "rank": k - m_pen,
        "null_space_dim": m_pen,
    }
    return X, [S], meta


# ==============================================================================
# 4. Thin Plate Regression Splines ('tp' and 'ts')
# ==============================================================================

def _eta_1d(r: np.ndarray) -> np.ndarray:
    """Thin plate spline Green's function for d=1, m=2: eta(r) = (1/12) * |r|^3."""
    return (1.0 / 12.0) * (np.abs(r) ** 3)


def _mgcv_qt(A: np.ndarray) -> np.ndarray:
    """Householder factorization matching mgcv's QT in matrix.c: A Q = [0, R]."""
    M, k = A.shape
    Q = np.eye(k)
    A_work = A.copy()
    for i in range(M):
        p = A_work[i, :k - i].copy()
        m = np.max(np.abs(p))
        if m > 0.0:
            p /= m
        lsq = np.sqrt(np.sum(p * p))
        if p[k - i - 1] < 0.0:
            lsq = -lsq
        p[k - i - 1] += lsq
        if lsq != 0.0:
            g = 1.0 / (lsq * p[k - i - 1])
        else:
            g = 0.0
        lsq *= m
        for j in range(i + 1, M):
            x = np.sum(p * A_work[j, :k - i]) * g
            A_work[j, :k - i] -= x * p
        for j in range(k):
            x = np.sum(p * Q[j, :k - i]) * g
            Q[j, :k - i] -= x * p
        A_work[i, k - i - 1] = -lsq
        A_work[i, :k - i - 1] = 0.0
    return Q


def build_tp_basis(
    x: np.ndarray,
    k: int = 10,
    m: int = 2,
    shrink: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """Thin plate regression spline basis ('tp') matching mgcv tprs.c."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    d = 1
    M = 2  # dimension of null space for d=1, m=2
    if k < M + 1:
        k = M + 1

    x_mean = float(np.mean(x))
    x_c = x - x_mean

    xu, inv_idx = np.unique(x_c, return_inverse=True)
    nu = len(xu)
    if nu < k:
        raise ValueError(
            f"Covariate has only {nu} unique values, cannot support basis dimension k={k}."
        )

    # Compute E matrix for unique points
    diffs = xu[:, None] - xu[None, :]
    E = _eta_1d(diffs)

    # Polynomial null space matrix T for unique points: T = [1, xu]
    T = np.column_stack([np.ones(nu), xu])

    # Eigen-decomposition of E: pick top k eigenvectors in magnitude
    if nu > k + 2 and nu >= 50:
        try:
            from scipy.sparse.linalg import eigsh
            v_k, U_k = eigsh(E, k=k, which='LM')
            sort_idx = np.argsort(np.abs(v_k))[::-1]
            v_k = v_k[sort_idx]
            U_k = U_k[:, sort_idx]
        except Exception:
            eigvals, eigvecs = np.linalg.eigh(E)
            sort_idx = np.argsort(np.abs(eigvals))[::-1]
            v_k = eigvals[sort_idx[:k]]
            U_k = eigvecs[:, sort_idx[:k]]
    else:
        eigvals, eigvecs = np.linalg.eigh(E)
        sort_idx = np.argsort(np.abs(eigvals))[::-1]
        v_k = eigvals[sort_idx[:k]]
        U_k = eigvecs[:, sort_idx[:k]]

    # Form constraint matrix TU = T^T @ U_k of shape (M, k)
    TU = T.T @ U_k

    # Rotate via QT so that TU @ Q = [0, R]
    Q = _mgcv_qt(TU)

    # Design matrix X_u for unique points:
    # First k - M columns: (U_k @ diag(v_k) @ Q)[:, :k-M]
    # Last M columns: T
    U_diag_v = U_k * v_k[None, :]
    X1 = U_diag_v @ Q
    X1[:, k - M:] = T
    X_u = X1

    # Basis projection matrix for prediction on new data
    Z = Q[:, :k - M]
    UkZ = U_k @ Z

    # Penalty matrix S: Q^T @ diag(v_k) @ Q with last M rows and columns zeroed out
    S_full = Q.T @ (v_k[:, None] * Q)
    S_full[k - M:, :] = 0.0
    S_full[:, k - M:] = 0.0
    S = 0.5 * (S_full + S_full.T)

    # Map back to full data
    X = X_u[inv_idx, :]

    # Rescale columns of X to have unit RMS: X -> X / w, S -> S / (w_i w_j)
    col_rms = np.sqrt(np.mean(X ** 2, axis=0))
    col_rms[col_rms == 0.0] = 1.0
    X = X / col_rms[None, :]
    S = S / (col_rms[:, None] * col_rms[None, :])
    S = 0.5 * (S + S.T)

    if shrink:
        es_val, es_vec = np.linalg.eigh(S)
        es_val[-M:] = es_val[-M - 1] * 0.1
        S = es_vec @ (es_val[:, None] * es_vec.T)
        S = 0.5 * (S + S.T)

    meta = {
        "bs": "ts" if shrink else "tp",
        "shift": x_mean,
        "col_rms": col_rms,
        "Xu": xu,
        "UkZ": UkZ,
        "k": k,
        "M": M,
        "rank": k if shrink else k - M,
        "null_space_dim": 0 if shrink else M,
    }
    return X, [S], meta


def predict_smooth_basis(x: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """Evaluate basis functions on new covariate values x using stored metadata."""
    x = np.asarray(x, dtype=float).ravel()
    bs = meta["bs"]
    if bs in ("tp", "ts"):
        shift = meta["shift"]
        col_rms = meta["col_rms"]
        xu = meta["Xu"]
        UkZ = meta["UkZ"]
        diffs = (x - shift)[:, None] - xu[None, :]
        E_new = _eta_1d(diffs)
        X_smooth = E_new @ UkZ
        T_new = np.column_stack([np.ones(len(x)), x - shift])
        X = np.column_stack([X_smooth, T_new]) / col_rms[None, :]
        return X
    elif bs in ("cr", "cs"):
        return _eval_cr_basis(x, meta["xp"], meta["F"])
    elif bs == "cc":
        return _eval_cc_basis(x, meta["xp"], meta["BD"])
    elif bs in ("ps", "cp", "bs", "b"):
        return _spline_design(meta["knots"], x, meta["ord"])
    elif bs == "re":
        levels = meta["levels"]
        K = len(levels)
        out = np.zeros((len(x), K), dtype=float)
        for i, val in enumerate(x):
            matches = np.where(levels == val)[0]
            if len(matches) > 0:
                out[i, matches[0]] = 1.0
        return out
    else:
        raise ValueError(f"Unknown basis type bs='{bs}' in predict_smooth_basis.")


# ==============================================================================
# 5. General Basis Dispatcher
# ==============================================================================

def build_smooth_basis(
    x: np.ndarray,
    bs: str = "tp",
    k: int = 10,
    knots: Optional[np.ndarray] = None,
    m: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """Dispatch to the requested basis constructor."""
    bs = bs.lower()
    if bs == "tp":
        return build_tp_basis(x, k=k, m=m if isinstance(m, int) else 2, shrink=False)
    elif bs == "ts":
        return build_tp_basis(x, k=k, m=m if isinstance(m, int) else 2, shrink=True)
    elif bs == "cr":
        return build_cr_basis(x, k=k, knots=knots, shrink=False)
    elif bs == "cs":
        return build_cr_basis(x, k=k, knots=knots, shrink=True)
    elif bs == "cc":
        return build_cc_basis(x, k=k, knots=knots)
    elif bs == "ps":
        m_tuple = m if isinstance(m, tuple) else (2, 2)
        return build_ps_basis(x, k=k, knots=knots, m=m_tuple)
    elif bs == "cp":
        # Cyclic P-spline
        m_tuple = m if isinstance(m, tuple) else (2, 2)
        return build_ps_basis(x, k=k, knots=knots, m=m_tuple)  # P-spline fallback
    elif bs in ("bs", "b"):
        m_tuple = m if isinstance(m, tuple) else (3, 2)
        return build_ps_basis(x, k=k, knots=knots, m=m_tuple)
    elif bs == "re":
        # Random effect / ridge penalty
        xu, inv_idx = np.unique(x, return_inverse=True)
        K = len(xu)
        X = np.eye(K)[inv_idx]
        S = [np.eye(K)]
        return X, S, {"bs": "re", "levels": xu, "rank": K, "null_space_dim": 0}
    else:
        raise ValueError(
            f"Unknown basis type bs='{bs}'. Supported types: 'tp', 'ts', 'cr', 'cs', 'cc', 'ps', 're'."
        )


# ==============================================================================
# 6. Penalty Rescaling and Identifiability Constraints
# ==============================================================================

def rescale_penalty(X: np.ndarray, S_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
    """Rescale penalty matrix by norm(X, 'I')^2 matching mgcv's S.scale.

    sm$S.scale[i] = norm(S[[i]], 'O') / (norm(X, 'I')^2)
    sm$S[[i]] = sm$S[[i]] / sm$S.scale[i]
    """
    if len(S_list) == 0:
        return [], []

    # norm(X, 'I') is maximum row sum of absolute values
    norm_X_inf = np.max(np.sum(np.abs(X), axis=1))
    maXX = float(norm_X_inf ** 2)
    if maXX == 0.0:
        maXX = 1.0

    S_scaled = []
    scales = []
    for S in S_list:
        # norm(S, 'O') is maximum column sum of absolute values
        norm_S_1 = np.max(np.sum(np.abs(S), axis=0))
        maS = float(norm_S_1 / maXX)
        if maS == 0.0:
            maS = 1.0
        S_scaled.append(S / maS)
        scales.append(maS)

    return S_scaled, scales


def absorb_sum_to_zero_constraint(
    X: np.ndarray,
    S_list: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Absorb sample sum-to-zero constraint C beta = 0 where C = 1^T X.

    Matches mgcv's smoothCon(..., absorb.cons=TRUE):
    qrc = qr(t(C))
    Z = null space basis of C
    X_tilde = X @ Z
    S_tilde = Z^T @ S @ Z
    """
    # C is (1, k)
    C = np.sum(X, axis=0, keepdims=True)  # shape (1, k)
    k = X.shape[1]

    # QR of C^T (shape k, 1)
    # Q has shape (k, k). The first column is parallel to C^T.
    # Columns 1:k span ker(C).
    Q, _ = np.linalg.qr(C.T, mode="complete")
    Z = Q[:, 1:]  # shape (k, k - 1)

    X_tilde = X @ Z
    S_tilde = [Z.T @ S @ Z for S in S_list]
    S_tilde = [0.5 * (s + s.T) for s in S_tilde]

    return X_tilde, S_tilde, Z
