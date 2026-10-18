"""Basis constructors and constraint absorption following Wood (2011, 2017).

Supported basis types:
- 'tp': Thin plate regression splines (default in Wood backend)
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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline


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

    knots = np.zeros(nk, dtype=float)
    knots[0] = xu[0]
    knots[-1] = xu[-1]
    knots[1:-1] = xu[lbi] * (1.0 - frac) + xu[1:][lbi] * frac
    return knots


def _get_cr_FS(xk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute wiggliness penalty S and matrix F mapping knot values to second derivs."""
    n = len(xk)
    h = np.diff(xk)
    n2 = n - 2

    D = np.zeros((n2, n), dtype=float)
    idx = np.arange(n2)
    D[idx, idx] = 1.0 / h[:-1]
    D[idx, idx + 1] = -1.0 / h[:-1] - 1.0 / h[1:]
    D[idx, idx + 2] = 1.0 / h[1:]

    ab = np.zeros((2, n2), dtype=float)
    ab[0, :] = (h[:-1] + h[1:]) / 3.0
    ab[1, : n2 - 1] = h[1:-1] / 6.0
    B_inv_D = linalg.solveh_banded(ab, D, lower=True)

    F = np.zeros((n, n), dtype=float)
    F[1:-1, :] = B_inv_D

    S = D.T @ B_inv_D
    return F, 0.5 * (S + S.T)


def _eval_cr_basis(x: np.ndarray, xk: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Evaluate cubic regression spline basis at points x with given knots and F."""
    n = len(x)
    nk = len(xk)
    X = np.zeros((n, nk), dtype=float)
    kmin, kmax = xk[0], xk[-1]

    m_low = x < kmin
    m_high = x > kmax
    m_mid = ~(m_low | m_high)

    if np.any(m_low):
        h = xk[1] - kmin
        xik = x[m_low] - kmin
        X[m_low, :] = (-xik * (h / 3.0))[:, None] * F[0, :] + (
            -xik * (h / 6.0)
        )[:, None] * F[1, :]
        X[m_low, 0] += 1.0 - xik / h
        X[m_low, 1] += xik / h

    if np.any(m_high):
        j = nk - 1
        h = kmax - xk[j - 1]
        xik = x[m_high] - kmax
        X[m_high, :] = (xik * (h / 6.0))[:, None] * F[j - 1, :] + (
            xik * (h / 3.0)
        )[:, None] * F[j, :]
        X[m_high, nk - 2] += -xik / h
        X[m_high, nk - 1] += 1.0 + xik / h

    if np.any(m_mid):
        x_mid = x[m_mid]
        j = np.clip(np.searchsorted(xk, x_mid, side="right") - 1, 0, nk - 2)
        xj, xj1 = xk[j], xk[j + 1]
        h = xj1 - xj
        ajm, ajp = xj1 - x_mid, x_mid - xj
        cjm = (ajm * (ajm * ajm / h - h)) / 6.0
        cjp = (ajp * (ajp * ajp / h - h)) / 6.0

        idx_mid = np.where(m_mid)[0]
        X[idx_mid, :] = cjm[:, None] * F[j, :] + cjp[:, None] * F[j + 1, :]
        np.add.at(X, (idx_mid, j), ajm / h)
        np.add.at(X, (idx_mid, j + 1), ajp / h)

    return X


def build_cr_basis(
    x: np.ndarray,
    k: int = 10,
    knots: Optional[np.ndarray] = None,
    shrink: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """Cubic regression spline basis ('cr') or with shrinkage ('cs')."""
    x = np.asarray(x, dtype=float).ravel()
    k = max(k, 3)
    if knots is None:
        xu = np.sort(np.unique(x))
        if len(xu) < k:
            raise ValueError(
                f"Insufficient unique covariate values ({len(xu)}) for k={k} knots."
            )
        xk = np.quantile(xu, np.linspace(0.0, 1.0, k))
    else:
        xk = np.asarray(knots, dtype=float).ravel()
        if len(xk) != k:
            raise ValueError(
                f"Number of supplied knots ({len(xk)}) != k ({k}) for cr smooth."
            )

    F, S = _get_cr_FS(xk)
    X = _eval_cr_basis(x, xk, F)

    if shrink:
        es_val, es_vec = np.linalg.eigh(S)
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


def _cwrap(x0: float, x1: float, x: np.ndarray) -> np.ndarray:
    """Map x onto [x0, x1] cyclically."""
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
    k = max(k, 4)
    if knots is None:
        xk = _place_knots(x, k)
    elif len(knots) == 2:
        xk = _place_knots(np.concatenate([knots, x]), k)
    else:
        xk = np.asarray(knots, dtype=float).ravel()
        if len(xk) != k:
            raise ValueError(
                f"Number of supplied knots ({len(xk)}) != k ({k}) for cc smooth."
            )

    h = np.diff(xk)
    n = len(xk) - 1
    idx = np.arange(n)
    prev_i = (idx - 1) % n
    next_i = (idx + 1) % n

    B = np.zeros((n, n), dtype=float)
    D = np.zeros((n, n), dtype=float)
    B[idx, prev_i] = h[prev_i] / 6.0
    B[idx, idx] = (h[prev_i] + h[idx]) / 3.0
    B[idx, next_i] = h[idx] / 6.0
    D[idx, prev_i] = 1.0 / h[prev_i]
    D[idx, idx] = -(1.0 / h[prev_i] + 1.0 / h[idx])
    D[idx, next_i] = 1.0 / h[idx]

    BD = np.linalg.solve(B, D)
    S = D.T @ BD
    S = 0.5 * (S + S.T)
    X = _eval_cc_basis(x, xk, BD)

    return X, [S], {"bs": "cc", "xp": xk, "BD": BD, "rank": n - 1, "null_space_dim": 1}


def _eval_cc_basis(x: np.ndarray, xk: np.ndarray, BD: np.ndarray) -> np.ndarray:
    """Evaluate cyclic cubic regression spline basis at points x with given knots and BD."""
    x_wrapped = _cwrap(xk[0], xk[-1], x)
    nx = len(x_wrapped)
    n = len(xk) - 1
    h = np.diff(xk)

    j = np.clip(np.searchsorted(xk, x_wrapped, side="right") - 1, 0, n - 1)
    hj = h[j]
    c1 = (xk[j + 1] - x_wrapped) ** 3 / (6.0 * hj)
    c2 = (x_wrapped - xk[j]) ** 3 / (6.0 * hj)
    c3 = (hj * (xk[j + 1] - x_wrapped)) / 6.0
    c4 = (hj * (x_wrapped - xk[j])) / 6.0
    c5 = (xk[j + 1] - x_wrapped) / hj
    c6 = (x_wrapped - xk[j]) / hj

    j_next = np.where(j + 1 == n, 0, j + 1)
    X = BD[j, :] * (c1 - c3)[:, None] + BD[j_next, :] * (c2 - c4)[:, None]
    idx = np.arange(nx)
    np.add.at(X, (idx, j), c5)
    np.add.at(X, (idx, j_next), c6)
    return X


def _spline_design(
    knots: np.ndarray, x: np.ndarray, ord: int, outer_ok: bool = False
) -> np.ndarray:
    """Evaluate standard B-spline design matrix."""
    n_bases = len(knots) - ord
    k_deg = ord - 1
    k_pad = np.concatenate(
        [
            np.repeat(knots[0] - 1.0, k_deg),
            knots,
            np.repeat(knots[-1] + 1.0, k_deg),
        ]
    )
    c = np.zeros((len(k_pad) - ord, n_bases), dtype=float)
    c[k_deg : k_deg + n_bases, :] = np.eye(n_bases)
    out = BSpline(k_pad, c, k_deg, extrapolate=False)(x)
    np.nan_to_num(out, copy=False, nan=0.0)
    return out


def _c_spline_des(x: np.ndarray, knots: np.ndarray, ord: int = 4) -> np.ndarray:
    """Cyclic B-spline basis."""
    knots = np.sort(np.asarray(knots, dtype=float).ravel())
    nk = len(knots)
    k1, kmax = knots[0], knots[-1]
    x_w = _cwrap(k1, kmax, x)
    xc = knots[nk - ord]
    knots_ext = np.concatenate([k1 - (kmax - knots[nk - ord : nk - 1]), knots])
    X = _spline_design(knots_ext, x_w, ord, outer_ok=True)
    ind = x_w > xc
    if np.any(ind):
        X[ind] += _spline_design(
            knots_ext, x_w[ind] - kmax + k1, ord, outer_ok=True
        )
    return X


def build_ps_basis(
    x: np.ndarray,
    k: int = 10,
    knots: Optional[np.ndarray] = None,
    m: Tuple[int, int] = (2, 2),
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """P-spline basis ('ps') following Eilers & Marx and Wood (2017)."""
    x = np.asarray(x, dtype=float).ravel()
    m_basis, m_pen = m
    k = max(k, m_basis + 1)
    nk_interior = k - m_basis

    if knots is None or len(knots) == 2:
        xl, xu = (
            (float(knots[0]), float(knots[1]))
            if knots is not None and len(knots) == 2
            else (float(np.min(x)), float(np.max(x)))
        )
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

    order = m_basis + 2
    X = _spline_design(knots_full, x, order, outer_ok=True)
    D = np.diff(np.eye(k), n=m_pen, axis=0)
    S = D.T @ D
    S = 0.5 * (S + S.T)

    return (
        X,
        [S],
        {
            "bs": "ps",
            "knots": knots_full,
            "ord": order,
            "m": m,
            "rank": k - m_pen,
            "null_space_dim": m_pen,
        },
    )


def build_cp_basis(
    x: np.ndarray,
    k: int = 10,
    knots: Optional[np.ndarray] = None,
    m: Tuple[int, int] = (2, 2),
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """Cyclic P-spline basis ('cp')."""
    x = np.asarray(x, dtype=float).ravel()
    m_basis, m_pen = m
    k = max(k, m_basis, 4)
    nk = k + 1

    if knots is None or len(knots) == 2:
        x0, x1 = (
            (float(np.min(knots)), float(np.max(knots)))
            if knots is not None and len(knots) == 2
            else (float(np.min(x)), float(np.max(x)))
        )
        knots_seq = np.linspace(x0, x1, nk)
    else:
        knots_seq = np.asarray(knots, dtype=float).ravel()
        if len(knots_seq) != nk:
            raise ValueError(f"there should be {nk} supplied knots")

    order = m_basis + 2
    X = _c_spline_des(x, knots_seq, ord=order)

    if m_pen > 0:
        De = np.diff(np.eye(k + m_pen), n=m_pen, axis=0)
        D = De[:, m_pen:].copy()
        D[:, -m_pen:] += De[:, :m_pen]
    else:
        D = np.eye(k)
    S = D.T @ D
    S = 0.5 * (S + S.T)

    return (
        X,
        [S],
        {
            "bs": "cp",
            "knots": knots_seq,
            "ord": order,
            "m": m,
            "rank": k - 1,
            "null_space_dim": 1,
        },
    )


def build_bs_basis(
    x: np.ndarray,
    k: int = 10,
    knots: Optional[np.ndarray] = None,
    m: Tuple[int, int] = (3, 2),
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """B-spline basis ('bs') with integrated squared derivative penalty."""
    x = np.asarray(x, dtype=float).ravel()
    m1, m2 = m
    k = max(k, m1 + 1)
    nk = k - m1 + 1

    if knots is None or len(knots) == 2:
        xl, xu = (
            (float(knots[0]), float(knots[1]))
            if knots is not None and len(knots) == 2
            else (float(np.min(x)), float(np.max(x)))
        )
        xr = xu - xl
        xl -= xr * 0.001
        xu += xr * 0.001
        dx = (xu - xl) / (nk - 1)
        knots_full = np.linspace(xl - dx * m1, xu + dx * m1, nk + 2 * m1)
    else:
        knots_full = np.asarray(knots, dtype=float).ravel()

    order = m1 + 1
    X = _spline_design(knots_full, x, order, outer_ok=True)

    k0 = knots_full[m1 : m1 + nk]
    h = np.diff(k0)
    pord = m1 - m2
    if pord > 0:
        xi, wi = np.polynomial.legendre.leggauss(pord + 1)
        pts = (0.5 * (k0[:-1, None] + k0[1:, None]) + 0.5 * h[:, None] * xi[None, :]).ravel()
        wts = (0.5 * h[:, None] * wi[None, :]).ravel()
    else:
        pts = 0.5 * (k0[:-1] + k0[1:])
        wts = h

    k_pad = np.concatenate(
        [
            np.repeat(knots_full[0] - 1.0, m1),
            knots_full,
            np.repeat(knots_full[-1] + 1.0, m1),
        ]
    )
    c = np.zeros((len(k_pad) - order, k), dtype=float)
    c[m1 : m1 + k, :] = np.eye(k)
    G = BSpline(k_pad, c, m1, extrapolate=False)(pts, nu=m2)
    np.nan_to_num(G, copy=False, nan=0.0)
    S = G.T @ (wts[:, None] * G)
    S = 0.5 * (S + S.T)

    return (
        X,
        [S],
        {
            "bs": "bs",
            "knots": knots_full,
            "ord": order,
            "m": m,
            "rank": k - m2,
            "null_space_dim": m2,
        },
    )


def _eta_1d(r: np.ndarray) -> np.ndarray:
    """Thin plate spline Green's function for d=1, m=2: eta(r) = (1/12) * |r|^3."""
    out = np.abs(r)
    out *= r * r * (1.0 / 12.0)
    return out


def _wood_qt(A: np.ndarray) -> np.ndarray:
    """Right-hand Householder factorization: A Q = [0, R]."""
    M, k = A.shape
    Q = np.eye(k)
    A_work = A.copy()
    for i in range(M):
        p = A_work[i, : k - i].copy()
        m = np.max(np.abs(p))
        if m > 0.0:
            p /= m
        lsq = np.sqrt(np.sum(p * p))
        if p[k - i - 1] < 0.0:
            lsq = -lsq
        p[k - i - 1] += lsq
        g = 1.0 / (lsq * p[k - i - 1]) if lsq != 0.0 else 0.0
        lsq *= m
        A_work[i + 1 :, : k - i] -= (A_work[i + 1 :, : k - i] @ p)[:, None] * (
            g * p
        )
        Q[:, : k - i] -= (Q[:, : k - i] @ p)[:, None] * (g * p)
        A_work[i, k - i - 1] = -lsq
        A_work[i, : k - i - 1] = 0.0
    return Q


def build_tp_basis(
    x: np.ndarray,
    k: int = 10,
    m: int = 2,
    shrink: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """Thin plate regression spline basis ('tp' / 'ts') following Wood (2003)."""
    x = np.asarray(x, dtype=float).ravel()
    M = 2
    k = max(k, M + 1)

    x_mean = float(np.mean(x))
    x_c = x - x_mean

    xu, inv_idx = np.unique(x_c, return_inverse=True)
    nu = len(xu)
    if nu < k:
        raise ValueError(
            f"Covariate has only {nu} unique values, cannot support basis dimension k={k}."
        )

    E = _eta_1d(xu[:, None] - xu[None, :])
    T = np.column_stack([np.ones(nu), xu])

    eigvals, eigvecs = np.linalg.eigh(E)
    top_k = np.argsort(np.abs(eigvals))[::-1][:k]
    sort_idx = top_k[np.argsort(eigvals[top_k])[::-1]]
    v_k, U_k = eigvals[sort_idx], eigvecs[:, sort_idx]

    Q = _wood_qt(T.T @ U_k)
    X_u = (U_k * v_k[None, :]) @ Q
    X_u[:, k - M :] = T
    UkZ = U_k @ Q[:, : k - M]

    S_full = Q.T @ (v_k[:, None] * Q)
    S_full[k - M :, :] = 0.0
    S_full[:, k - M :] = 0.0
    S = 0.5 * (S_full + S_full.T)

    X = X_u[inv_idx, :]
    col_rms = np.sqrt(np.mean(X**2, axis=0))
    col_rms[col_rms == 0.0] = 1.0
    X = X / col_rms[None, :]
    S = S / (col_rms[:, None] * col_rms[None, :])
    S = 0.5 * (S + S.T)

    if shrink:
        es_val, es_vec = np.linalg.eigh(S)
        es_val[:M] = es_val[M] * 0.1
        S = es_vec @ (es_val[:, None] * es_vec.T)
        S = 0.5 * (S + S.T)

    return (
        X,
        [S],
        {
            "bs": "ts" if shrink else "tp",
            "shift": x_mean,
            "col_rms": col_rms,
            "Xu": xu,
            "UkZ": UkZ,
            "k": k,
            "M": M,
            "rank": k if shrink else k - M,
            "null_space_dim": 0 if shrink else M,
        },
    )


def predict_smooth_basis(x: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """Evaluate basis functions on new covariate values x using stored metadata."""
    x = np.asarray(x, dtype=float).ravel()
    bs = meta["bs"]
    if bs in ("tp", "ts"):
        shift = meta["shift"]
        X_smooth = _eta_1d((x - shift)[:, None] - meta["Xu"][None, :]) @ meta["UkZ"]
        T_new = np.column_stack([np.ones(len(x)), x - shift])
        return np.column_stack([X_smooth, T_new]) / meta["col_rms"][None, :]
    if bs in ("cr", "cs"):
        return _eval_cr_basis(x, meta["xp"], meta["F"])
    if bs == "cc":
        return _eval_cc_basis(x, meta["xp"], meta["BD"])
    if bs == "cp":
        return _c_spline_des(x, meta["knots"], meta["ord"])
    if bs in ("ps", "bs", "b"):
        return _spline_design(meta["knots"], x, meta["ord"], outer_ok=True)
    if bs == "re":
        return (x[:, None] == meta["levels"][None, :]).astype(float)
    raise ValueError(f"Unknown basis type bs='{bs}' in predict_smooth_basis.")


def build_smooth_basis(
    x: np.ndarray,
    bs: str = "tp",
    k: int = 10,
    knots: Optional[np.ndarray] = None,
    m: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """Dispatch to the requested basis constructor."""
    bs = bs.lower()
    if bs in ("tp", "ts"):
        return build_tp_basis(
            x, k=k, m=m if isinstance(m, int) else 2, shrink=(bs == "ts")
        )
    if bs in ("cr", "cs"):
        return build_cr_basis(x, k=k, knots=knots, shrink=(bs == "cs"))
    if bs == "cc":
        return build_cc_basis(x, k=k, knots=knots)
    if bs == "ps":
        return build_ps_basis(
            x, k=k, knots=knots, m=m if isinstance(m, tuple) else (2, 2)
        )
    if bs == "cp":
        return build_cp_basis(
            x, k=k, knots=knots, m=m if isinstance(m, tuple) else (2, 2)
        )
    if bs in ("bs", "b"):
        return build_bs_basis(
            x, k=k, knots=knots, m=m if isinstance(m, tuple) else (3, 2)
        )
    if bs == "re":
        xu, inv_idx = np.unique(x, return_inverse=True)
        K = len(xu)
        return (
            np.eye(K)[inv_idx],
            [np.eye(K)],
            {"bs": "re", "levels": xu, "rank": K, "null_space_dim": 0},
        )
    raise ValueError(
        f"Unknown basis type bs='{bs}'. Supported types: 'tp', 'ts', 'cr', 'cs', 'cc', 'ps', 'cp', 'bs', 're'."
    )


def rescale_penalty(
    X: np.ndarray, S_list: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[float]]:
    """Rescale penalty matrix by norm(X, 'I')^2."""
    if not S_list:
        return [], []
    maXX = float(np.max(np.sum(np.abs(X), axis=1)) ** 2) or 1.0
    S_scaled, scales = [], []
    for S in S_list:
        maS = float(np.max(np.sum(np.abs(S), axis=0)) / maXX) or 1.0
        S_scaled.append(S / maS)
        scales.append(maS)
    return S_scaled, scales


def absorb_sum_to_zero_constraint(
    X: np.ndarray,
    S_list: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Absorb sample sum-to-zero constraint C beta = 0 where C = 1^T X."""
    C = np.sum(X, axis=0, keepdims=True)
    Q, _ = np.linalg.qr(C.T, mode="complete")
    Z = Q[:, 1:]
    X_tilde = X @ Z
    S_tilde = []
    for S in S_list:
        ZS = Z.T @ S @ Z
        S_tilde.append(0.5 * (ZS + ZS.T))
    return X_tilde, S_tilde, Z
