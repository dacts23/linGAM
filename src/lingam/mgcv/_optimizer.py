"""Penalized least squares solvers and smoothing parameter optimization for mgcv GAM backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import linalg, optimize, stats


def solve_pls_fixed(
    X: np.ndarray,
    y: np.ndarray,
    S_list: List[np.ndarray],
    sp: List[float],
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Solve penalized least squares for fixed smoothing parameters sp.

    Returns (coef, H_inv, rss, penalty_val).
    """
    n, p = X.shape
    if Q is None or R is None:
        Q, R = np.linalg.qr(X)

    qty = Q.T @ y
    Xty = R.T @ qty
    RtR = R.T @ R
    ss_orth = max(0.0, float(y @ y) - float(qty @ qty))

    # Accumulate total penalty
    S_total = np.zeros((p, p), dtype=float)
    for s_mat, l_val in zip(S_list, sp):
        S_total += float(l_val) * s_mat

    H = RtR + S_total
    H[np.diag_indices_from(H)] += 1e-12

    try:
        c, lower = linalg.cho_factor(H)
        coef = linalg.cho_solve((c, lower), Xty)
        H_inv = linalg.cho_solve((c, lower), np.eye(p))
    except Exception:
        coef = np.linalg.solve(H, Xty)
        H_inv = np.linalg.pinv(H)

    r_diff = qty - R @ coef
    rss = max(0.0, ss_orth + float(r_diff @ r_diff))
    pen_val = float(coef.T @ S_total @ coef)
    return coef, H_inv, rss, pen_val


def fit_mgcv_engine(
    X: np.ndarray,
    y: np.ndarray,
    S_list: List[np.ndarray],
    sp_fixed: Optional[List[Optional[float]]] = None,
    method: str = "REML",
    gamma: float = 1.0,
    coef_slices: Optional[List[Tuple[int, int]]] = None,
    M_p: int = 1,
) -> Dict[str, Any]:
    """Fit GAM using mgcv REML or GCV criteria and compute full statistics."""
    n, p = X.shape
    method = method.upper()
    if method not in ("REML", "GCV.CP", "GCV"):
        raise ValueError(f"Unknown method '{method}'. Choose 'REML' or 'GCV.Cp'.")

    Q, R = np.linalg.qr(X)
    RtR = R.T @ R
    qty = Q.T @ y
    Xty = R.T @ qty
    yty = float(y @ y)
    ss_orth = max(0.0, yty - float(qty @ qty))

    n_pens = len(S_list)

    # Determine which smoothing parameters to estimate
    sp_vals = [1.0] * n_pens
    free_indices = []
    if sp_fixed is not None and len(sp_fixed) == n_pens:
        for idx, val in enumerate(sp_fixed):
            if val is not None:
                sp_vals[idx] = float(val)
            else:
                free_indices.append(idx)
    else:
        free_indices = list(range(n_pens))

    # Precompute log-determinant baseline for each penalty
    # Find positive eigenvalues of each penalty matrix
    ev_pos_list = []
    rank_S_list = []
    supports = []
    for S in S_list:
        ev = np.linalg.eigvalsh(S)
        pos = ev[ev > max(np.max(ev) * 1e-10, 1e-14)]
        ev_pos_list.append(pos)
        rank_S_list.append(len(pos))
        supports.append(np.where(np.diag(S) > 0)[0])

    disjoint_pens = True
    for i in range(n_pens):
        for j in range(i + 1, n_pens):
            if np.intersect1d(supports[i], supports[j]).size > 0:
                disjoint_pens = False
                break
        if not disjoint_pens:
            break

    def _eval_criterion(
        rho_vec: np.ndarray,
        compute_details: bool = False,
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray], float, float, float]:
        current_sp = list(sp_vals)
        for i, idx in enumerate(free_indices):
            current_sp[idx] = float(np.exp(rho_vec[i]))

        S_total = np.zeros((p, p), dtype=float)
        for s_mat, l_val in zip(S_list, current_sp):
            S_total += l_val * s_mat

        H = RtR + S_total
        H[np.diag_indices_from(H)] += 1e-12

        try:
            c, lower = linalg.cho_factor(H)
            coef = linalg.cho_solve((c, lower), Xty)
            log_det_H = 2.0 * float(np.sum(np.log(np.diag(c))))
        except Exception:
            coef = np.linalg.solve(H, Xty)
            log_det_H = float(np.linalg.slogdet(H)[1])
            c, lower = None, False

        r_diff = qty - R @ coef
        rss = max(0.0, ss_orth + float(r_diff @ r_diff))
        pen_val = float(coef.T @ S_total @ coef)
        Dp = rss + pen_val

        H_inv = None
        edf = 0.0

        if method == "REML":
            scale_est = max(Dp / max(n - M_p, 1), 1e-12)
            # Log determinant of penalty S_lambda:
            if n_pens == 1 or disjoint_pens:
                log_det_S = sum(
                    rank_S_list[i] * np.log(current_sp[i]) + float(np.sum(np.log(ev_pos_list[i])))
                    for i in range(n_pens)
                )
            else:
                ev_tot = np.linalg.eigvalsh(S_total)
                pos_tot = ev_tot[ev_tot > max(np.max(ev_tot) * 1e-10, 1e-14)]
                log_det_S = float(np.sum(np.log(pos_tot))) if len(pos_tot) > 0 else 0.0

            score = (
                Dp / (2.0 * scale_est)
                + 0.5 * (n - M_p) * np.log(2.0 * np.pi * scale_est)
                + 0.5 * log_det_H
                - 0.5 * log_det_S
            )
        else:  # GCV
            if c is not None:
                try:
                    Z_Rt = linalg.solve_triangular(c, R.T, lower=lower)
                    edf = float(np.sum(Z_Rt ** 2))
                except Exception:
                    H_inv = linalg.cho_solve((c, lower), np.eye(p))
                    edf = float(np.trace(H_inv @ RtR))
            else:
                H_inv = np.linalg.pinv(H)
                edf = float(np.trace(H_inv @ RtR))

            denom = max(n - gamma * edf, 1e-6)
            score = (n * rss) / (denom ** 2)
            scale_est = rss / max(n - edf, 1)

        if compute_details:
            if H_inv is None:
                if c is not None:
                    try:
                        H_inv = linalg.cho_solve((c, lower), np.eye(p))
                    except Exception:
                        H_inv = np.linalg.pinv(H)
                else:
                    H_inv = np.linalg.pinv(H)
            if edf == 0.0:
                edf = float(np.trace(H_inv @ RtR))

        return score, coef, H_inv, rss, scale_est, edf

    # Optimization
    if len(free_indices) == 0:
        # All smoothing parameters are fixed
        best_score, best_coef, best_H_inv, best_rss, best_scale, best_edf = _eval_criterion(np.zeros(0), compute_details=True)
        best_sp = sp_vals
    elif len(free_indices) == 1:
        # Single 1D optimization: use Brent's scalar search
        def f_1d(r: float) -> float:
            return _eval_criterion(np.array([r]), compute_details=False)[0]

        opt = optimize.minimize_scalar(f_1d, bounds=(-15.0, 15.0), method="bounded")
        best_rho = np.array([opt.x])
        best_score, best_coef, best_H_inv, best_rss, best_scale, best_edf = _eval_criterion(best_rho, compute_details=True)
        best_sp = list(sp_vals)
        best_sp[free_indices[0]] = float(np.exp(opt.x))
    else:
        # Multi-dimensional optimization: use L-BFGS-B
        def f_nd(r: np.ndarray) -> float:
            return _eval_criterion(r, compute_details=False)[0]

        init_rho = np.zeros(len(free_indices))
        bounds = [(-15.0, 15.0)] * len(free_indices)
        opt = optimize.minimize(f_nd, init_rho, method="L-BFGS-B", bounds=bounds)
        best_rho = opt.x
        best_score, best_coef, best_H_inv, best_rss, best_scale, best_edf = _eval_criterion(best_rho, compute_details=True)
        best_sp = list(sp_vals)
        for i, idx in enumerate(free_indices):
            best_sp[idx] = float(np.exp(best_rho[i]))

    # Compute covariances and diagnostics
    # Scale parameter for Gaussian GAM
    scale_val = best_scale
    Vp = best_H_inv * scale_val  # Bayesian covariance
    Ve = (best_H_inv @ RtR @ best_H_inv) * scale_val  # Frequentist covariance
    se = np.sqrt(np.maximum(np.diag(Vp), 1e-14))

    # EDF per coefficient and per term
    F_mat = best_H_inv @ RtR
    edf_per_coef = np.diag(F_mat)
    total_edf = float(np.sum(edf_per_coef))

    edf_per_term: List[float] = []
    p_values_per_term: List[float] = []
    f_stat_per_term: List[float] = []

    if coef_slices is not None:
        df2 = max(n - total_edf, 1.0)
        for start, end in coef_slices:
            term_edf = float(np.sum(edf_per_coef[start:end]))
            edf_per_term.append(term_edf)

            # Wood (2013) approximate test statistic
            beta_sub = best_coef[start:end]
            V_sub = Vp[start:end, start:end]
            try:
                inv_V_sub = linalg.pinv(V_sub)
                stat = float((beta_sub.T @ inv_V_sub @ beta_sub) / max(term_edf, 1e-6))
                pval = float(1.0 - stats.f.cdf(stat, max(term_edf, 1.0), df2))
            except Exception:
                stat = 0.0
                pval = 1.0
            f_stat_per_term.append(stat)
            p_values_per_term.append(pval)

    # Null deviance and pseudo R^2
    y_mean = float(np.mean(y))
    null_deviance = float(np.sum((y - y_mean) ** 2))
    dev_explained = max(0.0, 1.0 - best_rss / max(null_deviance, 1e-12))
    r_sq_adj = max(
        0.0,
        1.0 - (best_rss / max(n - total_edf, 1)) / (null_deviance / max(n - 1, 1)),
    )

    # GCV score
    denom_gcv = max(n - gamma * total_edf, 1.0)
    gcv_score = (n * best_rss) / (denom_gcv ** 2)

    # Gaussian log-likelihood
    loglik = -0.5 * n * np.log(2.0 * np.pi * scale_val) - 0.5 * best_rss / scale_val
    aic_val = -2.0 * loglik + 2.0 * (total_edf + 1)
    bic_val = -2.0 * loglik + np.log(n) * (total_edf + 1)

    return {
        "coef": best_coef,
        "sp": best_sp,
        "scale": scale_val,
        "Vp": Vp,
        "Ve": Ve,
        "se": se,
        "edof": total_edf,
        "edof_per_coef": edf_per_coef,
        "edf_per_term": edf_per_term,
        "f_stat_per_term": f_stat_per_term,
        "p_values_per_term": p_values_per_term,
        "rss": best_rss,
        "deviance": best_rss,
        "null_deviance": null_deviance,
        "deviance_explained": dev_explained,
        "r_sq_adj": r_sq_adj,
        "GCV": gcv_score,
        "score": best_score,
        "method": method,
        "loglik": loglik,
        "aic": aic_val,
        "bic": bic_val,
        "n_samples": n,
        "H_inv": best_H_inv,
        "RtR": RtR,
    }
