"""Penalized least squares solvers and smoothing parameter optimization for Wood (2011) GAM backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg, optimize, stats


def fit_wood_engine(
    X: np.ndarray,
    y: np.ndarray,
    S_list: List[np.ndarray],
    sp_fixed: Optional[List[Optional[float]]] = None,
    method: str = "REML",
    gamma: float = 1.0,
    coef_slices: Optional[List[Tuple[int, int]]] = None,
    M_p: int = 1,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit GAM using Wood (2011) REML or GCV criteria and compute full statistics."""
    n, p = X.shape
    method = method.upper()
    if method not in ("REML", "GCV.CP", "GCV"):
        raise ValueError(f"Unknown method '{method}'. Choose 'REML' or 'GCV.Cp'.")

    if weights is not None:
        null_deviance = float(
            np.sum(weights * (y - float(np.average(y, weights=weights))) ** 2)
        )
        log_w_sum = float(np.sum(np.log(weights[weights > 0])))
        w_sqrt = np.sqrt(weights)
        X = X * w_sqrt[:, None]
        y = y * w_sqrt
    else:
        null_deviance = float(np.sum((y - float(np.mean(y))) ** 2))
        log_w_sum = 0.0

    if n > p:
        R_aug = linalg.qr(
            np.column_stack([X, y]), mode="r", check_finite=False
        )[0]
        R = R_aug[:p, :p]
        qty = R_aug[:p, p]
        ss_orth = float(R_aug[p, p] ** 2)
    else:
        Q, R = np.linalg.qr(X)
        qty = Q.T @ y
        ss_orth = max(0.0, float(y @ y) - float(qty @ qty))
    RtR = R.T @ R
    Xty = R.T @ qty

    n_pens = len(S_list)
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

    pen_bounds: List[Tuple[int, int]] = []
    pen_sub: List[np.ndarray] = []
    rank_S_list: List[int] = []
    log_det_const: List[float] = []
    supports: List[np.ndarray] = []
    for S in S_list:
        nz = np.where(np.diag(S) > 0)[0]
        s0, s1 = (int(nz[0]), int(nz[-1]) + 1) if len(nz) > 0 else (0, 0)
        pen_bounds.append((s0, s1))
        sub = S[s0:s1, s0:s1]
        pen_sub.append(sub)
        supports.append(nz)
        if s1 > s0:
            ev = linalg.eigvalsh(sub, check_finite=False)
            pos = ev[ev > max(float(ev[-1]) * 1e-10, 1e-14)]
            rank_S_list.append(len(pos))
            log_det_const.append(
                float(np.sum(np.log(pos))) if len(pos) > 0 else 0.0
            )
        else:
            rank_S_list.append(0)
            log_det_const.append(0.0)

    overlapping = [False] * n_pens
    for i in range(n_pens):
        for j in range(i + 1, n_pens):
            if (
                pen_bounds[i][0] < pen_bounds[j][1]
                and pen_bounds[j][0] < pen_bounds[i][1]
                and np.intersect1d(supports[i], supports[j]).size > 0
            ):
                overlapping[i] = overlapping[j] = True

    overlap_groups: List[Tuple[List[int], int, Dict[int, np.ndarray]]] = []
    visited = [False] * n_pens
    for i in range(n_pens):
        if overlapping[i] and not visited[i]:
            grp = [i]
            visited[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(n_pens):
                    if overlapping[j] and not visited[j]:
                        if any(
                            np.intersect1d(supports[g], supports[j]).size > 0
                            for g in grp
                        ):
                            grp.append(j)
                            visited[j] = True
                            changed = True
            g_s0 = min(pen_bounds[g][0] for g in grp)
            g_s1 = max(pen_bounds[g][1] for g in grp)
            S_sum = np.zeros((g_s1 - g_s0, g_s1 - g_s0), dtype=float)
            for g_i in grp:
                s0, s1 = pen_bounds[g_i]
                S_sum[s0 - g_s0 : s1 - g_s0, s0 - g_s0 : s1 - g_s0] += pen_sub[
                    g_i
                ]
            ev0, U0 = linalg.eigh(S_sum, check_finite=False)
            U_pos = U0[:, ev0 > max(float(ev0[-1]) * 1e-10, 1e-14)]
            proj_subs = {
                g_i: U_pos[
                    pen_bounds[g_i][0] - g_s0 : pen_bounds[g_i][1] - g_s0, :
                ].T
                @ pen_sub[g_i]
                @ U_pos[
                    pen_bounds[g_i][0] - g_s0 : pen_bounds[g_i][1] - g_s0, :
                ]
                for g_i in grp
            }
            overlap_groups.append((grp, U_pos.shape[1], proj_subs))

    total_pen_rank = sum(
        rank_S_list[i] for i in range(n_pens) if not overlapping[i]
    ) + sum(r_dim for _, r_dim, _ in overlap_groups)
    M_p = max(p - total_pen_rank, 0)

    eye_p = np.eye(p, dtype=float)

    def _eval_criterion(
        rho_vec: np.ndarray,
        compute_details: bool = False,
        compute_grad: bool = False,
    ) -> Any:
        current_sp = list(sp_vals)
        for i, idx in enumerate(free_indices):
            current_sp[idx] = float(np.exp(rho_vec[i]))

        S_total = np.zeros((p, p), dtype=float)
        for (s0, s1), sub, l_val in zip(pen_bounds, pen_sub, current_sp):
            if s1 > s0:
                S_total[s0:s1, s0:s1] += l_val * sub

        H = RtR + S_total
        H.flat[:: p + 1] += 1e-12

        try:
            c, lower = linalg.cho_factor(H, check_finite=False)
            coef = linalg.cho_solve((c, lower), Xty, check_finite=False)
            log_det_H = 2.0 * float(np.sum(np.log(np.diag(c))))
        except Exception:
            coef = np.linalg.solve(H, Xty)
            log_det_H = float(np.linalg.slogdet(H)[1])
            c, lower = None, False

        r_diff = qty - R @ coef
        rss = max(0.0, ss_orth + float(r_diff @ r_diff))
        Dp = rss + float(coef @ (S_total @ coef))

        H_inv = None
        if compute_details or compute_grad:
            if c is not None:
                try:
                    H_inv = linalg.cho_solve(
                        (c, lower), eye_p, check_finite=False
                    )
                except Exception:
                    H_inv = np.linalg.pinv(H)
            else:
                H_inv = np.linalg.pinv(H)

        edf = 0.0
        grad = np.zeros(len(free_indices), dtype=float) if compute_grad else None

        if method == "REML":
            df_reml = max(n - M_p, 1)
            scale_est = max(Dp / df_reml, 1e-12)
            log_det_S = sum(
                rank_S_list[i] * np.log(current_sp[i]) + log_det_const[i]
                for i in range(n_pens)
                if not overlapping[i]
            )
            S_inv_proj: Dict[int, np.ndarray] = {}
            for grp, r_dim, proj_subs in overlap_groups:
                S_tilde = sum(current_sp[g_i] * proj_subs[g_i] for g_i in grp)
                d = np.sqrt(np.maximum(np.diag(S_tilde), 1e-300))
                S_bal = S_tilde / (d[:, None] * d[None, :])
                try:
                    c_s, low_s = linalg.cho_factor(S_bal, check_finite=False)
                    log_det_S += 2.0 * float(
                        np.sum(np.log(np.diag(c_s))) + np.sum(np.log(d))
                    )
                    if compute_grad:
                        S_bal_inv = linalg.cho_solve(
                            (c_s, low_s), np.eye(r_dim), check_finite=False
                        )
                        S_inv = S_bal_inv / (d[:, None] * d[None, :])
                        for g_i in grp:
                            S_inv_proj[g_i] = S_inv
                except Exception:
                    ev_s, u_s = linalg.eigh(S_tilde, check_finite=False)
                    pos_s = np.maximum(ev_s, 1e-300)
                    log_det_S += float(np.sum(np.log(pos_s)))
                    if compute_grad:
                        S_inv = (u_s / pos_s[None, :]) @ u_s.T
                        for g_i in grp:
                            S_inv_proj[g_i] = S_inv

            score = (
                Dp / (2.0 * scale_est)
                + 0.5 * df_reml * np.log(2.0 * np.pi * scale_est)
                + 0.5 * log_det_H
                - 0.5 * log_det_S
            )
            if compute_grad and H_inv is not None and grad is not None:
                for k_i, idx in enumerate(free_indices):
                    s0, s1 = pen_bounds[idx]
                    sub = pen_sub[idx]
                    lam_i = current_sp[idx]
                    b_sub = coef[s0:s1]
                    dDp = lam_i * float(b_sub @ sub @ b_sub)
                    dlogH = lam_i * float(np.sum(H_inv[s0:s1, s0:s1] * sub))
                    if not overlapping[idx]:
                        dlogS = float(rank_S_list[idx])
                    else:
                        dlogS = lam_i * float(
                            np.sum(
                                S_inv_proj[idx]
                                * next(
                                    ps[idx]
                                    for grp, _, ps in overlap_groups
                                    if idx in grp
                                )
                            )
                        )
                    grad[k_i] = 0.5 * (dDp / scale_est + dlogH - dlogS)
        else:
            if H_inv is not None:
                edf = float(np.sum(H_inv * RtR))
            elif c is not None:
                try:
                    Z_Rt = linalg.solve_triangular(
                        c,
                        R.T,
                        lower=lower,
                        trans="N" if lower else "T",
                        check_finite=False,
                    )
                    edf = float(np.sum(Z_Rt**2))
                except Exception:
                    H_inv = linalg.cho_solve(
                        (c, lower), eye_p, check_finite=False
                    )
                    edf = float(np.sum(H_inv * RtR))
            else:
                H_inv = np.linalg.pinv(H)
                edf = float(np.sum(H_inv * RtR))

            denom = max(n - gamma * edf, 1e-6)
            score = (n * rss) / (denom**2)
            scale_est = rss / max(n - edf, 1)
            if compute_grad and H_inv is not None and grad is not None:
                res_grad_vec = H_inv @ (RtR @ coef - Xty)
                H_RtR_H = (H_inv @ RtR) @ H_inv
                for k_i, idx in enumerate(free_indices):
                    s0, s1 = pen_bounds[idx]
                    sub = pen_sub[idx]
                    lam_i = current_sp[idx]
                    drss = (
                        -2.0
                        * lam_i
                        * float(res_grad_vec[s0:s1] @ (sub @ coef[s0:s1]))
                    )
                    dedf = -lam_i * float(np.sum(H_RtR_H[s0:s1, s0:s1] * sub))
                    grad[k_i] = (n / (denom**2)) * drss + (
                        2.0 * n * rss * gamma / (denom**3)
                    ) * dedf

        if compute_details and H_inv is not None and edf == 0.0:
            edf = float(np.sum(H_inv * RtR))

        if compute_grad:
            return score, grad
        return score, coef, H_inv, rss, scale_est, edf

    if len(free_indices) == 0:
        best_score, best_coef, best_H_inv, best_rss, _, best_edf = (
            _eval_criterion(np.zeros(0), compute_details=True)
        )
        best_scale = best_rss / max(n - best_edf, 1)
        best_sp = sp_vals
    else:
        n_free = len(free_indices)
        candidates = [np.full(n_free, v) for v in (-8.0, -4.0, 0.0, 4.0, 8.0)]
        rho_init = min(candidates, key=lambda r: _eval_criterion(r)[0])
        opt = optimize.minimize(
            lambda r: _eval_criterion(r, compute_grad=True),
            rho_init,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-15.0, 15.0)] * n_free,
            options={"ftol": 1e-14, "gtol": 1e-10},
        )
        best_score, best_coef, best_H_inv, best_rss, best_scale, best_edf = (
            _eval_criterion(opt.x, compute_details=True)
        )
        best_sp = list(sp_vals)
        for i, idx in enumerate(free_indices):
            best_sp[idx] = float(np.exp(opt.x[i]))

    scale_val = best_scale
    Vp = best_H_inv * scale_val
    H_inv_RtR = best_H_inv @ RtR
    Ve = (H_inv_RtR @ best_H_inv) * scale_val
    se = np.sqrt(np.maximum(np.diag(Vp), 1e-14))

    edf_per_coef = np.diag(H_inv_RtR)
    total_edf = float(np.sum(edf_per_coef))

    edf_per_term: List[float] = []
    p_values_per_term: List[float] = []
    f_stat_per_term: List[float] = []

    if coef_slices is not None:
        df2 = max(n - total_edf, 1.0)
        for start, end in coef_slices:
            term_edf = float(np.sum(edf_per_coef[start:end]))
            edf_per_term.append(term_edf)
            beta_sub = best_coef[start:end]
            V_sub = Vp[start:end, start:end]
            try:
                stat = float(
                    (beta_sub.T @ linalg.pinv(V_sub) @ beta_sub)
                    / max(term_edf, 1e-6)
                )
                pval = float(1.0 - stats.f.cdf(stat, max(term_edf, 1.0), df2))
            except Exception:
                stat, pval = 0.0, 1.0
            f_stat_per_term.append(stat)
            p_values_per_term.append(pval)

    dev_explained = max(0.0, 1.0 - best_rss / max(null_deviance, 1e-12))
    r_sq_adj = max(
        0.0,
        1.0
        - (best_rss / max(n - total_edf, 1))
        / (null_deviance / max(n - 1, 1)),
    )

    gcv_score = (n * best_rss) / (max(n - gamma * total_edf, 1.0) ** 2)
    loglik = (
        -0.5 * n * np.log(2.0 * np.pi * scale_val)
        - 0.5 * best_rss / scale_val
        + 0.5 * log_w_sum
    )

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
        "log_weights_sum": log_w_sum,
        "aic": -2.0 * loglik + 2.0 * (total_edf + 1),
        "bic": -2.0 * loglik + np.log(n) * (total_edf + 1),
        "n_samples": n,
        "H_inv": best_H_inv,
        "RtR": RtR,
    }
