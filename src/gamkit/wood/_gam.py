"""Multi-term GAM with formula interface and REML/GCV optimization (mgcv backend)."""

from __future__ import annotations

from itertools import product as cartesian_product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from .._base import _GAMBase
from .._formula import parse_formula
from ._optimizer import fit_mgcv_engine
from ._terms import (
    _MgcvLinearTerm,
    _MgcvSplineTerm,
    _MgcvTensorTerm,
    _MgcvTerm,
    instantiate_mgcv_terms,
)


class GAMCore(_GAMBase):
    """Multi-term Generalized Additive Model with formula interface (mgcv backend)."""

    _scale_is_var = True

    def __init__(
        self,
        formula: str,
        method: str = "REML",
        gamma: float = 1.0,
        fit_intercept: bool = True,
        spline_order: int = 3,
    ) -> None:
        self.formula = formula
        self.method = method.upper()
        self.gamma = float(gamma)
        self.fit_intercept = bool(fit_intercept)
        self.spline_order = int(spline_order)

        configs = parse_formula(formula)
        self._term_configs = configs
        self._terms: List[_MgcvTerm] = instantiate_mgcv_terms(configs)

        self._coef_slices: List[Tuple[int, int]] = []
        self.coef_: Optional[np.ndarray] = None
        self.statistics_: Optional[Dict[str, Any]] = None
        self._design_matrix_: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"GAMCore(formula='{self.formula}', method='{self.method}') [{status}]"

    def fit(self, x: ArrayLike, y: ArrayLike) -> "GAMCore":
        """Fit multi-term GAM model using REML or GCV criteria."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("Input data contains NaN or infinity.")
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        self._compile(x_arr)
        self._compute_coef_slices()
        X = self._build_model_matrix(x_arr)
        self._design_matrix_ = X
        p = X.shape[1]

        S_list: List[np.ndarray] = []
        sp_fixed: List[Optional[float]] = []

        for i, term in enumerate(self._terms):
            start, end = self._coef_slices[i]
            pens = term.build_penalties()
            term_sps = term.sp if term.sp is not None else [None] * len(pens)
            for pen_idx, P_sub in enumerate(pens):
                P_full = np.zeros((p, p), dtype=float)
                P_full[start:end, start:end] = P_sub
                S_list.append(P_full)
                sp_fixed.append(
                    term_sps[pen_idx] if pen_idx < len(term_sps) else None
                )

        fit_res = fit_mgcv_engine(
            X=X,
            y=y_arr,
            S_list=S_list,
            sp_fixed=(
                sp_fixed if any(sp is not None for sp in sp_fixed) else None
            ),
            method=self.method,
            gamma=self.gamma,
            coef_slices=self._coef_slices,
            M_p=1 if self.fit_intercept else 0,
        )

        self.coef_ = fit_res["coef"]

        sp_idx = 0
        for term in self._terms:
            n_pens = len(term.build_penalties())
            if n_pens > 0:
                term.sp = fit_res["sp"][sp_idx : sp_idx + n_pens]
                sp_idx += n_pens

        self.statistics_ = {
            "edof": fit_res["edof"],
            "edof_per_coef": fit_res["edof_per_coef"],
            "scale": fit_res["scale"],
            "cov": fit_res["Vp"],
            "Vp": fit_res["Vp"],
            "Ve": fit_res["Ve"],
            "se": fit_res["se"],
            "rss": fit_res["rss"],
            "deviance": fit_res["deviance"],
            "null_deviance": fit_res["null_deviance"],
            "deviance_explained": fit_res["deviance_explained"],
            "r_sq_adj": fit_res["r_sq_adj"],
            "n_samples": fit_res["n_samples"],
            "GCV": fit_res["GCV"],
            "score": fit_res["score"],
            "method": fit_res["method"],
            "sp": fit_res["sp"],
            "edf_per_term": fit_res["edf_per_term"],
            "f_stat_per_term": fit_res["f_stat_per_term"],
            "p_values_per_term": fit_res["p_values_per_term"],
        }
        return self

    def gridsearch(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_splines_grids: Optional[List[ArrayLike]] = None,
        gamma: float = 1.0,
    ) -> "GAMCore":
        """Search over candidate basis dimensions k for smooth terms via REML/GCV."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        smooth_indices = [
            i
            for i, t in enumerate(self._terms)
            if isinstance(t, (_MgcvSplineTerm, _MgcvTensorTerm))
        ]
        if not smooth_indices or n_splines_grids is None:
            return self.fit(x_arr, y_arr)

        best_score = np.inf
        best_combo = None

        for combo in cartesian_product(*n_splines_grids):
            configs_copy = [dict(cfg) for cfg in self._term_configs]
            for idx_in_smooth, term_i in enumerate(smooth_indices):
                cfg = configs_copy[term_i]
                cfg["kwargs"] = {
                    **cfg.get("kwargs", {}),
                    "k": combo[idx_in_smooth],
                }
            cand_model = GAMCore(
                formula=self.formula,
                method=self.method,
                gamma=gamma,
                fit_intercept=self.fit_intercept,
                spline_order=self.spline_order,
            )
            cand_model._terms = instantiate_mgcv_terms(configs_copy)
            cand_model._term_configs = configs_copy
            try:
                cand_model.fit(x_arr, y_arr)
                if cand_model.statistics_["score"] < best_score:
                    best_score = cand_model.statistics_["score"]
                    best_combo = combo
            except Exception:
                continue

        if best_combo is not None:
            for idx_in_smooth, term_i in enumerate(smooth_indices):
                cfg = self._term_configs[term_i]
                cfg["kwargs"] = {
                    **cfg.get("kwargs", {}),
                    "k": best_combo[idx_in_smooth],
                }
            self._terms = instantiate_mgcv_terms(self._term_configs)

        return self.fit(x_arr, y_arr)

    def summary(self) -> str:
        """Formatted summary table matching R mgcv's summary.gam."""
        self._check_fitted()
        n = self.statistics_["n_samples"]
        score_name = "REML" if self.method == "REML" else "GCV"
        score_val = self.statistics_["score"]

        lines = [
            "",
            "Family: gaussian ",
            "Link function: identity ",
            "",
            f"Formula: {self.formula}",
            "",
        ]

        param_rows = []
        df_res = max(n - self.statistics_["edof"], 1.0)
        if self.fit_intercept:
            b0 = self.coef_[-1]
            se0 = self.statistics_["se"][-1]
            t0 = b0 / max(se0, 1e-12)
            p0 = 2.0 * (1.0 - stats.t.cdf(np.abs(t0), df=df_res))
            param_rows.append(("(Intercept)", b0, se0, t0, p0))

        for i, t in enumerate(self._terms):
            if isinstance(t, _MgcvLinearTerm):
                start, _ = self._coef_slices[i]
                b = self.coef_[start]
                se_b = self.statistics_["se"][start]
                tb = b / max(se_b, 1e-12)
                pb = 2.0 * (1.0 - stats.t.cdf(np.abs(tb), df=df_res))
                param_rows.append((f"l({t.feature})", b, se_b, tb, pb))

        if param_rows:
            lines.append("Parametric coefficients:")
            lines.append("            Estimate Std. Error t value Pr(>|t|)")
            for name, b, se_b, tb, pb in param_rows:
                lines.append(
                    f"{name:11s} {b:10.4f} {se_b:10.4f} {tb:7.3f} {pb:8.4g}"
                )
            lines.append("")

        smooth_rows = []
        for i, t in enumerate(self._terms):
            if isinstance(t, (_MgcvSplineTerm, _MgcvTensorTerm)):
                if isinstance(t, _MgcvSplineTerm):
                    label, ref_df = f"s({t.feature})", t.k - 1
                else:
                    label = f"te({','.join(str(f) for f in t.features)})"
                    ref_df = t.n_coefs
                smooth_rows.append(
                    (
                        label,
                        self.statistics_["edf_per_term"][i],
                        ref_df,
                        self.statistics_["f_stat_per_term"][i],
                        self.statistics_["p_values_per_term"][i],
                    )
                )

        if smooth_rows:
            lines.append("Approximate significance of smooth terms:")
            lines.append("        edf Ref.df      F p-value")
            for label, edf, ref_df, f_stat, pval in smooth_rows:
                lines.append(
                    f"{label:7s} {edf:6.2f} {ref_df:6d} {f_stat:6.3f} {pval:8.4g}"
                )
            lines.append("")

        lines.extend(
            [
                f"R-sq.(adj) = {self.statistics_['r_sq_adj']:.4f}   Deviance explained = {self.statistics_['deviance_explained'] * 100:.1f}%",
                f"{score_name} = {score_val:10.4f}  Scale est. = {self.statistics_['scale']:.4f}   n = {n}",
                "",
            ]
        )
        return "\n".join(lines)

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 200,
        figsize: Optional[Tuple[int, int]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple:
        """Plot partial dependence for all terms in the GAM."""
        return self._plot_gamcore_decomposition(
            x,
            y,
            n_grid=n_grid,
            extrap_frac=0.0,
            figsize=figsize,
            feature_names=feature_names,
        )
