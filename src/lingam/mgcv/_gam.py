"""Multi-term GAM with formula interface and REML/GCV optimization (mgcv backend)."""

from __future__ import annotations

from itertools import product as cartesian_product
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from .._formula import parse_formula
from ._optimizer import fit_mgcv_engine
from ._terms import (
    _MgcvFactorTerm,
    _MgcvLinearTerm,
    _MgcvSplineTerm,
    _MgcvTensorTerm,
    _MgcvTerm,
    instantiate_mgcv_terms,
)


class GAMCore:
    """Multi-term Generalized Additive Model with formula interface (mgcv backend).

    Parameters
    ----------
    formula : str
        Model specification, e.g.::

            "s(0, bs='tp', k=10) + te(1, 2, bs='cr', k=5) + f(3) + l(4)"

        Also accepts pyGAM syntax::

            "s(0, n_splines=10) + te(1, 2) + f(3) + l(4)"

    method : str, default 'REML'
        Smoothing parameter estimation criterion: 'REML' (mgcv default)
        or 'GCV.Cp'.
    gamma : float, default 1.0
        Degrees of freedom penalty multiplier for GCV.
    fit_intercept : bool, default True
        Whether to include an unpenalised intercept term.
    spline_order : int, default 3
        Default polynomial degree for B-splines / P-splines.
    """

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
        self._B_solve: Optional[np.ndarray] = None
        self._design_matrix_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.coef_ is not None

    @property
    def n_terms(self) -> int:
        return len(self._terms)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or gridsearch() first.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"GAMCore(formula='{self.formula}', method='{self.method}') [{status}]"

    def _compile(self, x: np.ndarray) -> None:
        for term in self._terms:
            term.compile(x)

    def _build_model_matrix(self, x: np.ndarray) -> np.ndarray:
        blocks = [t.build_columns(x) for t in self._terms]
        if self.fit_intercept:
            blocks.append(np.ones((len(x), 1), dtype=float))
        return np.hstack(blocks)

    def _compute_coef_slices(self) -> None:
        slices = []
        offset = 0
        for t in self._terms:
            slices.append((offset, offset + t.n_coefs))
            offset += t.n_coefs
        self._coef_slices = slices

    # ── fit ──────────────────────────────────────────────────────

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
        n, p = X.shape

        # Construct global penalty matrices for each term's penalties
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
                sp_fixed.append(term_sps[pen_idx] if pen_idx < len(term_sps) else None)

        fit_res = fit_mgcv_engine(
            X=X,
            y=y_arr,
            S_list=S_list,
            sp_fixed=sp_fixed if any(sp is not None for sp in sp_fixed) else None,
            method=self.method,
            gamma=self.gamma,
            coef_slices=self._coef_slices,
            M_p=1 if self.fit_intercept else 0,
        )

        self.coef_ = fit_res["coef"]
        self._B_solve = fit_res["H_inv"] @ X.T

        # Update terms with estimated smoothing parameters
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

    # ── predict / partial dependence ──────────────────────────────

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict values for new feature data x."""
        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        X = self._build_model_matrix(x_arr)
        return X @ self.coef_

    def partial_dependence(self, term_idx: int, x: ArrayLike) -> np.ndarray:
        """Evaluate the partial contribution of term term_idx."""
        self._check_fitted()
        if term_idx < 0 or term_idx >= len(self._terms):
            raise IndexError(f"term_idx {term_idx} out of range (0..{len(self._terms) - 1}).")
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        B_term = self._terms[term_idx].build_columns(x_arr)
        start, end = self._coef_slices[term_idx]
        return B_term @ self.coef_[start:end]

    # ── intervals ────────────────────────────────────────────────

    def confidence_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        """Compute Bayesian confidence intervals for the linear predictor."""
        return self._compute_intervals(x, width, prediction=False)

    def prediction_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        """Compute prediction intervals for new observations."""
        return self._compute_intervals(x, width, prediction=True)

    def _compute_intervals(self, x: ArrayLike, width: float, prediction: bool) -> np.ndarray:
        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        X = self._build_model_matrix(x_arr)
        lp = X @ self.coef_

        cov = self.statistics_["cov"]
        var = np.sum((X @ cov) * X, axis=1)
        if prediction:
            var += self.statistics_["scale"]

        alpha = (1.0 - width) / 2.0
        df = max(self.statistics_["n_samples"] - self.statistics_["edof"], 1.0)
        q_lo = stats.t.ppf(alpha, df=df)
        q_hi = stats.t.ppf(1.0 - alpha, df=df)

        se = np.sqrt(np.maximum(var, 1e-14))
        return np.column_stack([lp + q_lo * se, lp + q_hi * se])

    # ── statistics ────────────────────────────────────────────────

    def loglikelihood(self) -> float:
        """Gaussian log-likelihood of fitted model."""
        self._check_fitted()
        n = self.statistics_["n_samples"]
        rss = self.statistics_["rss"]
        scale = self.statistics_["scale"]
        return -0.5 * n * np.log(2.0 * np.pi * scale) - 0.5 * rss / scale

    def aic(self) -> float:
        """Akaike Information Criterion (AIC)."""
        self._check_fitted()
        return -2.0 * self.loglikelihood() + 2.0 * (self.statistics_["edof"] + 1)

    def bic(self) -> float:
        """Bayesian Information Criterion (BIC)."""
        self._check_fitted()
        n = self.statistics_["n_samples"]
        k = self.statistics_["edof"] + 1
        return -2.0 * self.loglikelihood() + np.log(n) * k

    def deviance_explained(self) -> float:
        """Fraction of null deviance explained (pseudo R-squared)."""
        self._check_fitted()
        return self.statistics_["deviance_explained"]

    # ── grid search ──────────────────────────────────────────────

    def gridsearch(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_splines_grids: Optional[List[ArrayLike]] = None,
        gamma: float = 1.0,
    ) -> "GAMCore":
        """Search over candidate basis dimensions k for smooth terms via REML/GCV.

        Parameters
        ----------
        x, y : array-like
            Training data.
        n_splines_grids : list of array-like, optional
            Candidate basis dimension arrays for smooth terms.
        gamma : float, default 1.0
            Degrees-of-freedom multiplier.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        # Collect smooth terms that can vary k
        smooth_indices = [
            i for i, t in enumerate(self._terms) if isinstance(t, (_MgcvSplineTerm, _MgcvTensorTerm))
        ]

        if not smooth_indices or n_splines_grids is None:
            # Just fit with current settings
            return self.fit(x_arr, y_arr)

        all_combos = list(cartesian_product(*n_splines_grids))
        best_score = np.inf
        best_combo = None

        for combo in all_combos:
            # Create term configs with new k values
            configs_copy = [dict(cfg) for cfg in self._term_configs]
            for idx_in_smooth, term_i in enumerate(smooth_indices):
                cfg = configs_copy[term_i]
                cfg["kwargs"] = dict(cfg.get("kwargs", {}))
                cfg["kwargs"]["k"] = combo[idx_in_smooth]

            terms_cand = instantiate_mgcv_terms(configs_copy)
            cand_model = GAMCore(
                formula=self.formula,
                method=self.method,
                gamma=gamma,
                fit_intercept=self.fit_intercept,
                spline_order=self.spline_order,
            )
            cand_model._terms = terms_cand
            cand_model._term_configs = configs_copy
            try:
                cand_model.fit(x_arr, y_arr)
                score = cand_model.statistics_["score"]
                if score < best_score:
                    best_score = score
                    best_combo = combo
            except Exception:
                continue

        if best_combo is not None:
            for idx_in_smooth, term_i in enumerate(smooth_indices):
                cfg = self._term_configs[term_i]
                cfg["kwargs"] = dict(cfg.get("kwargs", {}))
                cfg["kwargs"]["k"] = best_combo[idx_in_smooth]
            self._terms = instantiate_mgcv_terms(self._term_configs)

        return self.fit(x_arr, y_arr)

    # ── summary ──────────────────────────────────────────────────

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

        # Parametric coefficients: Intercept + linear terms
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
                lines.append(f"{name:11s} {b:10.4f} {se_b:10.4f} {tb:7.3f} {pb:8.4g}")
            lines.append("")

        # Smooth terms
        smooth_rows = []
        for i, t in enumerate(self._terms):
            if isinstance(t, (_MgcvSplineTerm, _MgcvTensorTerm)):
                if isinstance(t, _MgcvSplineTerm):
                    label = f"s({t.feature})"
                    ref_df = t.k - 1
                else:
                    feats_str = ",".join(str(f) for f in t.features)
                    label = f"te({feats_str})"
                    ref_df = t.n_coefs

                edf = self.statistics_["edf_per_term"][i]
                f_stat = self.statistics_["f_stat_per_term"][i]
                pval = self.statistics_["p_values_per_term"][i]
                smooth_rows.append((label, edf, ref_df, f_stat, pval))

        if smooth_rows:
            lines.append("Approximate significance of smooth terms:")
            lines.append("        edf Ref.df      F p-value")
            for label, edf, ref_df, f_stat, pval in smooth_rows:
                lines.append(f"{label:7s} {edf:6.2f} {ref_df:6d} {f_stat:6.3f} {pval:8.4g}")
            lines.append("")

        lines.extend([
            f"R-sq.(adj) = {self.statistics_['r_sq_adj']:.4f}   Deviance explained = {self.statistics_['deviance_explained'] * 100:.1f}%",
            f"{score_name} = {score_val:10.4f}  Scale est. = {self.statistics_['scale']:.4f}   n = {n}",
            "",
        ])
        return "\n".join(lines)

    # ── plotting ─────────────────────────────────────────────────

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 200,
        figsize: Optional[Tuple[int, int]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple["plt.Figure", np.ndarray]:
        """Plot partial dependence for all terms in the GAM."""
        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        def _fname(idx: int) -> str:
            if feature_names is not None and idx < len(feature_names):
                return feature_names[idx]
            return f"x[{idx}]"

        p_predictors = x_arr.shape[1]
        n_terms = len(self._terms)
        n_term_cols = min(3, n_terms)
        n_term_rows = int(np.ceil(n_terms / n_term_cols)) if n_terms > 0 else 0
        n_rows = 1 + n_term_rows
        n_cols = max(n_term_cols, 1)

        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)

        feat_mins = np.min(x_arr, axis=0)
        feat_maxs = np.max(x_arr, axis=0)
        medians = np.median(x_arr, axis=0)

        # 1. Overall fit
        ax_overall = fig.add_subplot(gs[0, :])
        grid0 = np.linspace(feat_mins[0], feat_maxs[0], n_grid)
        x_overall = np.tile(medians, (n_grid, 1))
        x_overall[:, 0] = grid0
        y_overall = self.predict(x_overall)
        sort_idx = np.argsort(grid0)

        ax_overall.plot(grid0[sort_idx], y_overall[sort_idx], "C0-", lw=2, label="GAM fit")
        ax_overall.scatter(x_arr[:, 0], y_arr, alpha=0.4, color="gray", s=10, label="Data")
        ax_overall.set(xlabel=_fname(0) if p_predictors > 1 else "x", ylabel="y")
        ax_overall.set_title(
            f"GAM Fit (Score={self.statistics_['score']:.3f}, EDF={self.statistics_['edof']:.1f}, n={self.statistics_['n_samples']})"
        )
        ax_overall.legend(fontsize=7, loc="best")

        # 2. Individual terms
        for i, term in enumerate(self._terms):
            row = 1 + i // n_term_cols
            col = i % n_term_cols
            ax = fig.add_subplot(gs[row, col])

            if isinstance(term, _MgcvSplineTerm):
                x_sweep = np.tile(medians, (n_grid, 1))
                g_feat = np.linspace(feat_mins[term.feature], feat_maxs[term.feature], n_grid)
                x_sweep[:, term.feature] = g_feat
                pdep = self.partial_dependence(i, x_sweep)
                sp_val = term.sp[0] if term.sp else 0.0
                ax.plot(g_feat, pdep, "C1-", lw=2, label=f"s({term.feature})")
                ax.axhline(0, color="gray", ls="--", lw=0.5)
                ax.set(xlabel=_fname(term.feature), ylabel="Contribution")
                ax.set_title(f"s({term.feature})  k={term.k}, sp={sp_val:.3g}")
                ax.legend(fontsize=7)

            elif isinstance(term, _MgcvTensorTerm):
                if term.d == 2:
                    ng2d = min(n_grid, 60)
                    g0 = np.linspace(feat_mins[term.features[0]], feat_maxs[term.features[0]], ng2d)
                    g1 = np.linspace(feat_mins[term.features[1]], feat_maxs[term.features[1]], ng2d)
                    f0, f1 = np.meshgrid(g0, g1, indexing="ij")
                    x_2d = np.tile(medians, (ng2d * ng2d, 1))
                    x_2d[:, term.features[0]] = f0.ravel()
                    x_2d[:, term.features[1]] = f1.ravel()
                    pdep = self.partial_dependence(i, x_2d).reshape(ng2d, ng2d)
                    im = ax.pcolormesh(f0, f1, pdep, shading="auto", cmap="RdBu_r")
                    plt.colorbar(im, ax=ax, label="Contribution")
                    ax.set(xlabel=_fname(term.features[0]), ylabel=_fname(term.features[1]))
                    feats_str = ",".join(str(f) for f in term.features)
                    ax.set_title(f"te({feats_str})")
            elif isinstance(term, _MgcvLinearTerm):
                x_sweep = np.tile(medians, (n_grid, 1))
                g_feat = np.linspace(feat_mins[term.feature], feat_maxs[term.feature], n_grid)
                x_sweep[:, term.feature] = g_feat
                pdep = self.partial_dependence(i, x_sweep)
                b = self.coef_[self._coef_slices[i][0]]
                ax.plot(g_feat, pdep, "C4-", lw=2)
                ax.axhline(0, color="gray", ls="--", lw=0.5)
                ax.set(xlabel=_fname(term.feature), ylabel="Contribution", title=f"l({term.feature}) beta={b:.4f}")
            elif isinstance(term, _MgcvFactorTerm):
                col_vals = np.asarray(x_arr[:, term.feature]).ravel()
                unique_lvls = np.unique(col_vals)
                contribs = []
                for lvl in unique_lvls:
                    x_lvl = medians.copy().reshape(1, -1)
                    x_lvl[0, term.feature] = lvl
                    contribs.append(self.partial_dependence(i, x_lvl)[0])
                colors = ["C2" if v >= 0 else "C3" for v in contribs]
                ax.bar(range(len(contribs)), contribs, color=colors, edgecolor="k", lw=0.5)
                ax.axhline(0, color="gray", ls="-", lw=0.5)
                ax.set_xticks(range(len(contribs)))
                ax.set_xticklabels(unique_lvls, fontsize=8)
                ax.set(ylabel="Contribution", title=f"f({term.feature})")

        fig.suptitle(f"GAMCore Decomposition — {self.formula}", fontsize=12, fontweight="bold")
        return fig, np.array(fig.axes)

    def plot_diagnostics(
        self,
        x: ArrayLike,
        y: ArrayLike,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Tuple["plt.Figure", np.ndarray]:
        """Diagnostic plots for model validation."""
        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        y_fit = self.predict(x_arr)
        residuals = y_arr - y_fit
        r2 = self.deviance_explained()

        figsize = figsize or (10, 9)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()

        # 1. Observed vs Fitted
        ax = axes[0]
        ax.scatter(y_fit, y_arr, alpha=0.5, edgecolors="k", s=20)
        lims = [min(y_arr.min(), y_fit.min()), max(y_arr.max(), y_fit.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="1:1")
        ax.set(xlabel="Fitted values", ylabel="Observed values", title=f"Observed vs Fitted (R²={r2:.3f})")
        ax.legend()

        # 2. Residuals vs Fitted
        ax = axes[1]
        ax.scatter(y_fit, residuals, alpha=0.5, edgecolors="k", s=20)
        ax.axhline(0, color="r", linestyle="--", lw=1.5)
        ax.set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs Fitted")

        # 3. Normal Q-Q
        ax = axes[2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q")
        ax.get_lines()[0].set_markerfacecolor("steelblue")
        ax.get_lines()[0].set_markeredgecolor("k")
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[1].set_color("r")

        # 4. Residual density
        ax = axes[3]
        ax.hist(residuals, bins="auto", density=True, color="steelblue", edgecolor="k", alpha=0.7)
        sigma = np.std(residuals, ddof=1)
        x_norm = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(x_norm, stats.norm.pdf(x_norm, 0.0, sigma), "r-", lw=2, label="Normal approx.")
        ax.set(xlabel="Residuals", ylabel="Density", title="Residual Distribution")
        ax.legend()

        fig.suptitle(f"GAMCore Diagnostics — {self.formula}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig, axes
