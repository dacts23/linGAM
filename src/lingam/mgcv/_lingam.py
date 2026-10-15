"""Single-term penalised GAM — LinGAM class (mgcv backend)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ._optimizer import fit_mgcv_engine
from ._smooths import (
    absorb_sum_to_zero_constraint,
    build_smooth_basis,
    predict_smooth_basis,
    rescale_penalty,
)


class LinGAM:
    """Penalised GAM for a single smooth term y = beta_0 + f(x) + eps (mgcv backend).

    Compatible with both pyGAM and R mgcv conventions.

    Parameters
    ----------
    n_splines : int, default 10
        Number of basis functions (basis dimension k in mgcv).
    k : int, optional
        Alias for n_splines (mgcv convention). If provided, overrides n_splines.
    lam : float, optional, default None
        Smoothing parameter (sp in mgcv). If None (default), estimated
        automatically via REML (or GCV). If a float >= 0 is provided,
        smoothing is fixed to that value.
    sp : float, optional
        Alias for lam (mgcv convention). If provided, overrides lam.
    bs : str, default 'tp'
        Basis type:
        - 'tp': Thin plate regression splines (mgcv default)
        - 'ts': Thin plate regression splines with shrinkage
        - 'cr': Cubic regression splines (natural boundary conditions)
        - 'cs': Cubic regression splines with shrinkage
        - 'cc': Cyclic cubic regression splines
        - 'ps': P-splines (discrete difference penalties)
        - 'cp': Cyclic P-splines
        - 'bs': B-splines with integrated derivative penalties
        - 're': Random effect / ridge penalty
    method : str, default 'REML'
        Smoothing parameter estimation criterion: 'REML' (mgcv default)
        or 'GCV.Cp'.
    gamma : float, default 1.0
        Multiplier for degrees of freedom in GCV criterion.
    spline_order : int, default 3
        Degree of polynomial (used for 'ps' and 'bs').
    constraint : str or None, default None
        Optional shape constraint. Supports None or 'periodic' (sets bs='cc').
    """

    def __init__(
        self,
        n_splines: int = 10,
        lam: Optional[float] = None,
        spline_order: int = 3,
        constraint: Optional[str] = None,
        k: Optional[int] = None,
        sp: Optional[float] = None,
        bs: str = "tp",
        method: str = "REML",
        gamma: float = 1.0,
    ) -> None:
        if k is not None:
            n_splines = int(k)
        if sp is not None:
            lam = float(sp)

        if constraint == "periodic":
            bs = "cc"
        elif constraint is not None:
            raise ValueError(
                f"Unsupported constraint '{constraint}' for mgcv backend. "
                "mgcv natively supports 'periodic' (bs='cc'). For soft shape "
                "constraints (monotonicity, convexity), use backend='pygam'."
            )

        if n_splines < 3:
            raise ValueError("n_splines (k) must be at least 3.")
        if lam is not None and lam < 0:
            raise ValueError("Smoothing parameter 'lam' / 'sp' must be non-negative.")

        self.n_splines = int(n_splines)
        self.k = self.n_splines
        self.lam = float(lam) if lam is not None else None
        self.sp = self.lam
        self.spline_order = int(spline_order)
        self.constraint = constraint
        self.bs = bs.lower()
        self.method = method.upper()
        self.gamma = float(gamma)

        self.edge_knots_: Optional[np.ndarray] = None
        self.knots_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.design_matrix_: Optional[np.ndarray] = None
        self.statistics_: Optional[Dict[str, Any]] = None
        self._B_solve: Optional[np.ndarray] = None

        self._meta: Dict[str, Any] = {}
        self._Z: Optional[np.ndarray] = None
        self._S_abs: Optional[np.ndarray] = None

    @property
    def n_coefs(self) -> int:
        """Effective number of model coefficients (smooth + intercept)."""
        if self.coef_ is not None:
            return len(self.coef_)
        return self.n_splines

    @property
    def is_fitted(self) -> bool:
        return self.coef_ is not None

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "This LinGAM instance is not fitted yet. "
                "Call 'fit' or 'gridsearch' first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        sp_str = f"sp={self.lam:.4g}" if self.lam is not None else "sp=auto"
        cons_str = f", constraint='{self.constraint}'" if self.constraint else ""
        return (
            f"LinGAM(k={self.n_splines}, bs='{self.bs}', {sp_str}, "
            f"method='{self.method}'{cons_str}) [{status}]"
        )

    # ── design & penalty ─────────────────────────────────────────

    def _compile_basis(self, x: np.ndarray, n_splines: Optional[int] = None) -> None:
        k_val = n_splines if n_splines is not None else self.n_splines
        m_param: Any = None
        if self.bs in ("ps", "cp", "bs", "b"):
            m_param = (self.spline_order - 1, 2)
        X_raw, S_list, meta = build_smooth_basis(x, bs=self.bs, k=k_val, m=m_param)
        S_scaled, _ = rescale_penalty(X_raw, S_list)
        _, S_abs_list, Z = absorb_sum_to_zero_constraint(X_raw, S_scaled)
        self._meta = meta
        self._Z = Z
        self._S_abs = S_abs_list[0] if S_abs_list else np.zeros((Z.shape[1], Z.shape[1]))

    def _build_design_matrix(
        self,
        x: np.ndarray,
        n_splines: Optional[int] = None,
    ) -> np.ndarray:
        if self._Z is None or n_splines is not None:
            self._compile_basis(x, n_splines=n_splines)
        X_raw = predict_smooth_basis(x, self._meta)
        X_smooth = X_raw @ self._Z
        # Model matrix: [X_smooth, 1]
        return np.hstack([X_smooth, np.ones((len(x), 1))])

    def _build_penalty(self) -> np.ndarray:
        k_smooth = self._Z.shape[1]
        m = k_smooth + 1
        P = np.zeros((m, m), dtype=float)
        P[:k_smooth, :k_smooth] = self._S_abs
        return P

    # ── fit ──────────────────────────────────────────────────────

    def fit(self, x: ArrayLike, y: ArrayLike) -> "LinGAM":
        """Fit the GAM model using mgcv REML or GCV criteria.

        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Training feature.
        y : array-like of shape (n_samples,)
            Target values.
        """
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("Input data contains NaN or infinity.")

        self.edge_knots_ = np.array([float(np.min(x_arr)), float(np.max(x_arr))])
        self._compile_basis(x_arr)
        X = self._build_design_matrix(x_arr)
        self.design_matrix_ = X
        P = self._build_penalty()

        k_smooth = self._Z.shape[1]
        sp_fixed = [self.lam] if self.lam is not None else [None]

        fit_res = fit_mgcv_engine(
            X=X,
            y=y_arr,
            S_list=[P],
            sp_fixed=sp_fixed,
            method=self.method,
            gamma=self.gamma,
            coef_slices=[(0, k_smooth)],
            M_p=1,
        )

        self.coef_ = fit_res["coef"]
        self.lam = float(fit_res["sp"][0])
        self.sp = self.lam
        self._B_solve = fit_res["H_inv"] @ X.T

        # Store knots metadata
        xp = self._meta.get("xp")
        if xp is None:
            xp = self._meta.get("knots")
        if xp is None:
            xp = self._meta.get("Xu")
        self.knots_ = np.asarray(xp) if xp is not None else np.linspace(self.edge_knots_[0], self.edge_knots_[1], self.n_splines)

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
            "f_stat_per_term": fit_res["f_stat_per_term"],
            "p_values_per_term": fit_res["p_values_per_term"],
            "edf_per_term": fit_res["edf_per_term"],
        }
        return self

    # ── grid search ──────────────────────────────────────────────

    def gridsearch(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_splines: Optional[ArrayLike] = None,
        lam: Optional[ArrayLike] = None,
        gamma: float = 1.0,
    ) -> "LinGAM":
        """Select optimal basis dimension k (and optionally lam) via REML or GCV.

        In mgcv, smoothing parameter lambda is estimated automatically for each
        basis dimension k.

        Parameters
        ----------
        x, y : array-like
            Training data.
        n_splines : array-like of int, optional
            Candidate basis dimensions. Default: np.arange(5, 25, 2).
        lam : array-like of float, optional
            If provided, tests combinations of fixed lam.
        gamma : float, default 1.0
            Degrees-of-freedom multiplier.
        """
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        self.edge_knots_ = np.array([float(np.min(x_arr)), float(np.max(x_arr))])

        if n_splines is None:
            n_splines = np.arange(5, 25, 2)
        n_splines_list = [int(v) for v in np.atleast_1d(n_splines)]

        best_score = np.inf
        best_k = self.n_splines
        best_lam = self.lam

        if lam is None:
            # Auto-estimate lambda for each candidate basis dimension k
            for k_cand in n_splines_list:
                cand_model = LinGAM(
                    n_splines=k_cand,
                    lam=None,
                    bs=self.bs,
                    method=self.method,
                    gamma=gamma,
                    spline_order=self.spline_order,
                )
                try:
                    cand_model.fit(x_arr, y_arr)
                    score = cand_model.statistics_["score"]
                    if score < best_score:
                        best_score = score
                        best_k = k_cand
                        best_lam = cand_model.lam
                except Exception:
                    continue
        else:
            lam_list = [float(v) for v in np.atleast_1d(lam)]
            for k_cand in n_splines_list:
                for l_cand in lam_list:
                    cand_model = LinGAM(
                        n_splines=k_cand,
                        lam=l_cand,
                        bs=self.bs,
                        method=self.method,
                        gamma=gamma,
                        spline_order=self.spline_order,
                    )
                    try:
                        cand_model.fit(x_arr, y_arr)
                        score = cand_model.statistics_["score"]
                        if score < best_score:
                            best_score = score
                            best_k = k_cand
                            best_lam = l_cand
                    except Exception:
                        continue

        self.n_splines = best_k
        self.k = best_k
        self.lam = best_lam
        self.sp = best_lam
        self.fit(x_arr, y_arr)
        return self

    # ── predict ──────────────────────────────────────────────────

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict target values for new data points x."""
        self._check_is_fitted()
        x_arr = np.asarray(x, dtype=float).ravel()
        X = self._build_design_matrix(x_arr)
        return X @ self.coef_

    # ── intervals ────────────────────────────────────────────────

    def confidence_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        """Compute Bayesian confidence intervals for the linear predictor."""
        return self._compute_intervals(x, width, prediction=False)

    def prediction_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        """Compute prediction intervals for new observations."""
        return self._compute_intervals(x, width, prediction=True)

    def _compute_intervals(self, x: ArrayLike, width: float, prediction: bool) -> np.ndarray:
        self._check_is_fitted()
        x_arr = np.asarray(x, dtype=float).ravel()
        X = self._build_design_matrix(x_arr)

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
        self._check_is_fitted()
        n = self.statistics_["n_samples"]
        rss = self.statistics_["rss"]
        scale = self.statistics_["scale"]
        return -0.5 * n * np.log(2.0 * np.pi * scale) - 0.5 * rss / scale

    def aic(self) -> float:
        """Akaike Information Criterion (AIC)."""
        self._check_is_fitted()
        return -2.0 * self.loglikelihood() + 2.0 * (self.statistics_["edof"] + 1)

    def bic(self) -> float:
        """Bayesian Information Criterion (BIC)."""
        self._check_is_fitted()
        n = self.statistics_["n_samples"]
        k = self.statistics_["edof"] + 1
        return -2.0 * self.loglikelihood() + np.log(n) * k

    def deviance_explained(self) -> float:
        """Fraction of null deviance explained (pseudo R-squared)."""
        self._check_is_fitted()
        return self.statistics_["deviance_explained"]

    # ── summary ──────────────────────────────────────────────────

    def summary(self) -> str:
        """Return formatted summary matching R mgcv's summary.gam output."""
        self._check_is_fitted()
        n = self.statistics_["n_samples"]
        k_smooth = self._Z.shape[1]
        beta_0 = self.coef_[-1]
        se_0 = self.statistics_["se"][-1]
        t_val = beta_0 / max(se_0, 1e-12)
        p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_val), df=max(n - self.statistics_["edof"], 1)))

        term_edf = self.statistics_["edf_per_term"][0] if self.statistics_["edf_per_term"] else self.statistics_["edof"] - 1.0
        f_stat = self.statistics_["f_stat_per_term"][0] if self.statistics_["f_stat_per_term"] else 0.0
        term_pval = self.statistics_["p_values_per_term"][0] if self.statistics_["p_values_per_term"] else 1.0

        score_name = "REML" if self.method == "REML" else "GCV"
        score_val = self.statistics_["score"]

        lines = [
            "",
            "Family: gaussian ",
            "Link function: identity ",
            "",
            f"Formula: y ~ s(x, bs='{self.bs}', k={self.n_splines})",
            "",
            "Parametric coefficients:",
            f"            Estimate Std. Error t value Pr(>|t|)",
            f"(Intercept) {beta_0:10.4f} {se_0:10.4f} {t_val:7.3f} {p_val:8.4g}",
            "",
            "Approximate significance of smooth terms:",
            f"        edf Ref.df      F p-value",
            f"s(x) {term_edf:6.2f} {self.n_splines - 1:6d} {f_stat:6.3f} {term_pval:8.4g}",
            "",
            f"R-sq.(adj) = {self.statistics_['r_sq_adj']:.4f}   Deviance explained = {self.statistics_['deviance_explained'] * 100:.1f}%",
            f"{score_name} = {score_val:10.4f}  Scale est. = {self.statistics_['scale']:.4f}   n = {n}",
            "",
        ]
        out_str = "\n".join(lines)
        return out_str

    # ── plotting ─────────────────────────────────────────────────

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 300,
        figsize: Tuple[int, int] = (16, 5),
    ) -> Tuple["plt.Figure", np.ndarray]:
        """Plot basis decomposition, coefficients, and overall fit."""
        self._check_is_fitted()
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        k_smooth = self._Z.shape[1]

        x_grid = np.linspace(np.min(x_arr), np.max(x_arr), n_grid)
        X_grid = self._build_design_matrix(x_grid)
        B_grid = X_grid[:, :k_smooth]
        y_pred = X_grid @ self.coef_

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Basis functions
        ax1 = axes[0]
        cmap = plt.get_cmap("tab20" if k_smooth <= 20 else "hsv")
        colors = [cmap(i % 20 if k_smooth <= 20 else i / k_smooth) for i in range(k_smooth)]
        for j in range(k_smooth):
            ax1.plot(x_grid, B_grid[:, j], color=colors[j], alpha=0.8, label=f"b_{j}")
        ax1.set(xlabel="x", ylabel="B_j(x)", title=f"Basis Functions (dim={k_smooth})")
        if k_smooth <= 12:
            ax1.legend(fontsize=7, loc="upper right")

        # 2. Coefficients
        ax2 = axes[1]
        vals = np.append(self.coef_[:k_smooth], self.coef_[-1])
        labels = [f"b_{j}" for j in range(k_smooth)] + ["intercept"]
        ax2.bar(range(len(vals)), vals, color=colors + ["dimgray"])
        ax2.set_xticks(range(len(vals)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax2.axhline(0, color="k")
        ax2.set(ylabel="Coefficient value", title="Estimated Coefficients")

        # 3. Fit and data
        ax3 = axes[2]
        ax3.scatter(x_arr, y_arr, alpha=0.5, color="gray", s=12, label="Data")
        ax3.plot(x_grid, y_pred, "k-", lw=2, label="GAM fit")
        ax3.set(xlabel="x", ylabel="y", title="Overall Model Fit")
        ax3.legend(fontsize=7, loc="upper left")

        fig.suptitle(
            f"LinGAM Decomposition (k={self.n_splines}, bs='{self.bs}', sp={self.lam:.4f}, method={self.method})"
        )
        return fig, axes

    def plot_diagnostics(
        self,
        x: ArrayLike,
        y: ArrayLike,
        figsize: Tuple[int, int] = (10, 9),
    ) -> Tuple["plt.Figure", np.ndarray]:
        """Diagnostic plots for model validation."""
        self._check_is_fitted()
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()

        y_fit = self.predict(x_arr)
        residuals = y_arr - y_fit
        r2 = self.deviance_explained()

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

        fig.suptitle(
            f"LinGAM Diagnostics (k={self.n_splines}, bs='{self.bs}', sp={self.lam:.4f})",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig, axes
