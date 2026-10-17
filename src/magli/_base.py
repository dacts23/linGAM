"""Shared base class, plotting, and tensor helpers for GAM backends."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def _validate_weights(
    weights: Optional[ArrayLike], n: int
) -> Optional[np.ndarray]:
    """Validate and return 1D observation weights array, or None."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float).ravel()
    if len(w) != n:
        raise ValueError(
            f"weights length ({len(w)}) must match n_samples ({n})."
        )
    if not np.all(np.isfinite(w)) or np.any(w < 0) or np.all(w == 0):
        raise ValueError(
            "weights must be finite, non-negative, and not all zero."
        )
    return w


def _khatri_rao(marginals: List[np.ndarray]) -> np.ndarray:
    """Row-wise Kronecker (Khatri-Rao) product of marginal basis matrices."""
    n = marginals[0].shape[0]
    result = marginals[0]
    for B_next in marginals[1:]:
        result = (result[:, :, None] * B_next[:, None, :]).reshape(n, -1)
    return result


def _kron_sum(penalties: List[np.ndarray]) -> np.ndarray:
    """Kronecker sum: sum_i I_0 kron ... kron P_i kron ... kron I_{d-1}."""
    sizes = [p.shape[0] for p in penalties]
    total = int(np.prod(sizes))
    result = np.zeros((total, total))
    for i, P_i in enumerate(penalties):
        I_pre = np.eye(int(np.prod(sizes[:i])))
        I_post = np.eye(int(np.prod(sizes[i + 1 :])))
        result += np.kron(np.kron(I_pre, P_i), I_post)
    return result


class _GAMBase:
    """Shared lifecycle, prediction, interval, criteria, and plotting methods."""

    _scale_is_var: bool = False
    formula: str = ""
    fit_intercept: bool = True
    _terms: List[Any] = []
    _coef_slices: List[Tuple[int, int]] = []
    coef_: Optional[np.ndarray] = None
    statistics_: Optional[Dict[str, Any]] = None
    _design_matrix_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.coef_ is not None

    @property
    def n_terms(self) -> int:
        return len(self._terms)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or gridsearch() first.")

    _check_is_fitted = _check_fitted

    def _compile(self, x: np.ndarray) -> None:
        for term in self._terms:
            term.compile(x)

    def _compute_coef_slices(self) -> None:
        slices = []
        offset = 0
        for t in self._terms:
            slices.append((offset, offset + t.n_coefs))
            offset += t.n_coefs
        self._coef_slices = slices

    def _build_model_matrix(self, x: np.ndarray) -> np.ndarray:
        blocks = [t.build_columns(x) for t in self._terms]
        if self.fit_intercept:
            blocks.append(np.ones((len(x), 1), dtype=float))
        return np.hstack(blocks)

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict target values for feature data x."""
        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        return self._build_model_matrix(x_arr) @ self.coef_

    def partial_dependence(self, term_idx: int, x: ArrayLike) -> np.ndarray:
        """Evaluate the partial contribution of term term_idx."""
        self._check_fitted()
        if term_idx < 0 or term_idx >= len(self._terms):
            raise IndexError(
                f"term_idx {term_idx} out of range (0..{len(self._terms) - 1})."
            )
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        B_term = self._terms[term_idx].build_columns(x_arr)
        start, end = self._coef_slices[term_idx]
        return B_term @ self.coef_[start:end]

    def confidence_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        """Compute confidence intervals for the linear predictor."""
        return self._compute_intervals(x, width, prediction=False)

    def prediction_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        """Compute prediction intervals for new observations."""
        return self._compute_intervals(x, width, prediction=True)

    def _compute_intervals(
        self, x: ArrayLike, width: float, prediction: bool
    ) -> np.ndarray:
        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        X = self._build_model_matrix(x_arr)
        lp = X @ self.coef_

        cov = self.statistics_["cov"]
        var = np.sum((X @ cov) * X, axis=1)
        if prediction:
            s = self.statistics_["scale"]
            var = var + (s if self._scale_is_var else s**2)

        alpha = (1.0 - width) / 2.0
        df = max(self.statistics_["n_samples"] - self.statistics_["edof"], 1.0)
        q_lo = stats.t.ppf(alpha, df=df)
        q_hi = stats.t.ppf(1.0 - alpha, df=df)

        se = np.sqrt(np.maximum(var, 1e-14 if self._scale_is_var else 0.0))
        return np.column_stack([lp + q_lo * se, lp + q_hi * se])

    def loglikelihood(self) -> float:
        """Gaussian log-likelihood of the fitted model."""
        self._check_fitted()
        n = self.statistics_["n_samples"]
        rss = self.statistics_["rss"]
        s = self.statistics_["scale"]
        s2 = s if self._scale_is_var else s**2
        return float(-0.5 * n * np.log(2.0 * np.pi * s2) - 0.5 * rss / s2)

    def aic(self) -> float:
        """Akaike Information Criterion (AIC)."""
        self._check_fitted()
        return float(-2.0 * self.loglikelihood() + 2.0 * (self.statistics_["edof"] + 1))

    def bic(self) -> float:
        """Bayesian Information Criterion (BIC)."""
        self._check_fitted()
        n = self.statistics_["n_samples"]
        return float(
            -2.0 * self.loglikelihood() + np.log(n) * (self.statistics_["edof"] + 1)
        )

    def deviance_explained(self) -> float:
        """Fraction of null deviance explained (pseudo R-squared)."""
        self._check_fitted()
        null_dev = self.statistics_.get("null_deviance")
        if not null_dev:
            return 0.0
        return float(max(0.0, 1.0 - self.statistics_["rss"] / null_dev))

    def _diagnostics_title(self) -> str:
        return f"{type(self).__name__} Diagnostics — {self.formula}"

    def _overall_plot_title(self) -> str:
        s = self.statistics_
        if self._scale_is_var:
            return f"GAM Fit (Score={s['score']:.3f}, EDF={s['edof']:.1f}, n={s['n_samples']})"
        constrained = [
            f"s({t.feature})={t.constraint}"
            for t in self._terms
            if getattr(t, "constraint", None)
        ]
        c_info = f"  constraints: {', '.join(constrained)}" if constrained else ""
        return f"GAM Fit  (GCV={s['GCV']:.3f}, EDF={s['edof']:.1f}, n={s['n_samples']}){c_info}"

    def _plot_gamcore_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 200,
        extrap_frac: float = 0.0,
        figsize: Optional[Tuple[int, int]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[Any, np.ndarray]:
        import matplotlib.pyplot as plt

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

        fig = plt.figure(figsize=figsize or (n_cols * 5, n_rows * 4))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)
        ax_overall = fig.add_subplot(gs[0, :])

        feat_mins = np.min(x_arr, axis=0)
        feat_maxs = np.max(x_arr, axis=0)
        medians = np.median(x_arr, axis=0)

        feat_grids = []
        for j in range(p_predictors):
            rng = (feat_maxs[j] - feat_mins[j]) or 1.0
            feat_grids.append(
                np.linspace(
                    feat_mins[j] - extrap_frac * rng,
                    feat_maxs[j] + extrap_frac * rng,
                    n_grid,
                )
            )

        x_overall = np.tile(medians, (n_grid, 1))
        x_overall[:, 0] = feat_grids[0]
        y_overall = self.predict(x_overall)
        sort_idx = np.argsort(x_overall[:, 0])
        ax_overall.plot(
            x_overall[sort_idx, 0], y_overall[sort_idx], "C0-", lw=2, label="GAM fit"
        )
        ax_overall.scatter(
            x_arr[:, 0], y_arr, alpha=0.4, color="gray", s=10, label="Data"
        )
        if extrap_frac > 0:
            ax_overall.axvspan(feat_grids[0][0], feat_mins[0], alpha=0.06, color="red")
            ax_overall.axvspan(
                feat_maxs[0],
                feat_grids[0][-1],
                alpha=0.06,
                color="red",
                label="Extrapolated",
            )
        ax_overall.set(
            xlabel=_fname(0) if p_predictors > 1 else "x",
            ylabel="y",
            title=self._overall_plot_title(),
        )
        ax_overall.legend(fontsize=7, loc="best")

        for i, term in enumerate(self._terms):
            ax = fig.add_subplot(gs[1 + i // n_term_cols, i % n_term_cols])
            tname = type(term).__name__

            if "SplineTerm" in tname:
                x_sweep = np.tile(medians, (n_grid, 1))
                fv = feat_grids[term.feature]
                x_sweep[:, term.feature] = fv
                pdep = self.partial_dependence(i, x_sweep)
                cons = getattr(term, "constraint", None)
                label = f"s({term.feature})" + (f" [{cons}]" if cons else "")
                ax.plot(fv, pdep, "C1-", lw=2, label=label)
                if extrap_frac > 0:
                    ax.axvspan(fv[0], feat_mins[term.feature], alpha=0.06, color="red")
                    ax.axvspan(
                        feat_maxs[term.feature], fv[-1], alpha=0.06, color="red"
                    )
                ax.axhline(0, color="gray", ls="--", lw=0.5)
                if hasattr(term, "n_splines"):
                    title = (
                        f"s({term.feature})   K={term.n_splines}, lam={term.lam:.3f}"
                        + (f"  [{cons}]" if cons else "")
                    )
                else:
                    sp_val = term.sp[0] if term.sp else 0.0
                    title = f"s({term.feature})  k={term.k}, sp={sp_val:.3g}"
                ax.set(xlabel=_fname(term.feature), ylabel="Contribution", title=title)
                ax.legend(fontsize=7)

            elif "TensorTerm" in tname:
                if term.d == 2:
                    ng2d = min(n_grid, 60 if self._scale_is_var else 75)
                    f0_idx, f1_idx = term.features[0], term.features[1]
                    g0 = np.linspace(
                        feat_grids[f0_idx][0], feat_grids[f0_idx][-1], ng2d
                    )
                    g1 = np.linspace(
                        feat_grids[f1_idx][0], feat_grids[f1_idx][-1], ng2d
                    )
                    f0, f1 = np.meshgrid(g0, g1, indexing="ij")
                    x_2d = np.tile(medians, (ng2d * ng2d, 1))
                    x_2d[:, f0_idx] = f0.ravel()
                    x_2d[:, f1_idx] = f1.ravel()
                    pdep_2d = self.partial_dependence(i, x_2d).reshape(ng2d, ng2d)
                    im = ax.pcolormesh(f0, f1, pdep_2d, shading="auto", cmap="RdBu_r")
                    plt.colorbar(im, ax=ax, label="Contribution")
                    ax.set(xlabel=_fname(f0_idx), ylabel=_fname(f1_idx))
                else:
                    x_sweep = np.tile(medians, (n_grid, 1))
                    fv = feat_grids[term.features[0]]
                    x_sweep[:, term.features[0]] = fv
                    ax.plot(fv, self.partial_dependence(i, x_sweep), "C2-", lw=1.5)
                    ax.set_xlabel(f"{_fname(term.features[0])} (projection)")
                feats_str = ",".join(str(f) for f in term.features)
                if hasattr(term, "n_splines_per"):
                    ks_str = ",".join(str(k) for k in term.n_splines_per)
                    ax.set_title(
                        f"te({feats_str})   K=({ks_str})   {term.n_coefs} coefs"
                    )
                else:
                    ax.set_title(f"te({feats_str})")

            elif "FactorTerm" in tname:
                unique_lvls = np.unique(np.asarray(x_arr[:, term.feature]).ravel())
                contribs = []
                for lvl in unique_lvls:
                    x_lvl = medians.copy().reshape(1, -1)
                    x_lvl[0, term.feature] = lvl
                    contribs.append(self.partial_dependence(i, x_lvl)[0])
                colors = ["C2" if v >= 0 else "C3" for v in contribs]
                ax.bar(
                    range(len(contribs)),
                    contribs,
                    color=colors,
                    edgecolor="k",
                    lw=0.5,
                )
                ax.axhline(0, color="gray", ls="-", lw=0.5)
                ax.set_xticks(range(len(contribs)))
                ax.set_xticklabels(unique_lvls, fontsize=8)
                title = (
                    f"f({term.feature})"
                    if self._scale_is_var
                    else f"f({term.feature})   lam={term.lam:.3f}"
                )
                ax.set(ylabel="Contribution", title=title)

            elif "LinearTerm" in tname:
                x_sweep = np.tile(medians, (n_grid, 1))
                fv = feat_grids[term.feature]
                x_sweep[:, term.feature] = fv
                beta = self.coef_[self._coef_slices[i][0]]
                ax.plot(fv, self.partial_dependence(i, x_sweep), "C4-", lw=2)
                if extrap_frac > 0:
                    ax.axvspan(fv[0], feat_mins[term.feature], alpha=0.06, color="red")
                    ax.axvspan(
                        feat_maxs[term.feature], fv[-1], alpha=0.06, color="red"
                    )
                ax.axhline(0, color="gray", ls="--", lw=0.5)
                ax.set(
                    xlabel=_fname(term.feature),
                    ylabel="Contribution",
                    title=f"l({term.feature})   beta={beta:.4f}",
                )

        fig.suptitle(
            f"GAMCore Decomposition — {self.formula}",
            fontsize=12,
            fontweight="bold",
        )
        return fig, np.array(fig.axes)

    def _plot_lingam_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        k_smooth: int,
        prefix: str,
        basis_title: str,
        fit_title: str,
        suptitle: str,
        knots: Optional[np.ndarray] = None,
        show_weighted: bool = False,
        n_grid: int = 300,
        figsize: Tuple[int, int] = (16, 5),
    ) -> Tuple[Any, np.ndarray]:
        import matplotlib.pyplot as plt

        self._check_fitted()
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()

        x_grid = np.linspace(np.min(x_arr), np.max(x_arr), n_grid)
        X_grid = self._build_model_matrix(x_grid.reshape(-1, 1))
        B_grid = X_grid[:, :k_smooth]
        y_pred = X_grid @ self.coef_

        cmap = plt.get_cmap(
            "tab20" if k_smooth <= 20 else "tab10" if k_smooth <= 10 else "hsv"
        )
        colors = [
            cmap(i % 20 if k_smooth <= 20 else i / k_smooth)
            for i in range(k_smooth)
        ]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax1 = axes[0]
        for j in range(k_smooth):
            ax1.plot(
                x_grid,
                B_grid[:, j],
                color=colors[j],
                ls="-",
                alpha=0.8,
                label=f"{prefix}_{j}",
            )
        if knots is not None and show_weighted:
            for i_knot, knot in enumerate(knots):
                ax1.axvline(
                    knot,
                    color="gray",
                    ls="--",
                    lw=1,
                    label="Knots" if i_knot == 0 else None,
                )
            ax1.minorticks_on()
        ax1.set(xlabel="x", ylabel="B_j(x)", title=basis_title)
        if show_weighted or k_smooth <= 12:
            ax1.legend(fontsize=6 if show_weighted else 7, loc="upper right")

        ax2 = axes[1]
        vals = np.append(self.coef_[:k_smooth], self.coef_[-1])
        labels = [f"{prefix}_{j}" for j in range(k_smooth)] + ["intercept"]
        ax2.bar(range(len(vals)), vals, color=colors + ["dimgray"])
        ax2.set_xticks(range(len(vals)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax2.axhline(0, color="k")
        ax2.set(ylabel="Coefficient value", title="Estimated Coefficients")

        ax3 = axes[2]
        ax3.scatter(x_arr, y_arr, alpha=0.5, color="gray", s=12, label="Data")
        if show_weighted:
            for j in range(k_smooth):
                ax3.plot(
                    x_grid,
                    self.coef_[j] * B_grid[:, j] + self.coef_[-1],
                    color=colors[j],
                    lw=1,
                    ls="-",
                )
            ax3.plot(x_grid, y_pred, "k-", label="Fit (sum)")
            if knots is not None:
                for knot in knots:
                    ax3.axvline(knot, color="gray", ls="--", lw=1)
            ax3.minorticks_on()
            ax3.legend(fontsize=6, loc="upper left")
        else:
            ax3.plot(x_grid, y_pred, "k-", lw=2, label="GAM fit")
            ax3.legend(fontsize=7, loc="upper left")
        ax3.set(xlabel="x", ylabel="y", title=fit_title)

        fig.suptitle(suptitle)
        return fig, axes

    def plot_diagnostics(
        self,
        x: ArrayLike,
        y: ArrayLike,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Any, np.ndarray]:
        """Diagnostic plots for model validation."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        y_fit = self.predict(x_arr)
        residuals = y_arr - y_fit
        r2 = self.deviance_explained()

        fig, axes = plt.subplots(2, 2, figsize=figsize or (10, 9))
        axes = axes.ravel()

        ax = axes[0]
        ax.scatter(y_fit, y_arr, alpha=0.5, edgecolors="k", s=20)
        lims = [min(y_arr.min(), y_fit.min()), max(y_arr.max(), y_fit.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="1:1")
        ax.set(
            xlabel="Fitted values",
            ylabel="Observed values",
            title=f"Observed vs Fitted ($R^2$={r2:.3f})",
        )
        ax.legend()

        ax = axes[1]
        ax.scatter(y_fit, residuals, alpha=0.5, edgecolors="k", s=20)
        ax.axhline(0, color="r", linestyle="--", lw=1.5)
        ax.set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs Fitted")

        ax = axes[2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q")
        ax.get_lines()[0].set_markerfacecolor("steelblue")
        ax.get_lines()[0].set_markeredgecolor("k")
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[1].set_color("r")

        ax = axes[3]
        ax.hist(
            residuals,
            bins="auto",
            density=True,
            color="steelblue",
            edgecolor="k",
            alpha=0.7,
        )
        sigma = np.std(residuals, ddof=1)
        x_norm = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(
            x_norm,
            stats.norm.pdf(x_norm, 0.0, sigma),
            "r-",
            lw=2,
            label="Normal approx.",
        )
        ax.set(xlabel="Residuals", ylabel="Density", title="Residual Distribution")
        ax.legend()

        fig.suptitle(self._diagnostics_title(), fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig, axes
