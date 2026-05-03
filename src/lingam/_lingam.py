"""Single-term penalised B-spline GAM — LinGAM class."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

from ._bspline import b_spline_basis
from ._penalty import (
    circular_second_diff_penalty,
    first_diff_penalty,
    second_diff_penalty,
)


class LinGAM:
    """Penalised B-spline GAM for a single smooth term y = f(x) + eps.

    Parameters
    ----------
    n_splines : int, default 10
        Number of B-spline basis functions K. Must satisfy K > spline_order.
    lam : float, default 1.0
        Smoothing parameter (lambda >= 0).
    spline_order : int, default 3
        B-spline order (polynomial degree).
    constraint : str or None, default None
        Shape constraint for the smooth term. One of ``None``,
        ``'mono_inc'``, ``'mono_dec'``, ``'convex'``, ``'concave'``,
        ``'periodic'``.
    """

    def __init__(
        self,
        n_splines: int = 10,
        lam: float = 1.0,
        spline_order: int = 3,
        constraint: Optional[str] = None,
    ) -> None:
        valid_constraints = {None, 'mono_inc', 'mono_dec', 'convex', 'concave', 'periodic'}
        if constraint not in valid_constraints:
            raise ValueError(
                f"Unknown constraint '{constraint}'. "
                f"Choose from: {sorted(c for c in valid_constraints if c is not None)}."
            )
        if n_splines <= spline_order:
            raise ValueError("n_splines must be greater than spline_order.")
        if lam < 0:
            raise ValueError("Smoothing parameter 'lam' must be non-negative.")
        if spline_order < 1:
            raise ValueError("spline_order must be >= 1.")

        self.n_splines = n_splines
        self.lam = lam
        self.spline_order = spline_order
        self.constraint = constraint
        self.edge_knots_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.design_matrix_: Optional[np.ndarray] = None
        self.statistics_: Optional[Dict[str, Any]] = None
        self._B_solve: Optional[np.ndarray] = None
        self.knots_: Optional[np.ndarray] = None

    @property
    def n_coefs(self) -> int:
        """Effective number of spline coefficients (reduced for periodic)."""
        if self.constraint == 'periodic':
            return self.n_splines - self.spline_order + 1
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
        cons = f", constraint='{self.constraint}'" if self.constraint else ""
        return (
            f"LinGAM(n_splines={self.n_splines}, lam={self.lam:.4g}, "
            f"spline_order={self.spline_order}{cons}) [{status}]"
        )

    # ── design & penalty ─────────────────────────────────────────

    def _build_design_matrix(
        self,
        x: np.ndarray,
        n_splines: Optional[int] = None,
    ) -> np.ndarray:
        K = n_splines if n_splines is not None else self.n_splines
        B = b_spline_basis(x, K, self.spline_order, self.edge_knots_,
                           periodic=self.constraint == 'periodic')
        return np.hstack([B, np.ones((len(x), 1))])

    def _build_penalty(
        self,
        lam: float,
        n_splines: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = n_splines if n_splines is not None else self.n_splines
        K_eff = K - self.spline_order + 1 if self.constraint == 'periodic' else K
        m = K_eff + 1

        if self.constraint in ('mono_inc', 'mono_dec'):
            S_pen = first_diff_penalty(K_eff, 1.0)
        elif self.constraint == 'periodic':
            S_pen = circular_second_diff_penalty(K_eff, 1.0)
        else:
            D2 = np.zeros((K_eff - 2, K_eff))
            idx = np.arange(K_eff - 2)
            D2[idx, idx] = 1.0
            D2[idx, idx + 1] = -2.0
            D2[idx, idx + 2] = 1.0
            S_pen = D2.T @ D2

        P = np.zeros((m, m))
        P[:K_eff, :K_eff] = lam * S_pen
        return P, S_pen

    # ── solver ───────────────────────────────────────────────────

    def _solve_pirls(
        self,
        X: np.ndarray,
        y: np.ndarray,
        P: np.ndarray,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _irls_solve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        P: np.ndarray,
        n_iter: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(y)
        med_y = np.median(y)
        residuals = y - med_y
        w = np.ones(n)
        coef, U1, B_solve = None, None, None

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
            coef, U1, B_solve = self._solve_pirls(X_w, y_w, P, Q=Qw, R=Rw)
            residuals = y - X @ coef

        return coef, U1, B_solve, w, residuals

    # ── grid search ──────────────────────────────────────────────

    def gridsearch(
        self,
        x: ArrayLike,
        y: ArrayLike,
        lam: Optional[ArrayLike] = None,
        n_splines: Optional[ArrayLike] = None,
        gamma: float = 1.4,
        robust: bool = False,
    ) -> "LinGAM":
        """Select optimal (lam, n_splines) by minimising GCV.

        Always uses the fast QR+Cholesky path for standard fits.
        When robust=True, uses Huber IRLS within each candidate
        evaluation.
        """
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        if lam is None:
            lam = np.logspace(-3, 3, 11)
        if n_splines is None:
            n_splines = np.arange(5, 25, 2)

        self.edge_knots_ = np.array([x.min(), x.max()])
        n = len(y)

        if robust:
            return self._gridsearch_robust(x, y, lam, n_splines, gamma)
        return self._gridsearch(x, y, lam, n_splines, gamma)

    def _gridsearch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam: np.ndarray,
        n_splines: np.ndarray,
        gamma: float,
    ) -> "LinGAM":
        n = len(y)
        best_gcv = np.inf
        best_result: Optional[Tuple[float, int, float, np.ndarray, float]] = None

        def _precompute_K(K_int: int):
            K = int(K_int)
            K_eff = K - self.spline_order + 1 if self.constraint == 'periodic' else K
            Xmat = self._build_design_matrix(x, n_splines=K)
            Q, R = np.linalg.qr(Xmat)
            _, S_pen = self._build_penalty(1.0, n_splines=K)
            m = K_eff + 1
            RtR = R.T @ R
            Rt_qty = R.T @ (Q.T @ y)
            P_base = np.zeros((m, m))
            P_base[:K_eff, :K_eff] = S_pen
            return K, K_eff, Xmat, P_base, RtR, Rt_qty, m

        with ThreadPoolExecutor() as executor:
            precomputed = list(executor.map(_precompute_K, n_splines))

        Xmats: Dict[int, np.ndarray] = {}
        precomputed_list = []
        for K, K_eff, Xmat, P_base, RtR, Rt_qty, m in precomputed:
            Xmats[K] = Xmat
            precomputed_list.append((K, K_eff, P_base, RtR, Rt_qty, m))

        lam_arr = np.atleast_1d(lam)

        def _eval_K(K_idx: int):
            K, K_eff, P_base, RtR, Rt_qty, m = precomputed_list[K_idx]
            best_gcv_K = np.inf
            best_l_K = float(lam_arr[0])
            best_coef_K = None
            best_edf_K = 0.0
            B_buf = np.empty((m, m))

            for l_val in lam_arr:
                np.add(RtR, float(l_val) * P_base, out=B_buf)
                B_buf.flat[::m + 1] += 1e-12

                try:
                    c_chol, lower = cho_factor(
                        B_buf, overwrite_a=True, check_finite=False,
                    )
                    coef = cho_solve((c_chol, lower), Rt_qty, check_finite=False)
                    B_inv_RtR = cho_solve(
                        (c_chol, lower), RtR, check_finite=False,
                    )
                    edf = float(np.trace(B_inv_RtR))
                except Exception:
                    B_fallback = RtR + float(l_val) * P_base
                    B_fallback.flat[::m + 1] += 1e-12
                    coef = np.linalg.solve(B_fallback, Rt_qty)
                    edf = float(np.trace(np.linalg.solve(B_fallback, RtR)))

                residuals = y - Xmats[K] @ coef
                rss = float(residuals @ residuals)
                gcv_score = (n * rss) / (n - gamma * edf) ** 2

                if gcv_score < best_gcv_K:
                    best_gcv_K = gcv_score
                    best_l_K = float(l_val)
                    best_coef_K = coef
                    best_edf_K = edf

            return best_gcv_K, K, best_l_K, best_coef_K, best_edf_K

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_eval_K, range(len(precomputed_list))))

        for gcv_score, K, l_val, coef, edf in results:
            if gcv_score < best_gcv:
                best_gcv = gcv_score
                best_result = (gcv_score, K, l_val, coef, edf)

        assert best_result is not None
        _, best_K, best_l, _, _ = best_result
        self.n_splines = best_K
        self.lam = best_l

        self._fit_internal(x, y, robust=False)
        return self

    def _gridsearch_robust(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam: np.ndarray,
        n_splines: np.ndarray,
        gamma: float,
    ) -> "LinGAM":
        n = len(y)
        best_gcv = np.inf
        best_result: Optional[Tuple] = None

        for K_int in n_splines:
            K = int(K_int)
            self.n_splines = K
            Xmat = self._build_design_matrix(x)
            _, S_pen = self._build_penalty(1.0)
            K_eff = self.n_coefs
            m = K_eff + 1
            P_buf = np.zeros((m, m))

            def _eval_lam(l_val: float):
                P_buf[:] = 0
                P_buf[:K_eff, :K_eff] = l_val * S_pen
                P_full = P_buf.copy()
                coef, U1, _, w, _ = self._irls_solve(Xmat, y, P_full, n_iter=10)
                residuals = y - Xmat @ coef
                rss = float(np.sum(w * residuals * residuals))
                edf = float(np.sum(U1 ** 2))
                gcv_score = (n * rss) / (n - gamma * edf) ** 2
                return gcv_score, K, l_val, coef, edf

            with ThreadPoolExecutor() as executor:
                for res in executor.map(
                    _eval_lam, (float(v) for v in lam),
                ):
                    if res[0] < best_gcv:
                        best_gcv = res[0]
                        best_result = res

        assert best_result is not None
        _, best_K, best_l, _, _ = best_result
        self.n_splines = best_K
        self.lam = best_l

        self._fit_internal(x, y, robust=True)
        return self

    # ── fit ──────────────────────────────────────────────────────

    def fit(self, x: ArrayLike, y: ArrayLike, robust: bool = False) -> "LinGAM":
        """Fit the model with current hyper-parameters."""
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        self.edge_knots_ = np.array([x.min(), x.max()])
        self._fit_internal(x, y, robust=robust)
        return self

    def _fit_internal(self, x: np.ndarray, y: np.ndarray, robust: bool = False) -> None:
        n = len(x)
        k = self.spline_order
        K = self.n_splines
        K_eff = self.n_coefs

        offset = self.edge_knots_[0]
        scale = self.edge_knots_[1] - self.edge_knots_[0]
        if scale == 0.0:
            scale = 1.0
        boundary_knots_norm = np.linspace(0.0, 1.0, 1 + K - k)
        self.knots_ = boundary_knots_norm * scale + offset

        X = self._build_design_matrix(x)
        self.design_matrix_ = X
        P, _ = self._build_penalty(self.lam)

        if robust:
            self.coef_, U1, B_solve, w, _ = self._irls_solve(X, y, P, n_iter=15)
            self._B_solve = B_solve
            y_hat = X @ self.coef_
            rss = np.sum(w * (y - y_hat) ** 2)
        else:
            self.coef_, U1, B_solve = self._solve_pirls(X, y, P)
            self._B_solve = B_solve
            y_hat = X @ self.coef_
            rss = np.sum((y - y_hat) ** 2)

        edof_per_coef = np.sum(U1 ** 2, axis=1)
        edof = np.sum(edof_per_coef)

        scale_val = np.sqrt(rss / max(n - edof, 1))
        cov = (B_solve @ B_solve.T) * scale_val ** 2
        se = np.sqrt(np.diag(cov))

        self.statistics_ = {
            'edof': edof,
            'edof_per_coef': edof_per_coef,
            'scale': scale_val,
            'cov': cov,
            'se': se,
            'rss': rss,
            'n_samples': n,
            'GCV': (n * rss) / max(n - 1.4 * edof, 1) ** 2,
        }

    # ── predict ──────────────────────────────────────────────────

    def predict(self, x: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        x = np.asarray(x).ravel()
        X = self._build_design_matrix(x)
        return X @ self.coef_

    # ── intervals ────────────────────────────────────────────────

    def confidence_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        return self._compute_intervals(x, width, prediction=False)

    def prediction_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        return self._compute_intervals(x, width, prediction=True)

    def _compute_intervals(self, x: np.ndarray, width: float, prediction: bool) -> np.ndarray:
        self._check_is_fitted()
        x = np.asarray(x).ravel()
        X = self._build_design_matrix(x)

        lp = X @ self.coef_
        cov = self.statistics_['cov']
        var = np.sum((X @ cov) * X, axis=1)

        if prediction:
            var += self.statistics_['scale'] ** 2

        alpha = (1.0 - width) / 2.0
        df = self.statistics_['n_samples'] - self.statistics_['edof']
        q_lo = stats.t.ppf(alpha, df=df)
        q_hi = stats.t.ppf(1.0 - alpha, df=df)

        se = np.sqrt(var)
        return np.column_stack([lp + q_lo * se, lp + q_hi * se])

    # ── statistics ────────────────────────────────────────────────

    def loglikelihood(self) -> float:
        """Gaussian log-likelihood."""
        self._check_is_fitted()
        n = self.statistics_['n_samples']
        rss = self.statistics_['rss']
        scale = self.statistics_['scale']
        return -0.5 * n * np.log(2 * np.pi * scale ** 2) - 0.5 * rss / scale ** 2

    def aic(self) -> float:
        """Akaike Information Criterion."""
        self._check_is_fitted()
        return -2 * self.loglikelihood() + 2 * (self.statistics_['edof'] + 1)

    def bic(self) -> float:
        """Bayesian Information Criterion."""
        self._check_is_fitted()
        n = self.statistics_['n_samples']
        k = self.statistics_['edof'] + 1
        return -2 * self.loglikelihood() + np.log(n) * k

    def deviance_explained(self) -> float:
        """Fraction of null deviance explained (pseudo R-squared)."""
        self._check_is_fitted()
        y = self.design_matrix_ @ self.coef_
        y_mean = np.mean(y)
        tss = np.sum((y - y_mean) ** 2) + self.statistics_['rss']
        if tss == 0:
            return 0.0
        return max(0.0, 1.0 - self.statistics_['rss'] / tss)

    # ── plotting ─────────────────────────────────────────────────

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 300,
        figsize: Tuple[int, int] = (16, 5),
    ) -> Tuple["plt.Figure", np.ndarray]:
        self._check_is_fitted()
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        K = self.n_splines
        K_eff = self.n_coefs

        x_min, x_max = x.min(), x.max()
        x_grid = np.linspace(x_min, x_max, n_grid)
        X_grid = self._build_design_matrix(x_grid)
        B_grid = X_grid[:, :K_eff]
        y_pred = X_grid @ self.coef_
        knots = self.knots_

        cmap = plt.get_cmap('tab20' if K_eff <= 20 else 'tab10' if K_eff <= 10 else 'hsv')
        colors = [cmap(i % 20 if K_eff <= 20 else i / K_eff) for i in range(K_eff)]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax1 = axes[0]
        for j in range(K_eff):
            ax1.plot(x_grid, B_grid[:, j], color=colors[j], ls='-', alpha=0.8, label=f'B_{j}')
        for i_knot, knot in enumerate(knots):
            ax1.axvline(knot, color='gray', ls='--', lw=1, label='Knots' if i_knot == 0 else None)
        ax1.set(xlabel='x', ylabel='B_j(x)', title=f'B-Spline Basis (K_eff={K_eff})')
        ax1.minorticks_on()
        ax1.legend(fontsize=6, loc='upper right')

        ax2 = axes[1]
        vals = np.append(self.coef_[:K_eff], self.coef_[K_eff])
        labels = [f'B_{j}' for j in range(K_eff)] + ['intercept']
        ax2.bar(range(len(vals)), vals, color=colors + ['dimgray'])
        ax2.set_xticks(range(len(vals)))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax2.axhline(0, color='k')
        ax2.set(ylabel='Coefficient value', title='Estimated Coefficients')

        ax3 = axes[2]
        ax3.scatter(x, y, alpha=0.5, color='gray', s=12, label='Data')
        for j in range(K_eff):
            weighted = self.coef_[j] * B_grid[:, j]
            ax3.plot(x_grid, weighted + self.coef_[K_eff], color=colors[j], lw=1, ls='-')
        ax3.plot(x_grid, y_pred, 'k-', label='Fit (sum)')
        for knot in knots:
            ax3.axvline(knot, color='gray', ls='--', lw=1)
        ax3.set(xlabel='x', ylabel='y', title='Weighted Basis Functions + Fit')
        ax3.minorticks_on()
        ax3.legend(fontsize=6, loc='upper left')

        cons_str = f", constraint='{self.constraint}'" if self.constraint else ""
        fig.suptitle(f"GAM Decomposition (K={K}{cons_str}, lam={self.lam:.4f})")
        return fig, axes

    def plot_diagnostics(
        self,
        x: ArrayLike,
        y: ArrayLike,
        figsize: Tuple[int, int] = (10, 9),
    ) -> Tuple["plt.Figure", np.ndarray]:
        """Diagnostic plots for model validation.

        Produces a 2×2 grid with:

        * Observed vs Fitted (with 1:1 line and :math:`R^2`)
        * Residuals vs Fitted
        * Normal Q-Q plot of residuals
        * Histogram of residuals with normal overlay

        Parameters
        ----------
        x, y : array-like
            Data used for validation (typically the training set).
        figsize : tuple of int, default (10, 9)
            Figure size in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of matplotlib.axes.Axes
        """
        self._check_is_fitted()
        x_arr = np.asarray(x).ravel()
        y_arr = np.asarray(y).ravel()

        y_fit = self.predict(x_arr)
        residuals = y_arr - y_fit
        r2 = self.deviance_explained()

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()

        # 1. Observed vs Fitted
        ax = axes[0]
        ax.scatter(y_fit, y_arr, alpha=0.5, edgecolors='k', s=20)
        lims = [
            np.min([y_arr.min(), y_fit.min()]),
            np.max([y_arr.max(), y_fit.max()]),
        ]
        ax.plot(lims, lims, 'r--', lw=1.5, label='1:1')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Observed values')
        ax.set_title(f'Observed vs Fitted ($R^2$={r2:.3f})')
        ax.legend()

        # 2. Residuals vs Fitted
        ax = axes[1]
        ax.scatter(y_fit, residuals, alpha=0.5, edgecolors='k', s=20)
        ax.axhline(0, color='r', linestyle='--', lw=1.5)
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted')

        # 3. Normal Q-Q
        ax = axes[2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q')
        ax.get_lines()[0].set_markerfacecolor('steelblue')
        ax.get_lines()[0].set_markeredgecolor('k')
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[1].set_color('r')

        # 4. Histogram of residuals
        ax = axes[3]
        ax.hist(
            residuals, bins='auto', density=True,
            color='steelblue', edgecolor='k', alpha=0.7,
        )
        sigma = np.std(residuals, ddof=1)
        x_norm = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(
            x_norm, stats.norm.pdf(x_norm, 0.0, sigma),
            'r-', lw=2, label='Normal approx.',
        )
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.legend()

        cons_str = f", constraint='{self.constraint}'" if self.constraint else ""
        fig.suptitle(
            f'LinGAM Diagnostics (K={self.n_splines}, '
            f'lam={self.lam:.4f}{cons_str})',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig, axes