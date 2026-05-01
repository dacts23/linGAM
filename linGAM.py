import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from typing import Optional, Tuple, Dict, Any, List
from numpy.typing import ArrayLike




class LinGAM:
    """
    Numerically exact equivalent of pyGAM's LinearGAM for a single smooth term.
    """

    def __init__(self, n_splines: int = 10, lam: float = 1.0, spline_order: int = 3):
        """
        Initialize GAM with hyperparameters.

        Parameters
        ----------
        n_splines : int, default 10
            Number of B-spline basis functions (K). Must satisfy K > spline_order.
        lam : float, default 1.0
            Smoothing parameter λ ≥ 0.
        spline_order : int, default 3
            Order k of the B-spline basis (= polynomial degree).
        """
        if n_splines <= spline_order:
            raise ValueError("n_splines must be greater than spline_order.")
        if lam < 0:
            raise ValueError("Smoothing parameter 'lam' must be non-negative.")
        if spline_order < 1:
            raise ValueError("spline_order must be >= 1.")

        self.n_splines = n_splines
        self.lam = lam
        self.spline_order = spline_order
        self.edge_knots_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.design_matrix_: Optional[np.ndarray] = None
        self.statistics_: Optional[Dict[str, Any]] = None
        self._B_solve: Optional[np.ndarray] = None
        self.knots_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been fitted."""
        return self.coef_ is not None

    def _check_is_fitted(self) -> None:
        """Raise an error if the model is not fitted."""
        if not self.is_fitted:
            raise RuntimeError("This LinGAM instance is not fitted yet. Call 'fit' or 'gridsearch' first.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"LinGAM(n_splines={self.n_splines}, lam={self.lam:.4g}, spline_order={self.spline_order}) [{status}]"

    def _b_spline_basis(self, x_in: ArrayLike, n_splines: Optional[int] = None) -> np.ndarray:
        """Construct B-spline basis matrix using vectorized De Boor recursion."""
        K = n_splines if n_splines is not None else self.n_splines
        k = self.spline_order

        offset = self.edge_knots_[0]
        scale = self.edge_knots_[1] - self.edge_knots_[0]
        if scale == 0.0:
            scale = 1.0

        boundary_knots = np.linspace(0.0, 1.0, 1 + K - k)
        diff = boundary_knots[1] - boundary_knots[0]

        x = (np.asarray(x_in).ravel() - offset) / scale
        x = np.r_[x, 0.0, 1.0]

        x_extrap_l = x < 0.0
        x_extrap_r = x > 1.0
        x_interp = ~(x_extrap_l | x_extrap_r)
        x = np.atleast_2d(x).T

        aug = np.arange(1, k + 1) * diff
        aug_knots = np.r_[-aug[::-1], boundary_knots, 1.0 + aug]
        aug_knots[-1] += 1e-9

        bases = ((x >= aug_knots[:-1]) & (x < aug_knots[1:])).astype(float)
        bases[-1] = bases[-2][::-1]

        n_active = len(aug_knots) - 1
        prev_bases = None

        for m in range(2, k + 2):
            n_active -= 1
            num_l = (x - aug_knots[:n_active]) * bases[:, :n_active]
            denom_l = aug_knots[m - 1:n_active + m - 1] - aug_knots[:n_active]
            left = num_l / denom_l

            num_r = (aug_knots[m:n_active + m] - x) * bases[:, 1:n_active + 1]
            denom_r = aug_knots[m:n_active + m] - aug_knots[1:n_active + 1]
            right = num_r / denom_r

            prev_bases = bases[-2:].copy()
            bases = left + right

        if (np.any(x_extrap_l) or np.any(x_extrap_r)) and k > 0:
            bases[~x_interp] = 0.0

            denom_l = aug_knots[k:-1] - aug_knots[:-k - 1]
            left_g = prev_bases[:, :-1] / denom_l

            denom_r = aug_knots[k + 1:] - aug_knots[1:-k]
            right_g = prev_bases[:, 1:] / denom_r

            grads = k * (left_g - right_g)

            if np.any(x_extrap_l):
                bases[x_extrap_l] = grads[0] * x[x_extrap_l] + bases[-2]

            if np.any(x_extrap_r):
                bases[x_extrap_r] = grads[1] * (x[x_extrap_r] - 1.0) + bases[-1]

        return bases[:-2]

    def _build_design_matrix(self, x: np.ndarray, n_splines: Optional[int] = None) -> np.ndarray:
        """Build the complete design matrix (B-spline basis + intercept)."""
        B = self._b_spline_basis(x, n_splines=n_splines)
        return np.hstack([B, np.ones((len(x), 1))])

    def _build_penalty(self, lam: float, n_splines: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Construct the full (K+1)x(K+1) block-diagonal penalty matrix."""
        K = n_splines if n_splines is not None else self.n_splines
        m = K + 1

        D2 = np.zeros((K - 2, K))
        idx = np.arange(K - 2)
        D2[idx, idx] = 1.0
        D2[idx, idx + 1] = -2.0
        D2[idx, idx + 2] = 1.0

        S_pen = D2.T @ D2

        P = np.zeros((m, m))
        P[:K, :K] = lam * S_pen
        return P, S_pen

    def _solve_pirls(self, X: np.ndarray, y: np.ndarray, P: np.ndarray,
                     Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve penalized least squares: (X^T X + P) * coef = X^T y.
        Uses Data Augmentation (QR + SVD) to avoid squaring the condition number of X.
        """
        n, m = X.shape
        eps = np.finfo(float).eps

        # Factor penalty matrix P = E^T E using Cholesky
        S_ridge = np.diag(np.ones(m) * np.sqrt(eps))
        E = np.linalg.cholesky(S_ridge + P).T
        
        # Compress X using QR: X = Q R
        if Q is None or R is None:
            Q, R = np.linalg.qr(X)

        min_nm = min(m, n)
        # SVD on stacked matrix [R; E] = U D V^T
        U, d, Vt = np.linalg.svd(np.vstack([R, E]), full_matrices=False)

        # Extract U1 (top half) corresponding to the data matrix R
        U1 = U[:min_nm, :min_nm]
        Vt = Vt[:min_nm]
        d_inv = 1.0 / d[:min_nm]

        # The solution simplifies to: coef = V D^-1 U1^T Q^T y
        B_solve = (Vt.T * d_inv) @ U1.T @ Q.T
        coef = B_solve @ y

        return coef, U1, B_solve

    def gridsearch(self, x: ArrayLike, y: ArrayLike,
                   lam: Optional[ArrayLike] = None,
                   n_splines: Optional[ArrayLike] = None,
                   gamma: float = 1.4,
                   fast: bool = True, robust: bool = False) -> "LinGAM":
        """
        Automatically select the optimal smoothing penalty (lam) and number of 
        basis functions (n_splines) by minimizing Generalized Cross Validation (GCV).
        
        GCV acts as a computationally efficient proxy for leave-one-out cross-validation, 
        balancing the Mean Squared Error (MSE) of the fit against the Effective Degrees 
        of Freedom (EDF) consumed by the spline's flexibility.
        
        Parameters
        ----------
        x : ArrayLike
            1D array of predictor values.
        y : ArrayLike
            1D array of target/response values.
        lam : array-like, optional
            Array of lambda candidates. Higher lambda forces a stiffer curve.
        n_splines : array-like, optional
            Array of basis function counts. Higher count increases flexibility.
        gamma : float, default=1.4
            Multiplier for the EDF penalty. Usually 1.4 to prevent overfitting.
        fast : bool, default=True
            Enables ultra-fast QR + Cholesky evaluations (Disabled if robust=True).
        robust : bool, default=False
            Uses Iteratively Reweighted Least Squares (IRLS) with Huber 
            weights to eliminate the influence of massive outliers.

        Returns
        -------
        LinGAM
            The fitted model instance itself.
        """
        if fast and not robust:
            return self.gridsearch_fast(x, y, lam=lam, n_splines=n_splines, gamma=gamma)
        return self.gridsearch_basic(x, y, lam=lam, n_splines=n_splines, gamma=gamma, robust=robust)

    def gridsearch_basic(self, x: ArrayLike, y: ArrayLike,
                         lam: Optional[ArrayLike] = None,
                         n_splines: Optional[ArrayLike] = None,
                         gamma: float = 1.4, robust: bool = False) -> "LinGAM":
        """
        Select optimal (lambda, n_splines) by minimizing GCV over a cartesian
        product grid using multithreading for lambda evaluation.
        """
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        if lam is None:
            lam = np.logspace(-3, 3, 11)
        if n_splines is None:
            n_splines = np.arange(5, 25, 2)

        self.edge_knots_ = np.array([x.min(), x.max()])
        n = len(y)

        best_gcv = np.inf
        best_result = None
        results: List[Tuple[float, int, float, np.ndarray, float]] = []

        for K in n_splines:
            self.n_splines = K
            Xmat = self._build_design_matrix(x)
            Q, R = np.linalg.qr(Xmat)
            _, S_pen = self._build_penalty(1.0)
            m = K + 1

            def eval_lam(l_val: float) -> Tuple[float, int, float, np.ndarray, float]:
                P = np.zeros((m, m))
                P[:K, :K] = l_val * S_pen
                
                if robust:
                    # 1. Median initialization prevents the spline from perfectly tracing outliers
                    med_y = np.median(y)
                    residuals = y - med_y
                    
                    coef, edf = None, 0.0
                    w = np.ones(n)
                    for _ in range(10): # Iteratively Reweighted Least Squares (IRLS)
                        # 2. Compute robust scale (Median Absolute Deviation)
                        mad = np.median(np.abs(residuals - np.median(residuals)))
                        scale_est = mad / 0.6745 if mad > 1e-6 else 1e-6
                        
                        # 3. Compute Huber weights (Convex, safe for flexible splines)
                        c = 1.345 * scale_est
                        abs_r = np.abs(residuals)
                        w = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-6))
                        
                        # 4. Scale X and y by sqrt(w) to solve Weighted Least Squares
                        W_sqrt = np.sqrt(w).reshape(-1, 1)
                        X_w = Xmat * W_sqrt
                        y_w = y * np.sqrt(w)
                        
                        Qw, Rw = np.linalg.qr(X_w)
                        coef, U1, _ = self._solve_pirls(X_w, y_w, P, Q=Qw, R=Rw)
                        residuals = y - Xmat @ coef
                        
                        # EDF is simply the sum of squared elements of SVD component U1
                        edf = np.sum(U1 ** 2)
                        
                    y_hat = Xmat @ coef
                    rss = np.sum(w * (y - y_hat) ** 2)
                else:
                    coef, U1, _ = self._solve_pirls(Xmat, y, P, Q=Q, R=R)
                    y_hat = Xmat @ coef
                    rss = np.sum((y - y_hat) ** 2)
                    edf = np.sum(U1 ** 2)
                    
                gcv_score = (n * rss) / (n - gamma * edf) ** 2
                return gcv_score, K, l_val, coef, edf

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(eval_lam, l_val) for l_val in lam]
                for future in futures:
                    res = future.result()
                    results.append(res)
                    if res[0] < best_gcv:
                        best_gcv = res[0]
                        best_result = res

        _, best_K, best_l, _, _ = best_result
        self.n_splines = best_K
        self.lam = best_l
        self.gcv_results_ = results

        self._fit_internal(x, y, robust=robust)
        return self

    def gridsearch_fast(self, x: ArrayLike, y: ArrayLike,
                        lam: Optional[ArrayLike] = None,
                        n_splines: Optional[ArrayLike] = None,
                        gamma: float = 1.4, robust: bool = False) -> "LinGAM":
        """
        Secondary grid search that multithreads BOTH n_splines (precomputing QR)
        and lam evaluations concurrently.
        """
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        if lam is None:
            lam = np.logspace(-3, 3, 11)
        if n_splines is None:
            n_splines = np.arange(5, 25, 2)

        self.edge_knots_ = np.array([x.min(), x.max()])
        n = len(y)

        best_gcv = np.inf
        best_result = None
        results: List[Tuple[float, int, float, np.ndarray, float]] = []

        # Phase 1: Precompute B-splines and QR decompositions for all K concurrently
        def precompute_K(K: int):
            Xmat = self._build_design_matrix(x, n_splines=K)
            Xmat_w = Xmat
            y_w = y
            
            Q, R = np.linalg.qr(Xmat_w)
            _, S_pen = self._build_penalty(1.0, n_splines=K)
            
            RtR = R.T @ R
            Rt_qty = R.T @ (Q.T @ y_w)
            
            P_base = np.zeros((K + 1, K + 1))
            P_base[:K, :K] = S_pen
            
            return K, Xmat, P_base, RtR, Rt_qty, K + 1

        with ThreadPoolExecutor() as executor:
            precomputed = list(executor.map(precompute_K, n_splines))

        # Phase 2: Evaluate concurrently using ultra-fast stripped-down Cholesky
        def eval_candidate(args) -> Tuple[float, int, float, np.ndarray, float]:
            K, l_val, Xmat, P_base, RtR, Rt_qty, m = args
            
            # Normal equations: (R^T R + lam * P) * coef = R^T Q^T y
            B = RtR + l_val * P_base
            B.flat[::m+1] += 1e-12 # Ridge penalty for stability
            
            try:
                # Cholesky is 2x faster than LU. Decompose once, solve twice.
                c, lower = cho_factor(B, overwrite_a=True, check_finite=False)
                coef = cho_solve((c, lower), Rt_qty, check_finite=False)
                
                # Trace of hat matrix (EDF) = Trace( B^-1 * R^T R )
                B_inv_RtR = cho_solve((c, lower), RtR, check_finite=False)
                edf = np.trace(B_inv_RtR)
            except Exception:
                # Fallback to LU if numerically non-positive definite
                coef = np.linalg.solve(B, Rt_qty)
                edf = np.trace(np.linalg.solve(B, RtR))
                
            y_hat = Xmat @ coef
            rss = np.sum((y - y_hat) ** 2)
            gcv_score = (n * rss) / (n - gamma * edf) ** 2
            return gcv_score, K, l_val, coef, edf

        tasks = []
        for K, Xmat, P_base, RtR, Rt_qty, m in precomputed:
            for l_val in lam:
                tasks.append((K, l_val, Xmat, P_base, RtR, Rt_qty, m))

        with ThreadPoolExecutor() as executor:
            for res in executor.map(eval_candidate, tasks):
                results.append(res)
                if res[0] < best_gcv:
                    best_gcv = res[0]
                    best_result = res

        _, best_K, best_l, _, _ = best_result
        self.n_splines = best_K
        self.lam = best_l
        self.gcv_results_ = results

        self._fit_internal(x, y, robust=robust)
        return self

    def fit(self, x: ArrayLike, y: ArrayLike, robust: bool = False) -> "LinGAM":
        """
        Fit a single Penalized B-spline model to the given (x, y) data using the 
        currently configured hyper-parameters (n_splines and lam).
        
        The mathematical objective is to minimize the penalized least squares loss:
            Loss = ||y - X*beta||^2 + lam * beta^T * S * beta
        
        where X is the B-spline basis design matrix, and S is the second-derivative 
        finite difference penalty matrix (which penalizes 'wiggly' curves).
        
        If robust=True, this uses Iteratively Reweighted Least Squares (IRLS) initialized 
        around the global median to iteratively down-weight outliers using the Huber loss.
        
        Parameters
        ----------
        x : ArrayLike
            1D array of predictor values.
        y : ArrayLike
            1D array of target/response values.
        robust : bool, default=False
            Whether to use Huber-weighted IRLS to ignore outliers.

        Returns
        -------
        LinGAM
            The fitted model instance itself.
        """
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        self.edge_knots_ = np.array([x.min(), x.max()])
        self._fit_internal(x, y, robust=robust)
        return self

    def _fit_internal(self, x: np.ndarray, y: np.ndarray, robust: bool = False) -> None:
        """Internal fit: builds basis, solves PIRLS, computes all statistics."""
        n = len(x)

        k = self.spline_order
        K = self.n_splines
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
            # 1. Initialize residuals around the global median
            med_y = np.median(y)
            residuals = y - med_y
            w = np.ones(n)
            
            for _ in range(15): # IRLS Loop
                # 2. Scale estimation using MAD
                mad = np.median(np.abs(residuals - np.median(residuals)))
                scale_est = mad / 0.6745 if mad > 1e-6 else 1e-6
                
                # 3. Compute Huber weights (Convex, safe for flexible splines)
                c = 1.345 * scale_est
                abs_r = np.abs(residuals)
                w = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-6))
                
                # 4. Convert to weighted least squares
                W_sqrt = np.sqrt(w).reshape(-1, 1)
                X_w = X * W_sqrt
                y_w = y * np.sqrt(w)
                
                self.coef_, U1, B_solve = self._solve_pirls(X_w, y_w, P)
                residuals = y - X @ self.coef_
            
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

        scale = np.sqrt(rss / (n - edof))
        cov = (B_solve @ B_solve.T) * scale ** 2
        se = np.sqrt(np.diag(cov))

        self.statistics_ = {
            'edof': edof,
            'edof_per_coef': edof_per_coef,
            'scale': scale,
            'cov': cov,
            'se': se,
            'rss': rss,
            'n_samples': n,
            'GCV': (n * rss) / (n - 1.4 * edof) ** 2,
        }

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Compute predictions at new covariate values."""
        self._check_is_fitted()
        x = np.asarray(x).ravel()
        X = self._build_design_matrix(x)
        return X @ self.coef_

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

    def plot_decomposition(self, x: ArrayLike, y: ArrayLike,
                           n_grid: int = 300, figsize: Tuple[int, int] = (16, 5)):
        """
        Visualize the GAM fit as a sum of individual, weighted B-spline basis functions.
        
        This plot mathematically deconstructs the final prediction into its constituent 
        local curves. Each basis function is multiplied by its corresponding fitted 
        coefficient, visually demonstrating how the localized splines overlap and sum 
        together to form the final global prediction curve.
        
        Parameters
        ----------
        x : ArrayLike
            1D array of original predictor values.
        y : ArrayLike
            1D array of original target/response values.
        n_grid : int, default=300
            Number of points to generate for the continuous x-axis grid. 
            A higher number results in smoother rendered spline curves.
        figsize : tuple, default=(16, 5)
            Width and height of the generated matplotlib figure.
        """
        self._check_is_fitted()
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        K = self.n_splines

        x_min, x_max = x.min(), x.max()
        x_grid = np.linspace(x_min, x_max, n_grid)
        X_grid = self._build_design_matrix(x_grid)
        B_grid = X_grid[:, :-1]
        y_pred = X_grid @ self.coef_

        # Knot locations in original scale
        knots = self.knots_

        cmap = plt.get_cmap('tab20' if K <= 20 else 'tab10' if K <= 10 else 'hsv')
        colors = [cmap(i % 20 if K <= 20 else i / K) for i in range(K)]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Raw basis functions
        ax1 = axes[0]
        for j in range(K):
            ax1.plot(x_grid, B_grid[:, j], color=colors[j], ls='-', alpha=0.8, label=f'B_{j}')
        for i, knot in enumerate(knots):
            ax1.axvline(knot, color='gray', ls='--', lw=1,
                        label='Knots' if i == 0 else None)
        ax1.set(xlabel='x', ylabel='B_j(x)', title=f'B-Spline Basis Functions (K={K})')
        ax1.minorticks_on()
        ax1.legend(fontsize=6, loc='upper right')

        # 2. Coefficients
        ax2 = axes[1]
        vals = np.append(self.coef_[:K], self.coef_[K])
        labels = [f'B_{j}' for j in range(K)] + ['intercept']
        ax2.bar(range(len(vals)), vals, color=colors + ['dimgray'])
                
        ax2.set_xticks(range(len(vals)))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax2.axhline(0, color='k')
        ax2.set(ylabel='Coefficient value', title='Estimated Coefficients')

        # 3. Weighted basis + fit
        ax3 = axes[2]
        ax3.scatter(x, y, alpha=0.5, color='gray', s=12, label='Data')
        for j in range(K):
            weighted = self.coef_[j] * B_grid[:, j]
            ax3.plot(x_grid, weighted + self.coef_[K], color=colors[j], lw=1, ls='-')
        ax3.plot(x_grid, y_pred, 'k-', label='Fit (sum)')
        for knot in knots:
            ax3.axvline(knot, color='gray', ls='--',lw=1)
        ax3.set(xlabel='x', ylabel='y', title='Weighted Basis Functions + Fit')
        ax3.minorticks_on()
        ax3.legend(fontsize=6, loc='upper left')

        fig.suptitle(f"GAM Decomposition (K={self.n_splines}, lam={self.lam:.4f})")
        return fig, axes


if __name__ == "__main__":
    has_pygam = True
    try:
        from pygam import LinearGAM, s
    except ImportError:
        has_pygam = False
        print("pyGAM is not installed. Skipping pyGAM comparison.")

    # 1. Synthetic Jet Engine EGT Data (Clustered with Bald Spots)
    np.random.seed(42)
    n_clusters = 12  # Reduced amount of clusters
    pts_per_cluster = 9
    
    # Fan speed intervals from 20% (idle) to 100% (max power)
    cluster_centers = np.linspace(20, 100, n_clusters)
    
    # Generate clusters by scattering X, then evaluating the true curve at those X's and adding Y-noise
    x_list, y_list = [], []
    for center in cluster_centers:
        cluster_x = np.random.normal(loc=center, scale=1, size=pts_per_cluster)
        cluster_y_true = 300 + 300 * np.exp(-(cluster_x - 12) / 8) + 6 * cluster_x
        cluster_y = cluster_y_true + np.random.normal(0, 3.0, size=pts_per_cluster)
        x_list.extend(cluster_x)
        y_list.extend(cluster_y)
        
    x_raw, y_raw = np.array(x_list), np.array(y_list)
    sort_idx = np.argsort(x_raw)
    x = x_raw[sort_idx]
    y_obs = y_raw[sort_idx]
    
    y_true = 300 + 300 * np.exp(-(x - 12) / 8) + 6 * x
    
    # Add a few massive outliers to test robust fitting
    y_obs[10] += 600
    y_obs[30] -= 500
    y_obs[60] += 700

    # ── Test: GCV Grid Search ──
    print("=" * 60)
    print("TEST: GCV GRID SEARCH (lam x n_splines)")
    print("=" * 60)

    lam_grid = np.logspace(-3, 3, 11)
    nsp_grid = np.arange(5, 15, 1)

    # Fast custom grid search (Standard)
    gcv_fast = LinGAM(n_splines=15, spline_order=3)
    gcv_fast.gridsearch(x, y_obs, lam=lam_grid, n_splines=nsp_grid, fast=True, robust=False)
    y_gs_fast = gcv_fast.predict(x)
    print(f"Custom Fast (Standard) best: n_splines={gcv_fast.n_splines}, lam={gcv_fast.lam:.4f}")
    
    # Fast custom grid search (Robust)
    gcv_robust = LinGAM(n_splines=15, spline_order=3)
    gcv_robust.gridsearch(x, y_obs, lam=lam_grid, n_splines=nsp_grid, fast=True, robust=True)
    y_gs_robust = gcv_robust.predict(x)
    print(f"Custom Fast (Robust) best: n_splines={gcv_robust.n_splines}, lam={gcv_robust.lam:.4f}")

    if has_pygam:
        pygam_gs = LinearGAM(s(0, n_splines=15, lam=6.0), fit_intercept=True, verbose=False)
        pygam_gs.gridsearch(x, y_obs, lam=lam_grid, n_splines=nsp_grid, progress=False)
        y_gs_pygam = pygam_gs.predict(x)
        print(f"pyGAM best: n_splines={pygam_gs.terms[0].n_splines}, lam={float(pygam_gs.terms[0].lam[0]):.4f}")
        
        match_params = (gcv_fast.n_splines == pygam_gs.terms[0].n_splines and
                        abs(gcv_fast.lam - float(pygam_gs.terms[0].lam[0])) < 1e-6)
        if match_params:
            pred_diff = np.max(np.abs(y_gs_pygam - y_gs_fast))
            print(f"pyGAM vs Custom Fast MATCH! Prediction diff: {pred_diff:.2e}")
        else:
            print("Parameters differ between pyGAM and Custom Fast.")

    # ── Visualization ──
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y_obs, alpha=0.15, color="gray", s=12, label="Noisy Data")
    plt.plot(x, y_true, "k--", lw=1.5, label="Ground Truth", alpha=0.6)
    
    plt.plot(x, y_gs_fast, "r-", lw=2,
             label=f"Standard Fit (K={gcv_fast.n_splines}, lam={gcv_fast.lam:.4f})", alpha=0.9)
             
    plt.plot(x, y_gs_robust, "g-", lw=2,
             label=f"Robust Fit (K={gcv_robust.n_splines}, lam={gcv_robust.lam:.4f})", alpha=0.9)

    if has_pygam:
        plt.plot(x, y_gs_pygam, "b:", lw=2,
                 label=f"pyGAM GS (K={pygam_gs.terms[0].n_splines}, lam={float(pygam_gs.terms[0].lam[0]):.4f})", alpha=0.8)
    
    gs_ci = gcv_robust.confidence_intervals(x, width=0.95)
    gs_pi = gcv_robust.prediction_intervals(x, width=0.95)
    plt.fill_between(x, gs_ci[:, 0], gs_ci[:, 1], alpha=0.15, color='green', label="Robust 95% CI")
    plt.fill_between(x, gs_pi[:, 0], gs_pi[:, 1], alpha=0.07, color='lightgreen', label="Robust 95% PI")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("GCV Grid Search")
    plt.legend(fontsize=7, loc='upper left')
    
    # Also generate the decomposition plot
    gcv_robust.plot_decomposition(x, y_obs)
    
    # Show all plots simultaneously
    plt.show()