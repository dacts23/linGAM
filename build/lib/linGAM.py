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
        """Construct B-spline basis matrix using vectorized De Boor recursion.

        B-splines are smooth, bell-shaped curves that are each non-zero only on a
        small interval. Think of them as "building blocks" — any smooth curve can be
        approximated by stacking and scaling these blocks. Each block overlaps with
        its neighbors, which is what makes the resulting curves smooth (no sharp
        corners between segments).

        The De Boor recursion builds higher-order (smoother) splines from lower-order
        ones: a cubic B-spline (order 4) is a weighted average of two quadratic
        B-splines (order 3), which themselves are averages of two linear B-splines,
        and so on. Each step makes the curve one degree smoother.
        """
        K = n_splines if n_splines is not None else self.n_splines
        k = self.spline_order

        # Map data to [0, 1] so that internal knots are evenly spaced regardless
        # of the original x-range. This prevents numerical issues when x has very
        # large or very small values, and makes the knot spacing computation simple.
        offset = self.edge_knots_[0]
        scale = self.edge_knots_[1] - self.edge_knots_[0]
        if scale == 0.0:
            scale = 1.0

        # Place internal knots evenly across [0, 1]. We need K - k + 1 boundary
        # positions: the fewer knots relative to the number of basis functions,
        # the more "overlap" — which is exactly what makes B-splines smooth.
        boundary_knots = np.linspace(0.0, 1.0, 1 + K - k)
        diff = boundary_knots[1] - boundary_knots[0]

        x = (np.asarray(x_in).ravel() - offset) / scale

        # Append 0.0 and 1.0 as sentinel values: these are used to compute the
        # slope (gradient) of the basis functions at the boundary so that we can
        # extrapolate linearly beyond the data range.
        x = np.r_[x, 0.0, 1.0]

        # Track which points fall outside [0, 1] so we can handle them separately.
        # Points inside the range use the B-spline formula; points outside get
        # a linear extension (no wild extrapolation).
        x_extrap_l = x < 0.0
        x_extrap_r = x > 1.0
        x_interp = ~(x_extrap_l | x_extrap_r)
        x = np.atleast_2d(x).T

        # Extend knots beyond [0, 1] on both sides so that every basis function
        # is fully defined at every data point. Without these "phantom knots", the
        # outermost basis functions near the boundary would be incomplete (like
        # trying to average two numbers when one is missing).
        aug = np.arange(1, k + 1) * diff
        aug_knots = np.r_[-aug[::-1], boundary_knots, 1.0 + aug]
        aug_knots[-1] += 1e-9  # Tiny nudge so the last interval includes its right endpoint

        # Start with order-1 (piecewise constant) basis: each function is 1 inside
        # its interval and 0 everywhere else — a staircase pattern.
        bases = ((x >= aug_knots[:-1]) & (x < aug_knots[1:])).astype(float)
        bases[-1] = bases[-2][::-1]  # Mirror the last two for boundary gradient calculation

        n_active = len(aug_knots) - 1
        prev_bases = None

        # De Boor recursion: build order-m splines from order-(m-1) splines.
        # At each step, two adjacent lower-order functions are blended using
        # a weighted average based on where x sits between their knot intervals.
        # This is the "smoothness engine" — each blending averages out corners.
        for m in range(2, k + 2):
            n_active -= 1

            # Left blend fraction: "how far is x into the left neighbor's interval?"
            # If x is near the left knot, the left basis dominates.
            num_l = (x - aug_knots[:n_active]) * bases[:, :n_active]
            denom_l = aug_knots[m - 1:n_active + m - 1] - aug_knots[:n_active]
            left = num_l / denom_l

            # Right blend fraction: "how far is x from the right neighbor's interval?"
            # If x is near the right knot, the right basis dominates.
            num_r = (aug_knots[m:n_active + m] - x) * bases[:, 1:n_active + 1]
            denom_r = aug_knots[m:n_active + m] - aug_knots[1:n_active + 1]
            right = num_r / denom_r

            # Save the second-to-last order bases before final blending — we need
            # these to compute boundary gradients for linear extrapolation later.
            prev_bases = bases[-2:].copy()
            bases = left + right

        # Handle extrapolation: instead of letting B-splines shoot off to infinity
        # outside the data range, we linearly extend using the slope at each boundary.
        # This is like putting a ruler against the curve at the edge and extending it.
        if (np.any(x_extrap_l) or np.any(x_extrap_r)) and k > 0:
            bases[~x_interp] = 0.0  # Wipe the B-spline values at extrapolated points

            # Compute the gradient (slope) of the (k-1)-order bases at the boundaries.
            # The derivatives of B-splines have a nice closed form: they are simply
            # k * (left_fraction - right_fraction), which follows from the chain rule
            # applied to the recursive definition above.
            denom_l = aug_knots[k:-1] - aug_knots[:-k - 1]
            left_g = prev_bases[:, :-1] / denom_l

            denom_r = aug_knots[k + 1:] - aug_knots[1:-k]
            right_g = prev_bases[:, 1:] / denom_r

            grads = k * (left_g - right_g)

            # Linear extension: f(x) = f(boundary) + slope * distance_from_boundary
            if np.any(x_extrap_l):
                bases[x_extrap_l] = grads[0] * x[x_extrap_l] + bases[-2]

            if np.any(x_extrap_r):
                bases[x_extrap_r] = grads[1] * (x[x_extrap_r] - 1.0) + bases[-1]

        # Remove the two sentinel rows we added earlier (for 0.0 and 1.0)
        return bases[:-2]

    def _build_design_matrix(self, x: np.ndarray, n_splines: Optional[int] = None) -> np.ndarray:
        """Build the complete design matrix (B-spline basis + intercept column).

        The design matrix has one column per basis function plus one column of
        all ones for the intercept (the "grand mean" of the data). Without the
        intercept, the spline would be forced through y=0 at every point. The
        intercept lets the curve shift vertically to match the data's average level.

        This is the "X" in the familiar equation y = X * beta, where beta contains
        one coefficient per basis function plus one for the intercept.
        """
        B = self._b_spline_basis(x, n_splines=n_splines)
        return np.hstack([B, np.ones((len(x), 1))])

    def _build_penalty(self, lam: float, n_splines: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Construct the second-derivative penalty matrix that discourages "wiggly" curves.

        The core idea: without a penalty, the spline would chase every data point and
        produce a wildly oscillating curve — classic overfitting. We prevent this by
        adding λ * βᵀ S β to the loss, which charges a "cost" for curvature.

        Second differences (D2) approximate the second derivative of the coefficient
        vector: if neighboring coefficients are similar, the second differences are
        small (smooth curve); if they differ wildly, the second differences are large
        (wiggly curve). Multiplying S = D2ᵀ D2 makes this cost quadratic — so the
        total penalty is large when the curve bends sharply, and small when it's gentle.

        The (K+1)×(K+1) matrix P pads S with a row/column of zeros so the intercept
        is NOT penalized — we want the model to freely choose the vertical offset.
        """
        K = n_splines if n_splines is not None else self.n_splines
        m = K + 1

        # Second-difference operator: each row has [1, -2, 1], measuring how much
        # the curve bends at each interior point. Big values = sharp bending.
        D2 = np.zeros((K - 2, K))
        idx = np.arange(K - 2)
        D2[idx, idx] = 1.0
        D2[idx, idx + 1] = -2.0
        D2[idx, idx + 2] = 1.0

        S_pen = D2.T @ D2

        # Embed S in a larger matrix that leaves the intercept unpenalized.
        # The bottom-right 0 ensures the intercept coefficient pays no penalty.
        P = np.zeros((m, m))
        P[:K, :K] = lam * S_pen
        return P, S_pen

    def _solve_pirls(self, X: np.ndarray, y: np.ndarray, P: np.ndarray,
                     Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve penalized least squares: (X^T X + P) * coef = X^T y.

        This finds the coefficient vector beta that minimizes:
            ||y - X beta||^2 + beta^T P beta

        Why not just solve the normal equations (X^T X + P) beta = X^T y directly?
        Because forming X^T X "squares" the condition number — if X is even slightly
        ill-conditioned (nearly redundant columns), X^T X amplifies that numerical
        instability dramatically. Instead, we use a "data augmentation" trick:

        1. Factor the penalty P = E^T E (Cholesky), turning the penalty into
           additional "fake data rows" that gently pull coefficients toward zero.
        2. Compress X with QR decomposition: X = Q R, separating orthogonal structure
           from scale. Q is well-conditioned by construction, so no information loss.
        3. Stack R on top of E and decompose [R; E] via SVD. SVD is the most
           numerically stable way to solve linear systems — it handles near-singular
           matrices gracefully by truncating tiny singular values.
        4. The final formula eta = V D^{-1} U1^T Q^T y is a stable matrix multiply.

        We also save B_solve = V D^{-1} U1^T Q^T (the "solution operator") to reuse
        later for computing confidence intervals — it's essentially the "inverse
        operator" that maps any y-vector to the corresponding beta, and its outer
        product B_solve @ B_solve^T is the covariance of beta (up to a scale factor).
        """
        n, m = X.shape
        eps = np.finfo(float).eps

        # Small ridge (epsilon * I) ensures Cholesky cannot fail — even when P
        # is zero (lambda=0, no penalty at all), the tiny diagonal makes P
        # positive-definite so the Cholesky factorization succeeds.
        S_ridge = np.diag(np.ones(m) * np.sqrt(eps))
        E = np.linalg.cholesky(S_ridge + P).T
        
        # QR compression: instead of working with the full n×m matrix X, we
        # work with the compact m×m upper triangle R. The orthogonal matrix Q
        # preserves all information from X without any loss.
        if Q is None or R is None:
            Q, R = np.linalg.qr(X)

        min_nm = min(m, n)
        # SVD on the stacked matrix. This is equivalent to solving the augmented
        # system [X; E] beta = [y; 0], but does it in a numerically optimal way.
        # The "E rows" encode the penalty — they say "keep coefficients small."
        U, d, Vt = np.linalg.svd(np.vstack([R, E]), full_matrices=False)

        # U1 is just the top portion of U, corresponding to R (the data part).
        # The bottom portion would correspond to E (the penalty part) — we don't
        # need it for solving, but U1 is crucial for computing effective degrees
        # of freedom later: EDF = sum of squares of U1's elements.
        U1 = U[:min_nm, :min_nm]
        Vt = Vt[:min_nm]
        d_inv = 1.0 / d[:min_nm]

        # The solution: multiply from right to left for efficiency.
        # This is mathematically equivalent to beta = (X^T X + P)^{-1} X^T y,
        # but avoids ever forming X^T X or explicitly inverting anything.
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

        Why GCV instead of plain cross-validation?
        -------------------------------------------
        Leave-one-out CV says: "remove one data point, fit the model, predict that
        point, and see how far off you were." Doing this for every point is slow.
        GCV is a clever shortcut — it computes an approximation to the average
        leave-one-out error using only a single fit:

            GCV = (n * RSS) / (n - gamma * EDF)^2

        - RSS measures how well the curve fits the data (lower = tighter fit).
        - EDF (effective degrees of freedom) measures how flexible the curve is
          (higher = more wiggly, more prone to overfitting).
        - gamma (default 1.4) inflates the penalty on flexibility, making GCV
          prefer slightly simpler models — similar to AIC or BIC corrections.

        The denominator (n - gamma * EDF)^2 shrinks as EDF grows, so GCV
        skyrockets when the model is too flexible. Balancing RSS and EDF gives
        the "sweet spot" where the curve fits well without memorizing noise.

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

        This is the "standard" path — it uses SVD for maximum numerical robustness.
        The fast path (gridsearch_fast) trades some numerical stability for speed
        by using Cholesky decomposition instead.
        """
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        # Default search grids: 11 lambdas from 0.001 to 1000 (log-spaced),
        # and 10 spline counts from 5 to 23. Log-spacing for lambda makes sense
        # because its effect is multiplicative — doubling lambda roughly halves
        # the curve's wiggliness, so we want to search on a log scale.
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
            # Rebuild the design matrix for each candidate K because the number
            # of columns changes with the number of basis functions.
            self.n_splines = K
            Xmat = self._build_design_matrix(x)
            # Pre-compute QR once per K — it's the expensive part, and lambda
            # only affects the penalty, not the design matrix, so QR is reusable.
            Q, R = np.linalg.qr(Xmat)
            _, S_pen = self._build_penalty(1.0)
            m = K + 1

            def eval_lam(l_val: float) -> Tuple[float, int, float, np.ndarray, float]:
                # Build the full penalty P for this lambda. We scale S_pen (computed
                # at lambda=1) by l_val instead of rebuilding from scratch each time.
                P = np.zeros((m, m))
                P[:K, :K] = l_val * S_pen
                
                if robust:
                    # ---- Robust path: Huber IRLS within each grid candidate ----
                    # For the intuition behind these steps, see _fit_internal().
                    # Median initialization prevents the spline from anchoring to outliers.
                    med_y = np.median(y)
                    residuals = y - med_y
                    
                    coef, edf = None, 0.0
                    w = np.ones(n)
                    for _ in range(10): # IRLS: iteratively reweight until convergence
                        # MAD-based scale: robust measure of "typical" residual size,
                        # insensitive to the magnitude of extreme outliers.
                        mad = np.median(np.abs(residuals - np.median(residuals)))
                        scale_est = mad / 0.6745 if mad > 1e-6 else 1e-6
                        
                        # Huber weights: full weight for "normal" residuals, diminishing
                        # weight for outliers. The threshold c=1.345*scale gives
                        # 95% statistical efficiency on clean (Gaussian) data.
                        c = 1.345 * scale_est
                        abs_r = np.abs(residuals)
                        w = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-6))
                        
                        # Scale rows of X and elements of y by sqrt(w) to solve weighted PIRLS.
                        # This is the standard trick for weighted least squares:
                        # instead of (X^T W X) beta = X^T W y, we solve
                        # (X_w^T X_w) beta = X_w^T y_w, which is unweighted.
                        W_sqrt = np.sqrt(w).reshape(-1, 1)
                        X_w = Xmat * W_sqrt
                        y_w = y * np.sqrt(w)
                        
                        Qw, Rw = np.linalg.qr(X_w)
                        coef, U1, _ = self._solve_pirls(X_w, y_w, P, Q=Qw, R=Rw)
                        residuals = y - Xmat @ coef
                        
                        # EDF from SVD: each singular value contributes a fractional amount
                        # to the total degrees of freedom. Sum of U1^2 entries
                        # gives EDF — see _solve_pirls() for why U1 encodes this.
                        edf = np.sum(U1 ** 2)
                        
                    y_hat = Xmat @ coef
                    # Weighted RSS: only well-behaved points (w≈1) contribute
                    # significantly to the residual sum of squares.
                    rss = np.sum(w * (y - y_hat) ** 2)
                else:
                    coef, U1, _ = self._solve_pirls(Xmat, y, P, Q=Q, R=R)
                    y_hat = Xmat @ coef
                    rss = np.sum((y - y_hat) ** 2)
                    edf = np.sum(U1 ** 2)
                    
                # GCV score: balances fit quality (RSS) against model complexity (EDF).
                # Think of it as: "how much error per degree of freedom spent?"
                # A model that fits perfectly but uses too many DOF gets penalized.
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
        Fast grid search that multithreads BOTH n_splines (precomputing QR)
        and lam evaluations concurrently.

        Strategy — avoid redundant work across two phases:
        Phase 1: For each candidate K, build the design matrix and decompose it
        once (QR). Form R^T R and R^T Q^T y — these are independent of lambda,
        so they're computed once and reused for every lambda in phase 2.
        Phase 2: For each (K, lambda) pair, the only thing that changes is the
        penalty strength. The system (R^T R + lambda * P_base) can be solved
        via Cholesky in O(K^3) — much cheaper than full SVD per candidate.
        """
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        # Log-spaced lambda grid: each step roughly doubles the penalty strength,
        # which is more efficient than a linear grid for exploring orders of magnitude.
        if lam is None:
            lam = np.logspace(-3, 3, 11)
        if n_splines is None:
            n_splines = np.arange(5, 25, 2)

        self.edge_knots_ = np.array([x.min(), x.max()])
        n = len(y)

        best_gcv = np.inf
        best_result = None
        results: List[Tuple[float, int, float, np.ndarray, float]] = []

        # Phase 1: Precompute B-splines and QR decompositions for all K concurrently.
        # This is the expensive part — building the design matrix and decomposing it.
        # Since lambda only affects the penalty weight (not the design matrix), we
        # can reuse R^T R and R^T Q^T y for all lambda values at a given K.
        def precompute_K(K: int):
            Xmat = self._build_design_matrix(x, n_splines=K)
            Xmat_w = Xmat
            y_w = y
            
            Q, R = np.linalg.qr(Xmat_w)
            _, S_pen = self._build_penalty(1.0, n_splines=K)
            
            # These two quantities are the building blocks for ALL lambda values
            # at this K. R^T R replaces X^T X (same information, but from QR),
            # and R^T Q^T y replaces X^T y. This is why QR helps: it compresses
            # the n×m data into a compact m×m form.
            RtR = R.T @ R
            Rt_qty = R.T @ (Q.T @ y_w)
            
            P_base = np.zeros((K + 1, K + 1))
            P_base[:K, :K] = S_pen
            
            return K, Xmat, P_base, RtR, Rt_qty, K + 1

        with ThreadPoolExecutor() as executor:
            precomputed = list(executor.map(precompute_K, n_splines))

        # Phase 2: Evaluate all (K, lambda) candidates using precomputed quantities.
        # For each pair, we just solve a small linear system — no need to rebuild
        # or decompose anything. Cholesky is ~2x faster than LU for this.
        def eval_candidate(args) -> Tuple[float, int, float, np.ndarray, float]:
            K, l_val, Xmat, P_base, RtR, Rt_qty, m = args
            
            # The normal equations with penalty: (R^T R + lambda * P) beta = R^T Q^T y
            # This is equivalent to (X^T X + lambda * P) beta = X^T y, but using
            # the QR-compressed form avoids numerical issues with X^T X.
            B = RtR + l_val * P_base
            B.flat[::m+1] += 1e-12  # Tiny ridge for numerical stability: prevents
                                     # Cholesky from failing when B is nearly singular
            
            try:
                # Cholesky decomposition: factor B = L L^T, then solve two triangular
                # systems. We solve twice — once for the coefficients, once for EDF.
                c, lower = cho_factor(B, overwrite_a=True, check_finite=False)
                coef = cho_solve((c, lower), Rt_qty, check_finite=False)
                
                # Effective degrees of freedom = trace of the "hat matrix" restricted
                # to the data space. Intuitively, EDF measures how many "free
                # parameters" the smooth is actually using. A straight line has EDF~2,
                # a wildly wiggly curve has EDF close to K+1.
                B_inv_RtR = cho_solve((c, lower), RtR, check_finite=False)
                edf = np.trace(B_inv_RtR)
            except Exception:
                # Cholesky requires a positive-definite matrix. If lambda is very small
                # and the design matrix is ill-conditioned, B might not be positive-
                # definite enough. LU-based solve doesn't have this requirement.
                coef = np.linalg.solve(B, Rt_qty)
                edf = np.trace(np.linalg.solve(B, RtR))
                
            y_hat = Xmat @ coef
            rss = np.sum((y - y_hat) ** 2)
            # GCV formula: see gridsearch() docstring for intuition.
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
        """Internal fit: builds basis, solves PIRLS, computes all statistics.

        This is the core fitting routine. After constructing the B-spline design
        matrix and penalty, it either solves ordinary penalized least squares (the
        standard case) or runs an iterative robust fitting loop (Huber IRLS) that
        progressively down-weights outliers until convergence.
        """
        n = len(x)

        k = self.spline_order
        K = self.n_splines
        # Store knot positions in the original x-scale for plotting and prediction.
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
            # ---- Robust fitting via Huber IRLS ----
            # Ordinary least squares is very sensitive to outliers because squaring
            # a large residual makes it dominate the objective. Huber IRLS fixes
            # this by assigning lower weights to points with large residuals —
            # effectively telling the model "don't trust that point."
            #
            # The algorithm:
            # 1. Start with the global median (robust to outliers, unlike the mean).
            # 2. Measure how spread out the "typical" residuals are using MAD —
            #    the Median Absolute Deviation. MAD is like standard deviation but
            #    ignores extreme values. The factor 0.6745 makes MAD comparable to
            #    standard deviation for normally-distributed data.
            # 3. Determine a threshold c = 1.345 * scale. The magic constant 1.345
            #    makes Huber loss 95% as efficient as least squares on clean data.
            # 4. Points with |residual| < c keep full weight (w=1). Points beyond
            #    c get weight c/|residual|, which shrinks as the residual grows.
            # 5. Re-fit the spline with these weights, update residuals, and repeat.
            #    The weights push the curve away from outliers each iteration.
            med_y = np.median(y)
            residuals = y - med_y
            w = np.ones(n)
            
            for _ in range(15): # IRLS Loop
                # 2. Scale estimation using MAD
                mad = np.median(np.abs(residuals - np.median(residuals)))
                scale_est = mad / 0.6745 if mad > 1e-6 else 1e-6
                
                # 3. Compute Huber weights: full weight for "normal" points,
                #    diminishing weight for outliers (c/|residual|).
                c = 1.345 * scale_est
                abs_r = np.abs(residuals)
                w = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-6))
                
                # 4. Convert to weighted least squares by multiplying each row
                #    of X and each element of y by sqrt(w). This is mathematically
                #    identical to solving (X^T W X + P) beta = X^T W y, but it's
                #    cleaner to let the existing solver handle it via transformed data.
                W_sqrt = np.sqrt(w).reshape(-1, 1)
                X_w = X * W_sqrt
                y_w = y * np.sqrt(w)
                
                self.coef_, U1, B_solve = self._solve_pirls(X_w, y_w, P)
                residuals = y - X @ self.coef_
            
            self._B_solve = B_solve
            y_hat = X @ self.coef_
            # Weighted RSS: only points with w≈1 contribute significantly.
            rss = np.sum(w * (y - y_hat) ** 2)
        else:
            # ---- Standard fitting: solve (X^T X + P) beta = X^T y ----
            self.coef_, U1, B_solve = self._solve_pirls(X, y, P)
            self._B_solve = B_solve
            y_hat = X @ self.coef_
            rss = np.sum((y - y_hat) ** 2)

        # ---- Compute effective degrees of freedom (EDF) ----
        # EDF tells us how many "free parameters" the fitted smooth is using.
        # Unlike linear regression (where EDF = number of predictors), a penalized
        # spline's EDF is fractional — each coefficient is "shrunk" by the penalty,
        # so it doesn't use a full degree of freedom. We get it from U1 (the data-
        # portion of the SVD): each singular value contributes a fraction between 0
        # and 1 to the total EDF.
        edof_per_coef = np.sum(U1 ** 2, axis=1)
        edof = np.sum(edof_per_coef)

        # ---- Compute residual standard error and covariance ----
        # The residual standard error sigma_hat = sqrt(RSS / (n - EDF)) estimates
        # the spread of data points around the fitted curve. Dividing by (n - EDF)
        # instead of (n - K - 1) accounts for the fact that the penalty has already
        # "used up" some flexibility — we don't want to double-count.
        scale = np.sqrt(rss / (n - edof))
        # The covariance of beta tells us how uncertain each coefficient estimate is.
        # B_solve is like the "inverse" of (X^T X + P) — it maps any y to beta.
        # Its outer product B_solve @ B_solve^T gives us the "shape" of the
        # uncertainty, and we scale it by sigma_hat^2 to get the magnitude right.
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
        """Compute confidence or prediction intervals at each x point.

        Two types of uncertainty bands:
        - Confidence interval: "How uncertain is the *average* prediction at this x?"
          This is only about the curve itself — narrower, because averaging many
          observations at the same x reduces noise.
        - Prediction interval: "How uncertain is a *single future observation* at this x?"
          Wider, because it includes both the curve uncertainty AND the irreducible
          scatter of individual points around the curve (sigma^2).

        Both use the t-distribution rather than the normal distribution because we're
        estimating sigma from the data (we don't know it exactly). The t-distribution
        has fatter tails than the normal — this widens the intervals to honestly reflect
        our uncertainty about sigma. The degrees of freedom are (n - EDF) rather than
        (n - k) because the penalty has already consumed EDF degrees of freedom.
        """
        self._check_is_fitted()
        x = np.asarray(x).ravel()
        X = self._build_design_matrix(x)

        lp = X @ self.coef_
        cov = self.statistics_['cov']

        # Variance of the predicted mean at each x: X cov X^T gives the uncertainty
        # in the fitted curve. Each row of X picks out a "slice" of the covariance
        # matrix corresponding to that prediction point, and we sum across columns
        # to get a scalar variance per point.
        var = np.sum((X @ cov) * X, axis=1)

        if prediction:
            # For prediction intervals, add the observation noise variance sigma^2.
            # This makes the band wider because individual points scatter around
            # the fitted curve, and we need to account for that scatter.
            var += self.statistics_['scale'] ** 2

        alpha = (1.0 - width) / 2.0
        # Use (n - EDF) degrees of freedom: the penalty has already "spent" some
        # degrees of freedom, so we subtract EDF from the sample size to get the
        # residual degrees of freedom for the t-distribution.
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