"""
gamcore.py — Multi-term Generalized Additive Model with formula interface.

Supports s() spline, te() tensor-product, f() factor, and l() linear terms.
Uses the same numerically robust SVD/QR+Cholesky solver and Huber IRLS engine
as linGAM, extended to block-diagonal multi-term penalties.

Example:
    model = GAMCore("s(0, n_splines=10) + te(1, 2, n_splines=5) + f(3)")
    model.gridsearch(X, y)
    y_hat = model.predict(X)
    pdep_0 = model.partial_dependence(0, X)
"""

import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, block_diag
from typing import Optional, Tuple, Dict, Any, List, Union
from numpy.typing import ArrayLike


# ==============================================================================
# Formula Parser
# ==============================================================================

def _parse_formula(formula: str) -> List[Dict[str, Any]]:
    """
    Parse a GAM formula string into a list of term configuration dicts.

    Supported syntax:
        s(feature, n_splines=10, lam=1.0)
        te(feature, feature, n_splines=5, lam=0.6)
        te(feature, feature, n_splines=[5, 8])
        f(feature, lam=0.5, coding='one-hot')
        l(feature, lam=0.1)

    Terms are separated by ``+``.  Positional arguments are feature indices;
    keyword arguments are passed through as-is (values auto-parsed via
    ``ast.literal_eval``).
    """
    if not formula or not formula.strip():
        raise ValueError("Formula string is empty.")

    # Remove whitespace (preserve it inside strings for ast.literal_eval)
    stripped = formula.strip()

    # Split on top-level '+', respecting parentheses
    chunks: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in stripped:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '+' and depth == 0:
            chunks.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        chunks.append(''.join(current))

    result: List[Dict[str, Any]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        m = re.match(r'(\w+)\s*\(([^)]*)\)', chunk)
        if not m:
            raise ValueError(f"Cannot parse term '{chunk}'. Expected e.g. s(0, n_splines=10).")

        func_name = m.group(1).lower()
        if func_name not in ('s', 'te', 'f', 'l'):
            raise ValueError(f"Unknown term type '{func_name}'. Expected s, te, f, or l.")

        args_str = m.group(2).strip()

        # Split arguments on commas (respecting brackets for list literals)
        arg_parts = _split_top_level_commas(args_str) if args_str else []

        positional: List[Any] = []
        kwargs: Dict[str, Any] = {}
        for part in arg_parts:
            part = part.strip()
            if '=' in part:
                key, val = part.split('=', 1)
                key = key.strip()
                val = val.strip()
                try:
                    kwargs[key] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    kwargs[key] = val
            else:
                try:
                    positional.append(ast.literal_eval(part))
                except (ValueError, SyntaxError):
                    positional.append(part)

        # Build feature list from positional args
        if func_name in ('s', 'f', 'l'):
            if len(positional) < 1:
                raise ValueError(f"'{func_name}()' requires at least one feature index.")
            features = [positional[0]]
        else:  # te
            if len(positional) < 2:
                raise ValueError(f"'te()' requires at least two feature indices.")
            features = list(positional)

        result.append({'type': func_name, 'features': features, 'kwargs': kwargs})

    return result


def _split_top_level_commas(s: str) -> List[str]:
    """Split on commas that are not inside brackets [ ]."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in s:
        if ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


# ==============================================================================
# Standalone B-Spline Basis (extracted from linGAM)
# ==============================================================================

def _b_spline_basis(x_in: ArrayLike, n_splines: int, spline_order: int,
                    edge_knots: np.ndarray) -> np.ndarray:
    """Construct B-spline basis matrix using vectorized De Boor recursion.

    Parameters
    ----------
    x_in : array-like, shape (n,)
        Predictor values at which to evaluate the basis.
    n_splines : int
        Number of basis functions K. Must satisfy K > spline_order.
    spline_order : int
        Order k of the B-spline (polynomial degree).
    edge_knots : np.ndarray, shape (2,)
        Domain boundaries [min, max] for mapping data to [0, 1].

    Returns
    -------
    bases : np.ndarray, shape (n, K)
        B-spline basis matrix.
    """
    K = n_splines
    k = spline_order

    # Map data to [0, 1]
    offset = edge_knots[0]
    scale = edge_knots[1] - edge_knots[0]
    if scale == 0.0:
        scale = 1.0

    # Place internal knots evenly across [0, 1]
    boundary_knots = np.linspace(0.0, 1.0, 1 + K - k)
    diff = boundary_knots[1] - boundary_knots[0]

    x = (np.asarray(x_in).ravel() - offset) / scale
    # Append sentinel values for boundary gradient computation
    x = np.r_[x, 0.0, 1.0]

    # Track extrapolated points
    x_extrap_l = x < 0.0
    x_extrap_r = x > 1.0
    x_interp = ~(x_extrap_l | x_extrap_r)
    x = np.atleast_2d(x).T

    # Extend knots beyond [0, 1]
    aug = np.arange(1, k + 1) * diff
    aug_knots = np.r_[-aug[::-1], boundary_knots, 1.0 + aug]
    aug_knots[-1] += 1e-9

    # Order-1 (piecewise constant) basis
    bases = ((x >= aug_knots[:-1]) & (x < aug_knots[1:])).astype(float)
    bases[-1] = bases[-2][::-1]

    n_active = len(aug_knots) - 1
    prev_bases = None

    # De Boor recursion
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

    # Linear extrapolation using boundary gradients
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


# ==============================================================================
# Penalty builders (per term type)
# ==============================================================================

def _second_diff_penalty(K: int, lam: float) -> np.ndarray:
    """Build the second-difference penalty S = D2ᵀD2 scaled by lam."""
    D2 = np.zeros((K - 2, K))
    idx = np.arange(K - 2)
    D2[idx, idx] = 1.0
    D2[idx, idx + 1] = -2.0
    D2[idx, idx + 2] = 1.0
    return lam * (D2.T @ D2)


def _ridge_penalty(K: int, lam: float) -> np.ndarray:
    """L2 ridge penalty: lam * I_K."""
    return lam * np.eye(K)


# ==============================================================================
# Solver functions (extracted from linGAM)
# ==============================================================================

def _solve_pirls(X: np.ndarray, y: np.ndarray, P: np.ndarray,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve penalized least squares: (XᵀX + P) beta = Xᵀy via SVD data augmentation.

    Returns (coef, U1, B_solve) where B_solve is the solution operator used for
    covariance and EDF computation.
    """
    n, m = X.shape
    eps = np.finfo(float).eps

    # Ridge on P ensures Cholesky succeeds even when P = 0
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


def _irls_solve(X: np.ndarray, y: np.ndarray, P: np.ndarray,
                n_iter: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Huber-weighted Iteratively Reweighted Least Squares.

    Iteratively down-weights outliers using the Huber loss. Returns
    (coef, U1, B_solve, weights, residuals).
    """
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
        coef, U1, B_solve = _solve_pirls(X_w, y_w, P, Q=Qw, R=Rw)
        residuals = y - X @ coef

    return coef, U1, B_solve, w, residuals


# ==============================================================================
# Internal Term Classes
# ==============================================================================

class _Term:
    """Base class for all GAM terms."""
    n_coefs: int = 0
    _lam: float = 1.0

    def compile(self, x: np.ndarray) -> None:
        """One-time setup using training data (called before first fit)."""
        pass

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        """Build the design matrix columns for this term."""
        raise NotImplementedError


class _SplineTerm(_Term):
    """Univariate penalized B-spline smooth term."""
    def __init__(self, feature: int, n_splines: int = 10, lam: float = 1.0,
                 spline_order: int = 3):
        self.feature = feature
        self.n_splines = n_splines
        self.lam = lam
        self.spline_order = spline_order
        self.n_coefs = n_splines
        self._edge_knots: Optional[np.ndarray] = None

    def compile(self, x: np.ndarray) -> None:
        col = np.asarray(x[:, self.feature]).ravel()
        self._edge_knots = np.array([col.min(), col.max()])

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        return _b_spline_basis(x[:, self.feature], self.n_splines,
                               self.spline_order, self._edge_knots)

    def build_penalty(self, lam: Optional[float] = None) -> np.ndarray:
        """Build the penalty matrix for this term (scaled by lam)."""
        return _second_diff_penalty(self.n_splines, lam if lam is not None else self.lam)

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        """Return unscaled penalty components that can be independently scaled.

        For a spline term: one base matrix (D2ᵀD2 at lam=1), one lam slot.
        """
        return [(_second_diff_penalty(self.n_splines, 1.0), 1)]


class _TensorTerm(_Term):
    """Tensor-product B-spline interaction term (2+D)."""
    def __init__(self, features: List[int], n_splines: Union[int, List[int]] = 5,
                 lam: Union[float, List[float]] = 1.0, spline_order: int = 3):
        self.features = list(features)
        self.d = len(self.features)
        if self.d < 2:
            raise ValueError("Tensor term requires at least 2 features.")

        # Broadcast n_splines to per-marginal list
        if isinstance(n_splines, int):
            self.n_splines_per = [n_splines] * self.d
        else:
            if len(n_splines) != self.d:
                raise ValueError(f"n_splines length ({len(n_splines)}) must match "
                                 f"number of features ({self.d}).")
            self.n_splines_per = list(n_splines)

        # Broadcast lam to per-marginal list
        if isinstance(lam, (int, float)):
            self.lam_per = [float(lam)] * self.d
        else:
            if len(lam) != self.d:
                raise ValueError(f"lam length ({len(lam)}) must match "
                                 f"number of features ({self.d}).")
            self.lam_per = [float(v) for v in lam]

        self.lam = self.lam_per[0]  # for display
        self.spline_order = spline_order
        self._n_coefs_per: List[int] = self.n_splines_per.copy()
        self.n_coefs = int(np.prod(self._n_coefs_per))
        self._edge_knots: List[np.ndarray] = []

    def compile(self, x: np.ndarray) -> None:
        self._edge_knots = []
        for feat in self.features:
            col = np.asarray(x[:, feat]).ravel()
            self._edge_knots.append(np.array([col.min(), col.max()]))

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        # Build marginal B-spline bases
        marginals = []
        for i, feat in enumerate(self.features):
            B_i = _b_spline_basis(x[:, feat], self.n_splines_per[i],
                                  self.spline_order, self._edge_knots[i])
            marginals.append(B_i)
        return self._khatri_rao(marginals)

    @staticmethod
    def _khatri_rao(marginals: List[np.ndarray]) -> np.ndarray:
        """Column-wise Kronecker (Khatri-Rao) product of marginals.
        For d marginals B0 (n×K0), ..., Bd-1 (n×Kd-1), returns n × prod(Ki).
        Each column is the element-wise product of one column from each marginal.
        """
        n = marginals[0].shape[0]
        # Expand each marginal with singleton axes and broadcast-multiply
        result = marginals[0]
        for i in range(1, len(marginals)):
            # Add trailing singleton axes to result
            result = result[..., :, None]  # shape: n × (prod_{j<i} Kj) × 1
            # Add leading singleton axes to next marginal
            B_next = marginals[i][:, None, :]  # shape: n × 1 × Ki
            result = result * B_next
            # Flatten the product axes
            result = result.reshape(n, -1)
        return result

    def build_penalty(self, lam: Optional[List[float]] = None) -> np.ndarray:
        """Kronecker sum of marginal penalties."""
        lams = lam if lam is not None else self.lam_per
        if isinstance(lams, (int, float)):
            lams = [float(lams)] * self.d

        marginal_penalties = []
        for i in range(self.d):
            P_i = _second_diff_penalty(self.n_splines_per[i], lams[i])
            marginal_penalties.append(P_i)

        return self._kron_sum(marginal_penalties)

    @staticmethod
    def _kron_sum(penalties: List[np.ndarray]) -> np.ndarray:
        """Compute Kronecker sum: Σᵢ I₀⊗...⊗Pᵢ⊗...⊗I_{d-1}."""
        d = len(penalties)
        sizes = [p.shape[0] for p in penalties]
        total = int(np.prod(sizes))
        result = np.zeros((total, total))
        for i, P_i in enumerate(penalties):
            I_pre = np.eye(int(np.prod(sizes[:i])))
            I_post = np.eye(int(np.prod(sizes[i + 1:])))
            result += np.kron(np.kron(I_pre, P_i), I_post)
        return result

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        """Return per-marginal Kronecker products (unscaled), one per lam slot.

        For a 2D tensor te(0,1) with n_splines=[K0, K1]:
            base_0 = kron(D0, I1)  — penalise direction 0
            base_1 = kron(I0, D1)  — penalise direction 1
        Each scaled by its own lambda.
        """
        bases = []
        sizes = list(self.n_splines_per)
        for i in range(self.d):
            D2_i = np.zeros((sizes[i] - 2, sizes[i]))
            idx = np.arange(sizes[i] - 2)
            D2_i[idx, idx] = 1.0
            D2_i[idx, idx + 1] = -2.0
            D2_i[idx, idx + 2] = 1.0
            S_i = D2_i.T @ D2_i
            I_pre = np.eye(int(np.prod(sizes[:i])))
            I_post = np.eye(int(np.prod(sizes[i + 1:])))
            bases.append((np.kron(np.kron(I_pre, S_i), I_post), 1))
        return bases


class _FactorTerm(_Term):
    """Categorical / factor term with one-hot encoding."""
    def __init__(self, feature: int, lam: float = 1.0, coding: str = 'one-hot'):
        self.feature = feature
        self.lam = lam
        self.coding = coding  # 'one-hot' or 'dummy'
        self._levels: Optional[np.ndarray] = None
        self.n_coefs = 0

    def compile(self, x: np.ndarray) -> None:
        col = np.asarray(x[:, self.feature]).ravel()
        self._levels = np.unique(col)
        self.n_coefs = len(self._levels) if self.coding == 'one-hot' else len(self._levels) - 1
        if self.n_coefs <= 0:
            raise ValueError(f"Factor term feature {self.feature} has no levels.")

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        col = np.asarray(x[:, self.feature]).ravel()
        B = (col[:, None] == self._levels[None, :]).astype(float)
        if self.coding == 'dummy':
            B = B[:, 1:]  # drop first level
        return B

    def build_penalty(self, lam: Optional[float] = None) -> np.ndarray:
        return _ridge_penalty(self.n_coefs, lam if lam is not None else self.lam)

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        return [(np.eye(self.n_coefs), 1)]  # ridge: lam * I


class _LinearTerm(_Term):
    """Linear (unpenalized by default) term."""
    def __init__(self, feature: int, lam: float = 0.0):
        self.feature = feature
        self.lam = lam
        self.n_coefs = 1

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x[:, self.feature]).ravel().reshape(-1, 1)

    def build_penalty(self, lam: Optional[float] = None) -> np.ndarray:
        l = lam if lam is not None else self.lam
        return np.array([[l]], dtype=float)

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        return [(np.array([[1.0]]), 1)]  # lam * [1]


# ==============================================================================
# GAMCore — the main model class
# ==============================================================================

class GAMCore:
    """Multi-term Generalized Additive Model with formula interface.

    Parameters
    ----------
    formula : str
        Model specification, e.g. ``"s(0, n_splines=10) + te(1, 2) + f(3) + l(4)"``.
    spline_order : int, default 3
        B-spline order (polynomial degree) for all spline/te terms. Individual
        terms can override via ``spline_order=...`` in the formula.
    fit_intercept : bool, default True
        Whether to include an unpenalized intercept term.
    """

    def __init__(self, formula: str, spline_order: int = 3, fit_intercept: bool = True):
        self.formula = formula
        self.spline_order = spline_order
        self.fit_intercept = fit_intercept

        # Parse formula into term configs, then instantiate term objects
        configs = _parse_formula(formula)
        self._terms: List[_Term] = []
        self._term_configs = configs  # store for grid search rebuilds

        for cfg in configs:
            t = cfg['type']
            feats = cfg['features']
            kw = cfg['kwargs']

            # Allow per-term spline_order override
            term_order = kw.pop('spline_order', spline_order)

            if t == 's':
                term = _SplineTerm(
                    feature=feats[0],
                    n_splines=kw.pop('n_splines', 10),
                    lam=kw.pop('lam', 1.0),
                    spline_order=term_order,
                )
            elif t == 'te':
                term = _TensorTerm(
                    features=feats,
                    n_splines=kw.pop('n_splines', 5),
                    lam=kw.pop('lam', 1.0),
                    spline_order=term_order,
                )
            elif t == 'f':
                term = _FactorTerm(
                    feature=feats[0],
                    lam=kw.pop('lam', 1.0),
                    coding=kw.pop('coding', 'one-hot'),
                )
            elif t == 'l':
                term = _LinearTerm(
                    feature=feats[0],
                    lam=kw.pop('lam', 0.0),
                )
            else:
                raise ValueError(f"Unknown term type: {t}")

            if kw:
                raise ValueError(f"Unrecognized kwargs for {t}(): {list(kw.keys())}")

            self._terms.append(term)

        self._coef_slices: List[Tuple[int, int]] = []
        self.coef_: Optional[np.ndarray] = None
        self.statistics_: Optional[Dict[str, Any]] = None
        self._B_solve: Optional[np.ndarray] = None
        self._design_matrix_: Optional[np.ndarray] = None

    # -- Properties -------------------------------------------------------

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
        return f"GAMCore(formula='{self.formula}') [{status}]"

    # -- Compile ----------------------------------------------------------

    def _compile(self, x: np.ndarray) -> None:
        """Compile all terms: determine levels, edge knots, etc."""
        for term in self._terms:
            term.compile(x)

    # -- Design matrix & penalty assembly ---------------------------------

    def _total_coefs(self) -> int:
        total = sum(t.n_coefs for t in self._terms)
        if self.fit_intercept:
            total += 1
        return total

    def _build_model_matrix(self, x: np.ndarray) -> np.ndarray:
        """Horizontally concatenate term design matrices + optional intercept."""
        blocks = [t.build_columns(x) for t in self._terms]
        if self.fit_intercept:
            blocks.append(np.ones((len(x), 1)))
        return np.hstack(blocks)

    def _build_penalty(self) -> np.ndarray:
        """Build block-diagonal penalty matrix from all terms."""
        blocks = [t.build_penalty() for t in self._terms]
        if self.fit_intercept:
            blocks.append(np.zeros((1, 1)))
        if len(blocks) == 1:
            return blocks[0]
        return block_diag(*blocks)

    def _compute_coef_slices(self) -> None:
        """Record (start, end) index into coef_ for each term."""
        slices = []
        offset = 0
        for t in self._terms:
            slices.append((offset, offset + t.n_coefs))
            offset += t.n_coefs
        # Intercept is not a "term" for partial_dependence purposes
        self._coef_slices = slices

    # -- Fit --------------------------------------------------------------

    def fit(self, x: ArrayLike, y: ArrayLike, robust: bool = False) -> "GAMCore":
        """Fit using current hyper-parameters.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Predictor matrix.
        y : array-like, shape (n,)
            Response vector.
        robust : bool, default False
            Use Huber IRLS to down-weight outliers.

        Returns
        -------
        GAMCore
            Fitted model instance.
        """
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        self._compile(x_arr)
        self._fit_internal(x_arr, y_arr, robust=robust)
        return self

    def _fit_internal(self, x: np.ndarray, y: np.ndarray, robust: bool = False) -> None:
        """Core fit routine: build design matrix + penalty, solve, compute stats."""
        n = len(x)
        X = self._build_model_matrix(x)
        self._design_matrix_ = X
        P = self._build_penalty()
        self._compute_coef_slices()

        if robust:
            coef, U1, B_solve, w, _ = _irls_solve(X, y, P, n_iter=15)
            y_hat = X @ coef
            rss = np.sum(w * (y - y_hat) ** 2)
        else:
            coef, U1, B_solve = _solve_pirls(X, y, P)
            y_hat = X @ coef
            rss = np.sum((y - y_hat) ** 2)

        self.coef_ = coef
        self._B_solve = B_solve

        # Effective degrees of freedom
        edof_per_coef = np.sum(U1 ** 2, axis=1)
        edof = np.sum(edof_per_coef)

        # Residual scale and covariance
        scale = np.sqrt(rss / max(n - edof, 1))
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
            'GCV': (n * rss) / max(n - 1.4 * edof, 1) ** 2,
        }

    # -- Predict -----------------------------------------------------------

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict response at new predictor values."""
        self._check_fitted()
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        X = self._build_model_matrix(x_arr)
        return X @ self.coef_

    def partial_dependence(self, term_idx: int, x: ArrayLike) -> np.ndarray:
        """Compute the additive contribution of a single term.

        Parameters
        ----------
        term_idx : int
            Zero-based index into the formula terms (order as written).
        x : array-like, shape (n, p)
            Predictor matrix.

        Returns
        -------
        pdep : np.ndarray, shape (n,)
            The term's contribution to the linear predictor.
        """
        self._check_fitted()
        if term_idx < 0 or term_idx >= len(self._terms):
            raise IndexError(f"term_idx {term_idx} out of range (0..{len(self._terms) - 1}).")

        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        # Build only this term's columns
        B_term = self._terms[term_idx].build_columns(x_arr)
        start, end = self._coef_slices[term_idx]
        return B_term @ self.coef_[start:end]

    # -- Intervals ---------------------------------------------------------

    def confidence_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        """Confidence intervals for the mean response."""
        return self._compute_intervals(x, width, prediction=False)

    def prediction_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        """Prediction intervals for new observations."""
        return self._compute_intervals(x, width, prediction=True)

    def _compute_intervals(self, x: np.ndarray, width: float, prediction: bool) -> np.ndarray:
        self._check_fitted()
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        X = self._build_model_matrix(x_arr)
        lp = X @ self.coef_

        cov = self.statistics_['cov']
        var = np.sum((X @ cov) * X, axis=1)
        if prediction:
            var += self.statistics_['scale'] ** 2

        alpha = (1.0 - width) / 2.0
        df = max(self.statistics_['n_samples'] - self.statistics_['edof'], 1)
        q_lo = stats.t.ppf(alpha, df=df)
        q_hi = stats.t.ppf(1.0 - alpha, df=df)

        se = np.sqrt(var)
        return np.column_stack([lp + q_lo * se, lp + q_hi * se])

    # -- Grid Search -------------------------------------------------------

    def gridsearch(self, x: ArrayLike, y: ArrayLike,
                   lam_grids: Optional[List[ArrayLike]] = None,
                   n_splines_grids: Optional[List[ArrayLike]] = None,
                   gamma: float = 1.4, fast: bool = True,
                   robust: bool = False) -> "GAMCore":
        """Automatically select optimal n_splines and lam via GCV.

        Parameters
        ----------
        x, y : array-like
            Training data.
        lam_grids : list of array-like, optional
            Per-term candidate lambda arrays. Defaults to logspace(-3, 3, 7 or 11).
        n_splines_grids : list of array-like, optional
            Per-term candidate n_splines arrays (only for s/te terms).
            Defaults to arange(5, 25, 2) for s, arange(4, 8) for te.
        gamma : float, default 1.4
            EDF penalty multiplier in GCV denominator.
        fast : bool, default True
            Use QR+Cholesky fast path (2-phase multi-threaded).
        robust : bool, default False
            Use Huber IRLS for evaluation.

        Returns
        -------
        GAMCore
            Fitted model with best hyper-parameters.
        """
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        if fast and not robust:
            return self._gridsearch_fast(x_arr, y_arr, lam_grids=lam_grids,
                                         n_splines_grids=n_splines_grids, gamma=gamma)
        return self._gridsearch_basic(x_arr, y_arr, lam_grids=lam_grids,
                                      n_splines_grids=n_splines_grids,
                                      gamma=gamma, robust=robust)

    def _build_default_grids(self, x: np.ndarray,
                             lam_grids: Optional[List[ArrayLike]],
                             n_splines_grids: Optional[List[ArrayLike]]):
        """Fill in / broadcast candidate grids to match per-slot counts."""
        # Count total lam / n_splines slots across all terms
        n_lam_slots = sum(
            len(t.lam_per) if isinstance(t, _TensorTerm) else 1
            for t in self._terms
        )
        n_nsp_slots = sum(
            len(t.n_splines_per) if isinstance(t, _TensorTerm) else 1
            for t in self._terms if isinstance(t, (_SplineTerm, _TensorTerm))
        )

        # Broadcast or fill lam_grids
        if lam_grids is None:
            lam_grids = [np.logspace(-3, 3, 11)] * n_lam_slots
        elif not isinstance(lam_grids, list):
            lam_grids = [np.atleast_1d(lam_grids)] * n_lam_slots
        elif len(lam_grids) == 1 and n_lam_slots > 1:
            lam_grids = lam_grids * n_lam_slots
        elif len(lam_grids) != n_lam_slots:
            raise ValueError(
                f"lam_grids has {len(lam_grids)} elements, "
                f"but model requires {n_lam_slots} lam slots "
                f"({self.formula}). Provide one grid per lam slot "
                f"or a single grid to broadcast."
            )

        # Broadcast or fill n_splines_grids (only s / te terms)
        if n_splines_grids is None:
            nsp = []
            for t in self._terms:
                if isinstance(t, _SplineTerm):
                    nsp.append(np.arange(5, 25, 2))
                elif isinstance(t, _TensorTerm):
                    nsp.extend([np.arange(4, 8)] * t.d)
            n_splines_grids = nsp
        elif not isinstance(n_splines_grids, list):
            n_splines_grids = [np.atleast_1d(n_splines_grids)] * max(n_nsp_slots, 1)
        elif len(n_splines_grids) == 1 and n_nsp_slots > 1:
            n_splines_grids = n_splines_grids * n_nsp_slots
        elif len(n_splines_grids) != n_nsp_slots:
            raise ValueError(
                f"n_splines_grids has {len(n_splines_grids)} elements, "
                f"but model requires {n_nsp_slots} n_splines slots "
                f"({self.formula}). Provide one grid per spline dimension "
                f"or a single grid to broadcast."
            )

        # Compile to get edge_knots for all terms (to build design matrices)
        self._compile(x)

        return lam_grids, n_splines_grids

    def _gridsearch_fast(self, x: np.ndarray, y: np.ndarray,
                         lam_grids: Optional[List[ArrayLike]] = None,
                         n_splines_grids: Optional[List[ArrayLike]] = None,
                         gamma: float = 1.4) -> "GAMCore":
        """Two-phase fast grid search with precomputed penalty bases.

        Phase 1 builds QR + unscaled penalty bases per K-combo (threaded).
        Phase 2 evaluates all (K, lam) pairs in parallel via Cholesky.
        Penalty matrices are assembled by scaling precomputed bases — no
        D2ᵀD2 rebuild per lam value.
        """
        lam_grids, n_splines_grids = self._build_default_grids(x, lam_grids, n_splines_grids)
        n = len(y)

        from itertools import product as cartesian_product

        nsp_idx = 0
        k_combo_elements = []
        for t in self._terms:
            if isinstance(t, _SplineTerm):
                grid = np.atleast_1d(n_splines_grids[nsp_idx])
                k_combo_elements.append([int(v) for v in grid])
                nsp_idx += 1
            elif isinstance(t, _TensorTerm):
                grids = [np.atleast_1d(g) for g in n_splines_grids[nsp_idx:nsp_idx + t.d]]
                combos = [tuple(int(v) for v in c) for c in cartesian_product(*grids)]
                k_combo_elements.append(combos)
                nsp_idx += t.d
            else:
                k_combo_elements.append([None])

        all_k_combos = list(cartesian_product(*k_combo_elements))
        lam_combos = list(cartesian_product(*lam_grids))

        # Phase 1: precompute QR + penalty bases per K-combo (threaded)
        def precompute_k(k_combo):
            configs_copy = [dict(cfg) for cfg in self._term_configs]
            for ci, cfg in enumerate(configs_copy):
                k_val = k_combo[ci]
                if cfg['type'] == 's' and k_val is not None:
                    cfg['kwargs']['n_splines'] = k_val
                elif cfg['type'] == 'te' and k_val is not None:
                    cfg['kwargs']['n_splines'] = list(k_val) if len(k_val) > 1 else k_val[0]

            terms_copy = GAMCore._instantiate_terms(configs_copy, self.spline_order)
            for t in terms_copy:
                t.compile(x)

            X = GAMCore._build_matrix_from_terms(terms_copy, x, self.fit_intercept)
            Q, R = np.linalg.qr(X)
            RtR = R.T @ R
            Rt_qty = R.T @ (Q.T @ y)

            bases_list, m = GAMCore._precompute_penalty_bases(terms_copy, self.fit_intercept)
            return X, RtR, Rt_qty, bases_list, m, terms_copy

        with ThreadPoolExecutor() as executor:
            precomputed = list(executor.map(precompute_k, all_k_combos))

        # Phase 2: evaluate all (K, lam) pairs in parallel
        n_k = len(precomputed)
        n_l = len(lam_combos)
        n_tasks = n_k * n_l

        def eval_task(idx):
            k_idx = idx // n_l
            l_idx = idx % n_l
            X_pc, RtR, Rt_qty, bases_list, m, terms_pc = precomputed[k_idx]
            l_combo = lam_combos[l_idx]

            # Assemble P by scaling precomputed bases — no D2 rebuild
            P_full = GAMCore._assemble_penalty_from_bases(bases_list, l_combo, m, self.fit_intercept)

            B = RtR + P_full
            B.flat[::m + 1] += 1e-12
            try:
                c, lower = cho_factor(B, overwrite_a=True, check_finite=False)
                coef = cho_solve((c, lower), Rt_qty, check_finite=False)
                B_inv_RtR = cho_solve((c, lower), RtR, check_finite=False)
                edf = np.trace(B_inv_RtR)
            except Exception:
                coef = np.linalg.solve(B, Rt_qty)
                edf = np.trace(np.linalg.solve(B, RtR))

            y_hat = X_pc @ coef
            rss = np.sum((y - y_hat) ** 2)
            gcv = (n * rss) / max(n - gamma * edf, 1) ** 2
            return gcv, k_idx, l_combo

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(eval_task, range(n_tasks)))

        best_gcv = np.inf
        best_k_idx = 0
        best_l_combo = None
        for gcv_val, k_idx, l_combo in results:
            if gcv_val < best_gcv:
                best_gcv = gcv_val
                best_k_idx = k_idx
                best_l_combo = l_combo

        _, _, _, _, _, best_terms = precomputed[best_k_idx]
        self._terms = best_terms
        self._set_lams_from_combo(best_l_combo)
        self._compile(x)
        self._fit_internal(x, y, robust=False)
        return self

    def _gridsearch_basic(self, x: np.ndarray, y: np.ndarray,
                          lam_grids: Optional[List[ArrayLike]] = None,
                          n_splines_grids: Optional[List[ArrayLike]] = None,
                          gamma: float = 1.4, robust: bool = False) -> "GAMCore":
        """Basic grid search: SVD solve per candidate, multithreaded over lam.

        Uses precomputed penalty bases (no D2 rebuild per lam) and avoids
        mutating term objects (thread-safe for parallel evaluation).
        """
        lam_grids, n_splines_grids = self._build_default_grids(x, lam_grids, n_splines_grids)
        n = len(y)

        from itertools import product as cartesian_product

        nsp_idx = 0
        k_combo_elements = []
        for t in self._terms:
            if isinstance(t, _SplineTerm):
                grid = np.atleast_1d(n_splines_grids[nsp_idx])
                k_combo_elements.append([int(v) for v in grid])
                nsp_idx += 1
            elif isinstance(t, _TensorTerm):
                grids = [np.atleast_1d(g) for g in n_splines_grids[nsp_idx:nsp_idx + t.d]]
                combos = [tuple(int(v) for v in c) for c in cartesian_product(*grids)]
                k_combo_elements.append(combos)
                nsp_idx += t.d
            else:
                k_combo_elements.append([None])

        all_k_combos = list(cartesian_product(*k_combo_elements))
        lam_combos = list(cartesian_product(*lam_grids))

        best_gcv = np.inf
        best_result = None

        for k_combo in all_k_combos:
            configs_copy = [dict(cfg) for cfg in self._term_configs]
            for ci, cfg in enumerate(configs_copy):
                k_val = k_combo[ci]
                if cfg['type'] == 's' and k_val is not None:
                    cfg['kwargs']['n_splines'] = k_val
                elif cfg['type'] == 'te' and k_val is not None:
                    cfg['kwargs']['n_splines'] = list(k_val) if len(k_val) > 1 else k_val[0]

            terms_k = GAMCore._instantiate_terms(configs_copy, self.spline_order)
            for t in terms_k:
                t.compile(x)

            X_full = GAMCore._build_matrix_from_terms(terms_k, x, self.fit_intercept)
            Q, R_ = np.linalg.qr(X_full) if not robust else (None, None)

            # Precompute penalty bases and offsets (no mutation in eval_lam)
            bases_list, m = GAMCore._precompute_penalty_bases(terms_k, self.fit_intercept)

            def eval_lam(l_combo):
                # Thread-safe: assemble P from precomputed bases, no term mutation
                P_full = GAMCore._assemble_penalty_from_bases(bases_list, l_combo, m, self.fit_intercept)

                if robust:
                    coef, U1, _, _, _ = _irls_solve(X_full, y, P_full, n_iter=10)
                else:
                    coef, U1, _ = _solve_pirls(X_full, y, P_full, Q=Q, R=R_)

                y_hat = X_full @ coef
                rss = np.sum((y - y_hat) ** 2)
                edf = np.sum(U1 ** 2)
                gcv = (n * rss) / max(n - gamma * edf, 1) ** 2
                return gcv, coef, edf, l_combo

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(eval_lam, lam_combos))

            for gcv_val, coef_val, edf_val, l_combo in results:
                if gcv_val < best_gcv:
                    best_gcv = gcv_val
                    best_result = (gcv_val, k_combo, l_combo, coef_val, edf_val, terms_k)

        if best_result is None:
            raise RuntimeError("Grid search failed: no valid result.")

        _, best_k_combo, best_l_combo, best_coef_vec, best_edf, best_terms = best_result

        self._terms = best_terms
        self._set_lams_from_combo(best_l_combo)
        self._compile(x)
        self._fit_internal(x, y, robust=robust)
        return self

    def _set_lams_from_combo(self, l_combo) -> None:
        """Apply a lam combination tuple to the current terms."""
        lam_idx = 0
        for t in self._terms:
            if isinstance(t, _TensorTerm):
                n_sl = t.d
                t.lam_per = list(l_combo[lam_idx:lam_idx + n_sl])
                lam_idx += n_sl
            else:
                t.lam = float(l_combo[lam_idx])
                lam_idx += 1

    @staticmethod
    def _assemble_penalty_from_bases(bases_list, l_combo, m, fit_intercept):
        """Build block-diagonal penalty from precomputed bases scaled by lambdas.

        Parameters
        ----------
        bases_list : list of (base_matrices, offsets, n_lam_slots) per term
            base_matrices is a list of ndarray (one per lam slot for this term)
            offsets is (start, end) into the full coef vector
            n_lam_slots is how many lambda values this term consumes
        l_combo : tuple of floats
            Lambda values for all terms, in order.
        m : int
            Total number of coefficients (including intercept if present).
        fit_intercept : bool
        """
        P = np.zeros((m, m))
        lam_idx = 0
        for base_mats, (start, end), n_slots in bases_list:
            for k in range(n_slots):
                P[start:end, start:end] += float(l_combo[lam_idx]) * base_mats[k]
                lam_idx += 1
        # Intercept: unpenalized (zero block already in P)
        return P

    @staticmethod
    def _precompute_penalty_bases(terms, fit_intercept):
        """Precompute unscaled penalty bases and offsets for each term.

        Returns list of (base_matrices, (start, end), n_lam_slots) per term,
        plus total coefficient count m.
        """
        bases_list = []
        off = 0
        m = 0
        for t in terms:
            start = off
            pb = t.penalty_bases()
            base_mats = [b[0] for b in pb]
            n_slots = sum(b[1] for b in pb)
            bases_list.append((base_mats, (start, start + t.n_coefs), n_slots))
            off += t.n_coefs
        m = off
        if fit_intercept:
            m += 1
            # Intercept: unpenalized — no bases needed, just a zero block
        return bases_list, m

    @staticmethod
    def _instantiate_terms(configs, spline_order):
        """Create term objects from parsed config dicts."""
        terms = []
        for cfg in configs:
            t = cfg['type']
            feats = cfg['features']
            kw = dict(cfg['kwargs'])
            term_order = kw.pop('spline_order', spline_order)
            if t == 's':
                terms.append(_SplineTerm(feats[0], kw.pop('n_splines', 10),
                                         kw.pop('lam', 1.0), term_order))
            elif t == 'te':
                terms.append(_TensorTerm(feats, kw.pop('n_splines', 5),
                                         kw.pop('lam', 1.0), term_order))
            elif t == 'f':
                terms.append(_FactorTerm(feats[0], kw.pop('lam', 1.0),
                                         kw.pop('coding', 'one-hot')))
            elif t == 'l':
                terms.append(_LinearTerm(feats[0], kw.pop('lam', 0.0)))
            if kw:
                raise ValueError(f"Unrecognized kwargs for {t}(): {list(kw.keys())}")
        return terms

    @staticmethod
    def _build_matrix_from_terms(terms, x, fit_intercept):
        blocks = [t.build_columns(x) for t in terms]
        if fit_intercept:
            blocks.append(np.ones((len(x), 1)))
        return np.hstack(blocks)

    @staticmethod
    def _build_penalty_from_terms(terms, fit_intercept):
        blocks = [t.build_penalty() for t in terms]
        if fit_intercept:
            blocks.append(np.zeros((1, 1)))
        if len(blocks) == 1:
            return blocks[0]
        return block_diag(*blocks)

    # -- Plotting ----------------------------------------------------------

    def plot_decomposition(self, x: ArrayLike, y: ArrayLike,
                           n_grid: int = 200, extrap_frac: float = 0.15,
                           figsize: Optional[Tuple[int, int]] = None) -> Tuple:
        """Plot overall fit and per-term partial dependence, showing extrapolation.

        The prediction grid extends ``extrap_frac`` beyond the training range.
        Extrapolated regions are shaded to distinguish them from the interpolation
        domain. Each term type gets a plot appropriate to its structure:
        s() → 1D curve,  te() → 2D heatmap (if 2 features) or curve,
        f() → bar chart,  l() → line.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Predictor matrix used for fitting.
        y : array-like, shape (n,)
            Response vector.
        n_grid : int, default 200
            Grid resolution per dimension.
        extrap_frac : float, default 0.15
            Fraction of data range to extend beyond min/max for extrapolation.
        figsize : tuple, optional
            Figure dimensions (auto-computed if None).

        Returns
        -------
        fig, axes
            Matplotlib figure and array of axes.
        """
        self._check_fitted()
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        n_features = x_arr.shape[1]
        n_terms = len(self._terms)
        p_predictors = n_features  # total input columns

        # Layout: first row = overall fit (full width), then per-term rows
        n_term_cols = min(3, n_terms)
        n_term_rows = int(np.ceil(n_terms / n_term_cols)) if n_terms > 0 else 0
        n_rows = 1 + n_term_rows
        n_cols = max(n_term_cols, 1)

        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)

        # -- Overall fit (spans all columns) -------------------------
        ax_overall = fig.add_subplot(gs[0, :])

        # Build per-feature grids and a representative "baseline" row
        feat_mins = np.min(x_arr, axis=0)
        feat_maxs = np.max(x_arr, axis=0)
        medians = np.median(x_arr, axis=0)

        feat_grids = []
        for j in range(p_predictors):
            rng = feat_maxs[j] - feat_mins[j]
            if rng == 0:
                rng = 1.0
            lo = feat_mins[j] - extrap_frac * rng
            hi = feat_maxs[j] + extrap_frac * rng
            feat_grids.append(np.linspace(lo, hi, n_grid))

        # For overall plot: sweep feature 0, hold others at median
        x_overall = np.tile(medians, (n_grid, 1))
        x_overall[:, 0] = feat_grids[0]
        y_overall = self.predict(x_overall)
        sort_idx = np.argsort(x_overall[:, 0])
        ax_overall.plot(x_overall[sort_idx, 0], y_overall[sort_idx], 'C0-', lw=2, label='GAM fit')

        ax_overall.scatter(x_arr[:, 0], y_arr, alpha=0.4, color='gray', s=10, label='Data')

        # Shade extrapolation region
        x_lo = feat_grids[0][0]
        x_hi = feat_grids[0][-1]
        ax_overall.axvspan(x_lo, feat_mins[0], alpha=0.06, color='red')
        ax_overall.axvspan(feat_maxs[0], x_hi, alpha=0.06, color='red', label='Extrapolated')
        ax_overall.set_xlabel(f'x[0]' if p_predictors > 1 else 'x')
        ax_overall.set_ylabel('y')
        ax_overall.set_title(f'GAM Fit  (GCV={self.statistics_["GCV"]:.3f}, EDF={self.statistics_["edof"]:.1f}, n={self.statistics_["n_samples"]})')
        ax_overall.legend(fontsize=7, loc='best')

        # -- Per-term plots ------------------------------------------
        for i, term in enumerate(self._terms):
            row = 1 + i // n_term_cols
            col = i % n_term_cols
            ax = fig.add_subplot(gs[row, col])

            if isinstance(term, _SplineTerm):
                x_sweep = np.tile(medians, (n_grid, 1))
                x_sweep[:, term.feature] = feat_grids[term.feature]
                pdep = self.partial_dependence(i, x_sweep)
                fv = feat_grids[term.feature]
                sort_idx_i = np.argsort(fv)
                ax.plot(fv[sort_idx_i], pdep[sort_idx_i], 'C1-', lw=2, label=f's({term.feature})')
                ax.axvspan(fv[0], feat_mins[term.feature], alpha=0.06, color='red')
                ax.axvspan(feat_maxs[term.feature], fv[-1], alpha=0.06, color='red')
                ax.axhline(0, color='gray', ls='--', lw=0.5)
                ax.set_xlabel(f'x[{term.feature}]')
                ax.set_ylabel('Contribution')
                ax.set_title(f's({term.feature})   K={term.n_splines}, lam={term.lam:.3f}')
                ax.legend(fontsize=7)

            elif isinstance(term, _TensorTerm):
                if term.d == 2:
                    ng2d = min(n_grid, 75)
                    g0 = np.linspace(feat_mins[term.features[0]] - extrap_frac * (feat_maxs[term.features[0]] - feat_mins[term.features[0]]),
                                     feat_maxs[term.features[0]] + extrap_frac * (feat_maxs[term.features[0]] - feat_mins[term.features[0]]),
                                     ng2d)
                    g1 = np.linspace(feat_mins[term.features[1]] - extrap_frac * (feat_maxs[term.features[1]] - feat_mins[term.features[1]]),
                                     feat_maxs[term.features[1]] + extrap_frac * (feat_maxs[term.features[1]] - feat_mins[term.features[1]]),
                                     ng2d)
                    f0, f1 = np.meshgrid(g0, g1, indexing='ij')
                    x_2d = np.tile(medians, (ng2d * ng2d, 1))
                    x_2d[:, term.features[0]] = f0.ravel()
                    x_2d[:, term.features[1]] = f1.ravel()
                    pdep = self.partial_dependence(i, x_2d)
                    pdep_2d = pdep.reshape(ng2d, ng2d)
                    im = ax.pcolormesh(f0, f1, pdep_2d, shading='auto', cmap='RdBu_r')
                    plt.colorbar(im, ax=ax, label='Contribution')
                    ax.set_xlabel(f'x[{term.features[0]}]')
                    ax.set_ylabel(f'x[{term.features[1]}]')
                else:
                    x_sweep = np.tile(medians, (n_grid, 1))
                    x_sweep[:, term.features[0]] = feat_grids[term.features[0]]
                    pdep = self.partial_dependence(i, x_sweep)
                    fv = feat_grids[term.features[0]]
                    ax.plot(fv, pdep, 'C2-', lw=1.5)
                    ax.set_xlabel(f'x[{term.features[0]}] (projection)')
                feats_str = ','.join(str(f) for f in term.features)
                ks_str = ','.join(str(k) for k in term.n_splines_per)
                ax.set_title(f'te({feats_str})   K=({ks_str})   {term.n_coefs} coefs')

            elif isinstance(term, _FactorTerm):
                col_vals = np.asarray(x_arr[:, term.feature]).ravel()
                unique_levels = np.unique(col_vals)
                level_contribs = []
                for lvl in unique_levels:
                    x_lvl = medians.copy().reshape(1, -1)
                    x_lvl[0, term.feature] = lvl
                    contrib = self.partial_dependence(i, x_lvl)
                    level_contribs.append(contrib[0])
                colors = ['C2' if v >= 0 else 'C3' for v in level_contribs]
                ax.bar(range(len(level_contribs)), level_contribs, color=colors, edgecolor='k', lw=0.5)
                ax.axhline(0, color='gray', ls='-', lw=0.5)
                ax.set_xticks(range(len(level_contribs)))
                ax.set_xticklabels(unique_levels, fontsize=8)
                ax.set_ylabel('Contribution')
                ax.set_title(f'f({term.feature})   lam={term.lam:.3f}')

            elif isinstance(term, _LinearTerm):
                x_sweep = np.tile(medians, (n_grid, 1))
                x_sweep[:, term.feature] = feat_grids[term.feature]
                pdep = self.partial_dependence(i, x_sweep)
                fv = feat_grids[term.feature]
                beta = self.coef_[self._coef_slices[i][0]]
                ax.plot(fv, pdep, 'C4-', lw=2)
                ax.axvspan(fv[0], feat_mins[term.feature], alpha=0.06, color='red')
                ax.axvspan(feat_maxs[term.feature], fv[-1], alpha=0.06, color='red')
                ax.axhline(0, color='gray', ls='--', lw=0.5)
                ax.set_xlabel(f'x[{term.feature}]')
                ax.set_ylabel('Contribution')
                ax.set_title(f'l({term.feature})   beta={beta:.4f}')

        fig.suptitle(f'GAMCore Decomposition — {self.formula}', fontsize=12, fontweight='bold')
        return fig, np.array(fig.axes)


# ==============================================================================
# Comprehensive Demo: Concrete Compressive Strength Prediction
# ==============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  GAMCore — Concrete Compressive Strength GAM")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Problem: Predict 28-day concrete strength from mix design + curing
    #
    # Features:
    #   x[0] = Age (days)                  [1 .. 90]   — nonlinear
    #   x[1] = Water / Cement ratio        [0.30 .. 0.60]
    #   x[2] = Cement type                 [Ordinary=0, Rapid=1, Sulfate=2]
    #   x[3] = Curing temperature (deg C)  [15 .. 35]  — linear correction
    #
    # True generative model (Abrams' law + maturity + interaction):
    #   strength = 55*(1-exp(-0.09*age))           ← nonlinear curing (s)
    #            - 35*(wc - 0.30)                   ← w/c effect
    #            + cement_bias[type]                ← factor offset
    #            + 0.08*(temp - 25)                 ← linear temp correction
    #            + 0.12*age*(0.60 - wc)             ← age:w/c interaction (te)
    #            + noise ~ N(0, 2.5)
    # ------------------------------------------------------------------

    np.random.seed(42)
    n = 200

    # Generate features
    age = np.random.uniform(1, 90, n)                    # days
    wc = np.random.uniform(0.30, 0.60, n)                # w/c ratio
    cement_labels = np.random.choice(
        ['Ordinary', 'Rapid-Hardening', 'Sulfate-Resistant'], n,
        p=[0.5, 0.3, 0.2]
    )
    cement_map = {'Ordinary': 0, 'Rapid-Hardening': 1, 'Sulfate-Resistant': 2}
    cement = np.array([cement_map[c] for c in cement_labels], dtype=float)
    temp = np.random.uniform(15, 35, n)                  # deg C

    # True strength (before noise)
    cement_bias = {'Ordinary': 0.0, 'Rapid-Hardening': 8.0, 'Sulfate-Resistant': 5.0}
    strength_true = (
        55.0 * (1 - np.exp(-0.09 * age))                 # curing curve
        - 35.0 * (wc - 0.30)                              # w/c effect
        + np.array([cement_bias[c] for c in cement_labels])  # cement type
        + 0.08 * (temp - 25)                              # temperature
        + 0.12 * age * (0.60 - wc)                        # interaction: high w/c ages slower
    )
    strength = strength_true + np.random.normal(0, 2.5, n)

    X = np.column_stack([age, wc, cement, temp])

    # Inject a few measurement outliers (bad cylinder breaks)
    outlier_idx = [10, 50, 120, 180]
    strength[outlier_idx] += np.random.choice([-20, 25], len(outlier_idx))

    print(f"\n  n = {n} observations, 4 features")
    print(f"  y range: [{strength.min():.1f}, {strength.max():.1f}] MPa")

    # ------------------------------------------------------------------
    # EXPLORATORY DATA ANALYSIS — Justify each term type visually
    # ------------------------------------------------------------------
    print("\n  -- Exploratory data plots (justifying formula choices) --")

    fig_eda, axs = plt.subplots(2, 3, figsize=(16, 10))
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axs

    # ---- 1. Strength vs Age (scatter) ----
    ax1.scatter(age, strength, c=wc, cmap='viridis', alpha=0.6, s=14, edgecolors='k', linewidth=0.3)
    # Overlay true mean curve
    age_smooth = np.linspace(1, 90, 300)
    ax1.plot(age_smooth, 55 * (1 - np.exp(-0.09 * age_smooth)) - 35 * (0.45 - 0.30) + 0.12 * age_smooth * 0.15, 'r--', lw=2, label='True at w/c=0.45')
    cbar = plt.colorbar(ax1.collections[0], ax=ax1, label='w/c ratio')
    ax1.set_xlabel('Age (days)')
    ax1.set_ylabel('Compressive Strength (MPa)')
    ax1.set_title('Strength vs Age (nonlinear curing)')
    ax1.legend(fontsize=7)
    # Annotation
    ax1.annotate('s(age)\nNonlinear:\nrapid early gain,\nplateau after ~28d',
                 xy=(45, 30), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    # ---- 2. Interaction: Strength vs Age, split by w/c bins ----
    wc_bins = ['Low w/c\n(0.30-0.40)', 'Mid w/c\n(0.40-0.50)', 'High w/c\n(0.50-0.60)']
    wc_colors = ['#2166ac', '#f4a582', '#b2182b']
    for bi, (lo, hi) in enumerate([(0.30, 0.40), (0.40, 0.50), (0.50, 0.60)]):
        mask = (wc >= lo) & (wc < hi)
        ax2.scatter(age[mask], strength[mask], c=wc_colors[bi], alpha=0.5, s=12, label=wc_bins[bi])
        # Loess-like rolling mean
        if mask.sum() > 5:
            idx_sort = np.argsort(age[mask])
            age_s = age[mask][idx_sort]
            str_s = strength[mask][idx_sort]
            # Simple rolling mean
            window = max(5, mask.sum() // 6)
            roll = np.convolve(str_s, np.ones(window) / window, mode='same')
            ax2.plot(age_s, roll, '-', color=wc_colors[bi], lw=2.5)
    ax2.set_xlabel('Age (days)')
    ax2.set_ylabel('Compressive Strength (MPa)')
    ax2.set_title('Strength vs Age by w/c ratio')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.annotate('te(age, w/c)\nInteraction:\nhigh w/c concretes\ngain strength slower',
                 xy=(50, 15), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    # ---- 3. Factor: Strength by Cement Type (boxplot + strip) ----
    cement_cats = ['Ordinary', 'Rapid-Hard.', 'Sulfate-Res.']
    cement_vals = [strength[cement_labels == c] for c in cement_cats]
    bp = ax3.boxplot(cement_vals, tick_labels=cement_cats, patch_artist=True,
                      widths=0.5, showfliers=False)
    for patch, color in zip(bp['boxes'], ['#66c2a5', '#fc8d62', '#8da0cb']):
        patch.set_facecolor(color)
    # Strip plot
    for i, vals in enumerate(cement_vals):
        jitter = np.random.normal(0, 0.05, len(vals))
        ax3.scatter(np.full_like(vals, i + 1) + jitter, vals, alpha=0.4, s=10, color='black')
    ax3.set_ylabel('Compressive Strength (MPa)')
    ax3.set_title('Strength by Cement Type')
    ax3.annotate('f(cement)\nCategorical:\nthree discrete\ngroups, no\nnatural order',
                 xy=(2.5, 30), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    # ---- 4. Residual vs Temperature (after removing main effects approximately) ----
    # Remove rough age + w/c + cement effects
    residual_temp = strength - 55 * (1 - np.exp(-0.09 * age)) + 35 * (wc - 0.30) - \
                    np.array([cement_bias[c] for c in cement_labels])
    ax4.scatter(temp, residual_temp, c='steelblue', alpha=0.5, s=14, edgecolors='k', linewidth=0.3)
    # Fit a linear trend line
    coeffs = np.polyfit(temp, residual_temp, 1)
    ax4.plot(np.sort(temp), np.polyval(coeffs, np.sort(temp)), 'r-', lw=2, label=f'slope={coeffs[0]:.3f}')
    ax4.set_xlabel('Curing Temperature (deg C)')
    ax4.set_ylabel('Approx. Residual (MPa)')
    ax4.set_title('Residual vs Temperature')
    ax4.legend(fontsize=7)
    ax4.annotate('l(temp)\nLinear:\nweak monotonic\ntrend, 1 df\nis sufficient',
                 xy=(28, 3), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    # ---- 5. Water/Cement ratio main effect ----
    # Show that w/c has a roughly linear decreasing relationship with strength
    wc_effect_mask = (age > 25) & (age < 35)  # at ~28 days
    if wc_effect_mask.sum() > 10:
        ax5.scatter(wc[wc_effect_mask], strength[wc_effect_mask], c='darkgreen', alpha=0.5, s=14, edgecolors='k', linewidth=0.3)
        coeffs2 = np.polyfit(wc[wc_effect_mask], strength[wc_effect_mask], 1)
        wc_sorted = np.sort(wc[wc_effect_mask])
        ax5.plot(wc_sorted, np.polyval(coeffs2, wc_sorted), 'r-', lw=2)
    else:
        ax5.scatter(wc, strength, c='darkgreen', alpha=0.5, s=14, edgecolors='k', linewidth=0.3)
    ax5.set_xlabel('Water/Cement Ratio')
    ax5.set_ylabel('Strength at ~28d (MPa)')
    ax5.set_title('W/C ratio effect (age ~28d)')
    ax5.annotate('Main effect captured\nby intercept +\nte(age,w/c) interaction',
                 xy=(0.45, 20), fontsize=8, ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    # ---- 6. Summary: What the GAM will model ----
    ax6.axis('off')
    summary_text = (
        "GAM Additive Decomposition\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "y = s(age)        ← nonlinear curing\n"
        "  + te(age, w/c)  ← interaction\n"
        "  + f(cement)     ← categorical offset\n"
        "  + l(temp)       ← linear correction\n"
        "  + intercept\n\n"
        "Why GAM instead of linear model?\n"
        "• s() captures the S-shaped curing\n"
        "  curve without manual feature engineering\n"
        "• te() models interaction without\n"
        "  pre-specifying its functional form\n"
        "• f() automatically handles categorical\n"
        "  variables via one-hot encoding\n"
        "• λ penalty prevents overfitting\n"
        "  (chosen automatically via GCV)"
    )
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    fig_eda.suptitle('Exploratory Data Analysis — Justifying Formula Terms', fontsize=14, fontweight='bold')
    fig_eda.tight_layout()

    # ------------------------------------------------------------------
    # Model selection: which term type for which feature?
    # ------------------------------------------------------------------
    #
    #   age (x0)  →  s(0)    Spline — nonlinear curing curve, s-shaped.
    #                          A linear or polynomial term would miss the
    #                          plateau after 28 days and overfit the ramp.
    #
    #   w/c (x1)  →  *(in te)   Not a standalone term here; its effect
    #                          enters through the interaction. Adding a
    #                          separate s(1) would double-count the main
    #                          effect unless we centre.
    #
    #   cement (x2) → f(2)   Factor — three discrete categories with no
    #                          natural ordering. One-hot encoding lets each
    #                          type have its own additive offset.
    #
    #   temp (x3)  →  l(3)    Linear — small, monotonic correction.
    #                          A spline would waste degrees of freedom
    #                          modelling noise; one coefficient is enough.
    #
    #   age:w/c interaction → te(0,1)  Tensor-product — the effect of
    #                          age on strength depends on w/c ratio.
    #                          Low w/c concrete cures faster.
    # ------------------------------------------------------------------

    formula = "s(0, n_splines=10) + te(0, 1, n_splines=6) + f(2) + l(3)"

    print(f"\n  Formula: {formula}")
    print("    s(0)     = age (nonlinear curing curve)")
    print("    te(0,1)  = age x w/c interaction (curing rate modifier)")
    print("    f(2)     = cement type (categorical baseline)")
    print("    l(3)     = curing temperature (linear correction)")

    # -- Standard fit --------------------------------------------------
    print("\n  -- Standard fit --")
    model_std = GAMCore(formula)
    model_std.fit(X, strength, robust=False)
    s_std = model_std.statistics_
    print(f"    GCV = {s_std['GCV']:.3f}")
    print(f"    EDF = {s_std['edof']:.2f}  (effective degrees of freedom)")
    print(f"    RSS = {s_std['rss']:.1f}")
    print(f"    sigma_hat = {s_std['scale']:.2f} MPa")

    # -- Robust fit (Huber IRLS — downweights the injected outliers) --
    print("\n  -- Robust fit (Huber IRLS) --")
    model_rob = GAMCore(formula)
    model_rob.fit(X, strength, robust=True)
    stats_r = model_rob.statistics_
    print(f"    GCV = {stats_r['GCV']:.3f}")
    print(f"    RSS = {stats_r['rss']:.1f}")

    # -- Per-term contributions ---------------------------------------
    print("\n  -- Per-term additive contributions --")
    X_eval = X.copy()
    for i in range(model_std.n_terms):
        term = model_std._terms[i]
        pdep = model_std.partial_dependence(i, X_eval)
        if isinstance(term, _SplineTerm):
            print(f"    s({term.feature})  K={term.n_splines:2d}  lam={term.lam:.3f}  "
                  f"contrib range [{pdep.min():.1f}, {pdep.max():.1f}] MPa")
        elif isinstance(term, _TensorTerm):
            print(f"    te({term.features})  K=({','.join(str(k) for k in term.n_splines_per)})  "
                  f"coefs={term.n_coefs}  contrib range [{pdep.min():.1f}, {pdep.max():.1f}] MPa")
        elif isinstance(term, _FactorTerm):
            print(f"    f({term.feature})  levels={getattr(term,'_levels',[])}  lam={term.lam:.3f}  "
                  f"contrib range [{pdep.min():.1f}, {pdep.max():.1f}] MPa")
        elif isinstance(term, _LinearTerm):
            beta = model_std.coef_[model_std._coef_slices[i][0]]
            print(f"    l({term.feature})  beta={beta:.4f} MPa/degC  "
                  f"contrib range [{pdep.min():.1f}, {pdep.max():.1f}] MPa")

    # -- Prediction at key ages --------------------------------------
    print("\n  -- Strength predictions at key ages (Ordinary cement, w/c=0.45, 20 degC) --")
    for test_age in [1, 3, 7, 14, 28, 56, 90]:
        x_row = np.array([[test_age, 0.45, 0, 20.0]])
        pred = model_std.predict(x_row)[0]
        ci = model_std.confidence_intervals(x_row, width=0.95)[0]
        print(f"    {test_age:3d} days:  {pred:5.1f} MPa  (95% CI: [{ci[0]:5.1f}, {ci[1]:5.1f}])")

    # -- Grid search --------------------------------------------------
    print("\n  -- Quick grid search on age term only --")
    model_gs = GAMCore(formula)
    # Use tiny candidate sets to keep runtime reasonable
    model_gs.gridsearch(X, strength,
                        lam_grids=[np.logspace(-1, 1, 2)],
                        n_splines_grids=[np.array([10])],
                        fast=True)
    print(f"    Best params found via GCV")
    print(f"    GCV = {model_gs.statistics_['GCV']:.3f}")

    print("\n" + "=" * 72)
    print("  All tests passed — generating diagnostic plots ...")
    print("=" * 72)

    # -- Observed vs Fitted plot (model fit diagnostics) ---------------
    y_hat_std = model_std.predict(X)
    y_hat_rob = model_rob.predict(X)
    residuals_std = strength - y_hat_std
    residuals_rob = strength - y_hat_rob
    r2_std = 1 - np.var(residuals_std) / np.var(strength)
    r2_rob = 1 - np.var(residuals_rob) / np.var(strength)

    fig_fit, (axf1, axf2, axf3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: y vs y_fit (standard)
    axf1.scatter(strength, y_hat_std, c='steelblue', alpha=0.5, s=14, edgecolors='k', linewidth=0.3)
    lims = [min(strength.min(), y_hat_std.min()) - 2, max(strength.max(), y_hat_std.max()) + 2]
    axf1.plot(lims, lims, 'r--', lw=1.5, label='1:1 line')
    axf1.set_xlim(lims); axf1.set_ylim(lims)
    axf1.set_xlabel('Observed Strength (MPa)')
    axf1.set_ylabel('Fitted Strength (MPa)')
    axf1.set_title(f'Standard GAM: Observed vs Fitted\nR2 = {r2_std:.3f}  |  GCV = {s_std["GCV"]:.3f}')
    axf1.legend(fontsize=8)
    axf1.set_aspect('equal')

    # Panel 2: y vs y_fit (robust)
    axf2.scatter(strength, y_hat_rob, c='darkorange', alpha=0.5, s=14, edgecolors='k', linewidth=0.3)
    axf2.plot(lims, lims, 'r--', lw=1.5, label='1:1 line')
    axf2.set_xlim(lims); axf2.set_ylim(lims)
    axf2.set_xlabel('Observed Strength (MPa)')
    axf2.set_ylabel('Fitted Strength (MPa)')
    axf2.set_title(f'Robust GAM: Observed vs Fitted\nR2 = {r2_rob:.3f}  |  GCV = {stats_r["GCV"]:.3f}')
    axf2.legend(fontsize=8)
    axf2.set_aspect('equal')

    # Panel 3: Residuals vs Fitted (standard)
    axf3.scatter(y_hat_std, residuals_std, c='steelblue', alpha=0.5, s=14, edgecolors='k', linewidth=0.3)
    axf3.axhline(0, color='red', ls='--', lw=1.5)
    axf3.set_xlabel('Fitted Strength (MPa)')
    axf3.set_ylabel('Residuals (MPa)')
    axf3.set_title('Residuals vs Fitted (Standard)')
    # Annotate outliers
    outlier_mask = np.abs(residuals_std) > 3 * np.std(residuals_std)
    if outlier_mask.any():
        axf3.scatter(y_hat_std[outlier_mask], residuals_std[outlier_mask],
                     c='red', s=30, edgecolors='k', linewidth=1, label='>3σ outlier')
        axf3.legend(fontsize=8)

    fig_fit.suptitle('Model Fit Diagnostics — y vs y_fit', fontsize=13, fontweight='bold')
    fig_fit.tight_layout()

    # -- Decomposition plot -------------------------------------------
    model_std.plot_decomposition(X, strength, extrap_frac=0.15)
    plt.show()
