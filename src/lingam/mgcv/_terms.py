"""Term classes for mgcv GAM backend: spline, tensor product, factor, and linear terms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ._smooths import (
    absorb_sum_to_zero_constraint,
    build_smooth_basis,
    predict_smooth_basis,
    rescale_penalty,
)


class _MgcvTerm:
    """Base class for all mgcv GAM terms."""

    def __init__(self) -> None:
        self.n_coefs: int = 0
        self.sp: Optional[List[float]] = None

    def compile(self, x: np.ndarray) -> None:
        """Initialize domain, knots, or factor levels from training data."""
        raise NotImplementedError

    def build_columns(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Build design matrix columns for this term."""
        raise NotImplementedError

    def build_penalties(self) -> List[np.ndarray]:
        """Return penalty matrices for this term."""
        raise NotImplementedError


class _MgcvSplineTerm(_MgcvTerm):
    """Univariate smooth term s(x, bs='tp', k=10)."""

    def __init__(
        self,
        feature: int,
        k: int = 10,
        bs: str = "tp",
        sp: Optional[float] = None,
        m: Optional[Union[int, Tuple[int, int]]] = None,
        knots: Optional[np.ndarray] = None,
        constraint: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.feature = int(feature)
        self.k = int(k)
        self.bs = bs.lower()
        if constraint == "periodic" and self.bs == "tp":
            self.bs = "cc"
        self.sp = [float(sp)] if sp is not None else None
        self.m = m
        self.knots = knots
        self._Z: Optional[np.ndarray] = None
        self._penalties: List[np.ndarray] = []
        self._meta: Dict[str, Any] = {}
        self.n_coefs = 0

    def compile(self, x: np.ndarray) -> None:
        col = np.asarray(x[:, self.feature], dtype=float).ravel()
        X_raw, S_list, self._meta = build_smooth_basis(
            col, bs=self.bs, k=self.k, knots=self.knots, m=self.m
        )
        S_scaled, scales = rescale_penalty(X_raw, S_list)
        self._scales = scales
        X_abs, S_abs, self._Z = absorb_sum_to_zero_constraint(X_raw, S_scaled)
        self._penalties = S_abs
        self.n_coefs = X_abs.shape[1]

    def build_columns(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        col = np.asarray(x[:, self.feature], dtype=float).ravel()
        if self._Z is None:
            self.compile(x)
        X_raw = predict_smooth_basis(col, self._meta)
        return X_raw @ self._Z

    def build_penalties(self) -> List[np.ndarray]:
        return [p.copy() for p in self._penalties]


class _MgcvTensorTerm(_MgcvTerm):
    """Tensor product smooth te(x1, x2, bs='cr', k=5)."""

    def __init__(
        self,
        features: List[int],
        k: Union[int, List[int]] = 5,
        bs: Union[str, List[str]] = "cr",
        sp: Optional[Union[float, List[float]]] = None,
    ) -> None:
        super().__init__()
        self.features = [int(f) for f in features]
        self.d = len(self.features)
        if self.d < 2:
            raise ValueError("Tensor product term requires at least 2 features.")

        if isinstance(k, int):
            self.k_per = [k] * self.d
        else:
            self.k_per = [int(v) for v in k]

        if isinstance(bs, str):
            self.bs_per = [bs] * self.d
        else:
            self.bs_per = [str(v) for v in bs]

        if sp is None:
            self.sp = None
        elif isinstance(sp, (int, float)):
            self.sp = [float(sp)] * self.d
        else:
            self.sp = [float(v) for v in sp]

        self._Z: Optional[np.ndarray] = None
        self._penalties: List[np.ndarray] = []
        self._marginal_meta: List[Dict] = []
        self.n_coefs = 0

    def compile(self, x: np.ndarray) -> None:
        n = x.shape[0]
        marg_X = []
        marg_S = []
        self._marginal_meta = []

        for i, feat in enumerate(self.features):
            col = np.asarray(x[:, feat], dtype=float).ravel()
            Xi, Si, meta_i = build_smooth_basis(
                col, bs=self.bs_per[i], k=self.k_per[i]
            )
            # Normalize marginal penalty by its largest eigenvalue as in mgcv
            ev_max = np.linalg.eigvalsh(Si[0])[-1]
            if ev_max > 0:
                Si[0] = Si[0] / ev_max
            marg_X.append(Xi)
            marg_S.append(Si[0])
            self._marginal_meta.append(meta_i)

        # Row-wise tensor product
        X_te = marg_X[0]
        for i in range(1, self.d):
            n_curr = X_te.shape[1]
            n_next = marg_X[i].shape[1]
            out = np.empty((n, n_curr * n_next), dtype=float)
            col_idx = 0
            for c1 in range(n_curr):
                for c2 in range(n_next):
                    out[:, col_idx] = X_te[:, c1] * marg_X[i][:, c2]
                    col_idx += 1
            X_te = out

        # Tensor penalties
        TS_list = []
        sizes = [m.shape[1] for m in marg_X]
        for i in range(self.d):
            I_pre = np.eye(int(np.prod(sizes[:i])))
            I_post = np.eye(int(np.prod(sizes[i + 1:])))
            P_i = np.kron(np.kron(I_pre, marg_S[i]), I_post)
            P_i = 0.5 * (P_i + P_i.T)
            TS_list.append(P_i)

        # Scale penalties and absorb sum-to-zero constraint
        S_scaled, scales = rescale_penalty(X_te, TS_list)
        self._scales = scales
        X_abs, S_abs, self._Z = absorb_sum_to_zero_constraint(X_te, S_scaled)
        self._penalties = S_abs
        self.n_coefs = X_abs.shape[1]

    def build_columns(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        if self._Z is None:
            self.compile(x)
        n = x.shape[0]
        marg_X = []
        for i, feat in enumerate(self.features):
            col = np.asarray(x[:, feat], dtype=float).ravel()
            Xi = predict_smooth_basis(col, self._marginal_meta[i])
            marg_X.append(Xi)

        X_te = marg_X[0]
        for i in range(1, self.d):
            n_curr = X_te.shape[1]
            n_next = marg_X[i].shape[1]
            out = np.empty((n, n_curr * n_next), dtype=float)
            col_idx = 0
            for c1 in range(n_curr):
                for c2 in range(n_next):
                    out[:, col_idx] = X_te[:, c1] * marg_X[i][:, c2]
                    col_idx += 1
            X_te = out

        return X_te @ self._Z

    def build_penalties(self) -> List[np.ndarray]:
        return [p.copy() for p in self._penalties]


class _MgcvFactorTerm(_MgcvTerm):
    """Categorical factor term f(feature)."""

    def __init__(
        self,
        feature: int,
        coding: str = "dummy",
    ) -> None:
        super().__init__()
        self.feature = int(feature)
        self.coding = coding
        self._levels: Optional[np.ndarray] = None
        self.n_coefs = 0

    def compile(self, x: np.ndarray) -> None:
        col = np.asarray(x[:, self.feature]).ravel()
        self._levels = np.unique(col)
        # Dummy coding leaves out first level as reference
        self.n_coefs = len(self._levels) - 1 if self.coding == "dummy" else len(self._levels)

    def build_columns(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        col = np.asarray(x[:, self.feature]).ravel()
        if self._levels is None:
            self.compile(x)
        B = (col[:, None] == self._levels[None, :]).astype(float)
        if self.coding == "dummy":
            B = B[:, 1:]
        return B

    def build_penalties(self) -> List[np.ndarray]:
        # Unpenalized parametric factor in standard mgcv
        return []


class _MgcvLinearTerm(_MgcvTerm):
    """Parametric linear term l(feature)."""

    def __init__(self, feature: int) -> None:
        super().__init__()
        self.feature = int(feature)
        self.n_coefs = 1

    def compile(self, x: np.ndarray) -> None:
        pass

    def build_columns(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        return np.asarray(x[:, self.feature], dtype=float).ravel().reshape(-1, 1)

    def build_penalties(self) -> List[np.ndarray]:
        return []


def instantiate_mgcv_terms(configs: List[Dict[str, Any]]) -> List[_MgcvTerm]:
    """Instantiate mgcv term objects from parsed formula config dicts."""
    terms: List[_MgcvTerm] = []
    for cfg in configs:
        kind = cfg["type"]
        feats = cfg["features"]
        kw = dict(cfg["kwargs"])

        # Map pyGAM/mgcv parameter aliases
        k_val = kw.pop("k", None)
        if k_val is None:
            k_val = kw.pop("n_splines", 10)

        sp_val = kw.pop("sp", None)
        if sp_val is None:
            sp_val = kw.pop("lam", None)

        bs_val = kw.pop("bs", "tp")
        m_val = kw.pop("m", None)
        knots_val = kw.pop("knots", None)
        constraint_val = kw.pop("constraint", None)

        if kind == "s":
            terms.append(
                _MgcvSplineTerm(
                    feature=feats[0],
                    k=k_val,
                    bs=bs_val,
                    sp=sp_val,
                    m=m_val,
                    knots=knots_val,
                    constraint=constraint_val,
                )
            )
        elif kind == "te":
            te_k = k_val if isinstance(k_val, (int, list)) else 5
            terms.append(
                _MgcvTensorTerm(
                    features=feats,
                    k=te_k,
                    bs=bs_val,
                    sp=sp_val,
                )
            )
        elif kind == "f":
            coding = kw.pop("coding", "dummy")
            terms.append(_MgcvFactorTerm(feature=feats[0], coding=coding))
        elif kind == "l":
            terms.append(_MgcvLinearTerm(feature=feats[0]))
        else:
            raise ValueError(f"Unknown term type '{kind}' in formula.")

    return terms
