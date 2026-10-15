"""Term classes for mgcv GAM backend: spline, tensor product, factor, and linear terms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .._base import _khatri_rao
from ..pygam._terms import _FactorTerm, _LinearTerm, _Term as _MgcvTerm
from ._smooths import (
    absorb_sum_to_zero_constraint,
    build_smooth_basis,
    predict_smooth_basis,
    rescale_penalty,
)


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
        S_scaled, _ = rescale_penalty(X_raw, S_list)
        X_abs, self._penalties, self._Z = absorb_sum_to_zero_constraint(
            X_raw, S_scaled
        )
        self.n_coefs = X_abs.shape[1]

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        col = np.asarray(x[:, self.feature], dtype=float).ravel()
        if self._Z is None:
            self.compile(x)
        return predict_smooth_basis(col, self._meta) @ self._Z

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
        self.features = [int(f) for f in features]
        self.d = len(self.features)
        if self.d < 2:
            raise ValueError(
                "Tensor product term requires at least 2 features."
            )

        self.k_per = (
            [k] * self.d if isinstance(k, int) else [int(v) for v in k]
        )
        self.bs_per = (
            [bs] * self.d if isinstance(bs, str) else [str(v) for v in bs]
        )
        self.sp = (
            None
            if sp is None
            else [float(sp)] * self.d
            if isinstance(sp, (int, float))
            else [float(v) for v in sp]
        )
        self._Z: Optional[np.ndarray] = None
        self._penalties: List[np.ndarray] = []
        self._marginal_meta: List[Dict] = []
        self.n_coefs = 0

    def compile(self, x: np.ndarray) -> None:
        marg_X, marg_S, self._marginal_meta = [], [], []
        for i, feat in enumerate(self.features):
            col = np.asarray(x[:, feat], dtype=float).ravel()
            Xi, Si, meta_i = build_smooth_basis(
                col, bs=self.bs_per[i], k=self.k_per[i]
            )
            ev_max = np.linalg.eigvalsh(Si[0])[-1]
            if ev_max > 0:
                Si[0] = Si[0] / ev_max
            marg_X.append(Xi)
            marg_S.append(Si[0])
            self._marginal_meta.append(meta_i)

        X_te = _khatri_rao(marg_X)
        sizes = [m.shape[1] for m in marg_X]
        TS_list = []
        for i in range(self.d):
            I_pre = np.eye(int(np.prod(sizes[:i])))
            I_post = np.eye(int(np.prod(sizes[i + 1 :])))
            P_i = np.kron(np.kron(I_pre, marg_S[i]), I_post)
            TS_list.append(0.5 * (P_i + P_i.T))

        S_scaled, _ = rescale_penalty(X_te, TS_list)
        X_abs, self._penalties, self._Z = absorb_sum_to_zero_constraint(
            X_te, S_scaled
        )
        self.n_coefs = X_abs.shape[1]

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        if self._Z is None:
            self.compile(x)
        marg_X = [
            predict_smooth_basis(
                np.asarray(x[:, feat], dtype=float).ravel(),
                self._marginal_meta[i],
            )
            for i, feat in enumerate(self.features)
        ]
        return _khatri_rao(marg_X) @ self._Z

    def build_penalties(self) -> List[np.ndarray]:
        return [p.copy() for p in self._penalties]


class _MgcvFactorTerm(_FactorTerm):
    """Categorical factor term f(feature)."""

    def __init__(self, feature: int, coding: str = "dummy") -> None:
        super().__init__(feature, lam=0.0, coding=coding)


_MgcvLinearTerm = _LinearTerm


def instantiate_mgcv_terms(configs: List[Dict[str, Any]]) -> List[_MgcvTerm]:
    """Instantiate mgcv term objects from parsed formula config dicts."""
    terms: List[_MgcvTerm] = []
    for cfg in configs:
        kind = cfg["type"]
        feats = cfg["features"]
        kw = dict(cfg["kwargs"])

        k_val = kw.pop("k", kw.pop("n_splines", 10))
        sp_val = kw.pop("sp", kw.pop("lam", None))
        bs_val = kw.pop("bs", "tp")

        if kind == "s":
            terms.append(
                _MgcvSplineTerm(
                    feature=feats[0],
                    k=k_val,
                    bs=bs_val,
                    sp=sp_val,
                    m=kw.pop("m", None),
                    knots=kw.pop("knots", None),
                    constraint=kw.pop("constraint", None),
                )
            )
        elif kind == "te":
            te_k = k_val if isinstance(k_val, (int, list)) else 5
            terms.append(
                _MgcvTensorTerm(
                    features=feats, k=te_k, bs=bs_val, sp=sp_val
                )
            )
        elif kind == "f":
            terms.append(
                _MgcvFactorTerm(
                    feature=feats[0], coding=kw.pop("coding", "dummy")
                )
            )
        elif kind == "l":
            terms.append(_MgcvLinearTerm(feature=feats[0]))
        else:
            raise ValueError(f"Unknown term type '{kind}' in formula.")
    return terms
