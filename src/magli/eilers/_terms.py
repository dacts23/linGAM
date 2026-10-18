"""GAM term classes: spline, tensor, factor, and linear."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .._base import _khatri_rao, _kron_sum
from ._bspline import b_spline_basis
from ._penalty import (
    circular_second_diff_penalty,
    ridge_penalty,
    second_diff_penalty,
    shape_constraint_penalty,
)

_CONSTRAINTS = frozenset({
    None, 'mono_inc', 'mono_dec', 'convex', 'concave', 'periodic',
})
_SHAPE_CONSTRAINTS = frozenset({'mono_inc', 'mono_dec', 'convex', 'concave'})


class _Term:
    """Base class for all GAM terms."""

    n_coefs: int = 0
    sp: Optional[List[float]] = None

    def compile(self, x: np.ndarray) -> None:
        """One-time setup using training data."""

    def build_penalties(self) -> List[np.ndarray]:
        return []


class _SplineTerm(_Term):
    """Univariate penalised B-spline smooth term."""

    __slots__ = (
        'feature', 'n_splines', 'lam', 'spline_order',
        '_edge_knots', 'constraint', 'n_coefs',
    )

    def __init__(
        self,
        feature: int,
        n_splines: int = 10,
        lam: float = 1.0,
        spline_order: int = 3,
        constraint: Optional[str] = None,
    ) -> None:
        if constraint not in _CONSTRAINTS:
            raise ValueError(
                f"Unknown constraint '{constraint}'. "
                f"Choose from: {sorted(c for c in _CONSTRAINTS if c is not None)}."
            )
        min_k = (2 * spline_order) if constraint == 'periodic' else (spline_order + 1)
        if n_splines < min_k:
            raise ValueError(
                f"n_splines ({n_splines}) must be at least {min_k} "
                f"for spline_order ({spline_order})"
                + (" with periodic constraint." if constraint == 'periodic' else ".")
            )
        self.feature = feature
        self.n_splines = n_splines
        self.lam = lam
        self.spline_order = spline_order
        self.constraint = constraint
        self.n_coefs = (
            n_splines - spline_order
            if constraint == 'periodic'
            else n_splines
        )
        self._edge_knots: Optional[np.ndarray] = None

    def compile(self, x: np.ndarray) -> None:
        col = np.asarray(x[:, self.feature]).ravel()
        self._edge_knots = np.array([col.min(), col.max()])

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        return b_spline_basis(
            x[:, self.feature],
            self.n_splines,
            self.spline_order,
            self._edge_knots,  # type: ignore[arg-type]
            periodic=self.constraint == 'periodic',
        )

    def build_penalty(self, lam: Optional[float] = None) -> np.ndarray:
        l = lam if lam is not None else self.lam
        K_eff = self.n_coefs
        if self.constraint == 'periodic':
            return circular_second_diff_penalty(K_eff, l)
        return second_diff_penalty(K_eff, l)

    @property
    def has_shape_constraint(self) -> bool:
        return self.constraint in _SHAPE_CONSTRAINTS

    def build_constraint_penalty(self, coef_sub: np.ndarray) -> np.ndarray:
        if self.constraint in _SHAPE_CONSTRAINTS:
            return shape_constraint_penalty(coef_sub, self.constraint)
        return np.zeros((self.n_coefs, self.n_coefs))

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        return [(self.build_penalty(1.0), 1)]


class _TensorTerm(_Term):
    """Tensor-product B-spline interaction term (2+D)."""

    __slots__ = (
        'features', 'd', 'n_splines_per', 'lam_per', 'lam',
        'spline_order', '_edge_knots', 'n_coefs',
    )

    def __init__(
        self,
        features: List[int],
        n_splines: Union[int, List[int]] = 5,
        lam: Union[float, List[float]] = 1.0,
        spline_order: int = 3,
    ) -> None:
        self.features = list(features)
        self.d = len(self.features)
        if self.d < 2:
            raise ValueError("Tensor term requires at least 2 features.")

        if isinstance(n_splines, int):
            self.n_splines_per = [n_splines] * self.d
        elif len(n_splines) != self.d:
            raise ValueError(
                f"n_splines length ({len(n_splines)}) must match "
                f"number of features ({self.d})."
            )
        else:
            self.n_splines_per = list(n_splines)

        if isinstance(lam, (int, float)):
            self.lam_per = [float(lam)] * self.d
        elif len(lam) != self.d:
            raise ValueError(
                f"lam length ({len(lam)}) must match "
                f"number of features ({self.d})."
            )
        else:
            self.lam_per = [float(v) for v in lam]

        self.lam = self.lam_per[0]
        self.spline_order = spline_order
        self.n_coefs = int(np.prod(self.n_splines_per))
        self._edge_knots: List[np.ndarray] = []

    def compile(self, x: np.ndarray) -> None:
        self._edge_knots = [
            np.array([np.min(x[:, f]), np.max(x[:, f])]) for f in self.features
        ]

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        return _khatri_rao([
            b_spline_basis(
                x[:, feat],
                self.n_splines_per[i],
                self.spline_order,
                self._edge_knots[i],
            )
            for i, feat in enumerate(self.features)
        ])

    def build_penalty(self, lam: Optional[List[float]] = None) -> np.ndarray:
        lams = lam if lam is not None else self.lam_per
        if isinstance(lams, (int, float)):
            lams = [float(lams)] * self.d
        return _kron_sum([
            second_diff_penalty(self.n_splines_per[i], lams[i])
            for i in range(self.d)
        ])

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        sizes = list(self.n_splines_per)
        bases: List[Tuple[np.ndarray, int]] = []
        for i, sz in enumerate(sizes):
            S_i = second_diff_penalty(sz, 1.0)
            I_pre = np.eye(int(np.prod(sizes[:i])))
            I_post = np.eye(int(np.prod(sizes[i + 1:])))
            bases.append((np.kron(np.kron(I_pre, S_i), I_post), 1))
        return bases


class _FactorTerm(_Term):
    """Categorical / factor term with one-hot or dummy encoding."""

    __slots__ = ('feature', 'lam', 'coding', '_levels', 'n_coefs')

    def __init__(
        self,
        feature: int,
        lam: float = 1.0,
        coding: str = 'one-hot',
    ) -> None:
        self.feature = int(feature)
        self.lam = lam
        self.coding = coding
        self._levels: Optional[np.ndarray] = None
        self.n_coefs = 0

    def compile(self, x: np.ndarray) -> None:
        self._levels = np.unique(np.asarray(x[:, self.feature]).ravel())
        self.n_coefs = (
            len(self._levels)
            if self.coding == 'one-hot'
            else len(self._levels) - 1
        )
        if self.n_coefs <= 0:
            raise ValueError(f"Factor term feature {self.feature} has no levels.")

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        if self._levels is None:
            self.compile(x)
        col = np.asarray(x[:, self.feature]).ravel()
        B = (col[:, None] == self._levels[None, :]).astype(float)
        return B[:, 1:] if self.coding == 'dummy' else B

    def build_penalty(self, lam: Optional[float] = None) -> np.ndarray:
        return ridge_penalty(self.n_coefs, lam if lam is not None else self.lam)

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        return [(np.eye(self.n_coefs), 1)]


class _LinearTerm(_Term):
    """Linear (unpenalised by default) term."""

    __slots__ = ('feature', 'lam', 'n_coefs')

    def __init__(self, feature: int, lam: float = 0.0) -> None:
        self.feature = int(feature)
        self.lam = lam
        self.n_coefs = 1

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x[:, self.feature], dtype=float).ravel().reshape(-1, 1)

    def build_penalty(self, lam: Optional[float] = None) -> np.ndarray:
        return np.array([[lam if lam is not None else self.lam]], dtype=float)

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        return [(np.array([[1.0]]), 1)]


def instantiate_terms(
    configs: List[Dict],
    spline_order: int,
) -> List[_Term]:
    """Create term objects from parsed config dicts."""
    terms: List[_Term] = []
    for cfg in configs:
        t = cfg['type']
        feats = cfg['features']
        kw = dict(cfg['kwargs'])
        term_order = kw.pop('spline_order', spline_order)
        k_val = kw.pop('n_splines', kw.pop('k', None))
        lam_val = kw.pop('lam', kw.pop('sp', None))

        if t == 's':
            terms.append(_SplineTerm(
                feats[0],
                k_val if k_val is not None else 10,
                lam_val if lam_val is not None else 1.0,
                term_order,
                constraint=kw.pop('constraint', None),
            ))
        elif t == 'te':
            terms.append(_TensorTerm(
                feats,
                k_val if k_val is not None else 5,
                lam_val if lam_val is not None else 1.0,
                term_order,
            ))
        elif t == 'f':
            terms.append(_FactorTerm(
                feats[0],
                lam_val if lam_val is not None else 1.0,
                kw.pop('coding', 'one-hot'),
            ))
        elif t == 'l':
            terms.append(_LinearTerm(
                feats[0],
                lam_val if lam_val is not None else 0.0,
            ))
        else:
            raise ValueError(f"Unknown term type: {t}")

        if kw:
            raise ValueError(
                f"Unrecognised kwargs for {t}(): {list(kw.keys())}"
            )
    return terms
