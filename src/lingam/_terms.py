"""GAM term classes: spline, tensor, factor, and linear."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ._bspline import b_spline_basis
from ._penalty import (
    circular_second_diff_penalty,
    first_diff_penalty,
    ridge_penalty,
    second_diff_penalty,
)

_CONSTRAINTS = frozenset({
    None, 'mono_inc', 'mono_dec', 'convex', 'concave', 'periodic',
})


class _Term:
    """Base class for all GAM terms."""

    __slots__ = ('n_coefs',)

    def __init__(self) -> None:
        self.n_coefs: int = 0

    def compile(self, x: np.ndarray) -> None:
        """One-time setup using training data."""

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        """Build the design matrix columns for this term."""
        raise NotImplementedError


class _SplineTerm(_Term):
    """Univariate penalised B-spline smooth term.

    Parameters
    ----------
    feature : int
        Column index in the predictor matrix.
    n_splines : int
        Number of basis functions.
    lam : float
        Smoothing parameter.
    spline_order : int
        B-spline order (polynomial degree + 1).
    constraint : str or None
        Shape constraint. One of ``None``, ``'mono_inc'``,
        ``'mono_dec'``, ``'convex'``, ``'concave'``, ``'periodic'``.

        * ``None`` — standard second-difference penalty (smoothness).
        * ``'mono_inc'`` — first-difference penalty (encourages
          monotonically increasing trend).
        * ``'mono_dec'`` — first-difference penalty (encourages
          monotonically decreasing trend).
        * ``'convex'`` — second-difference penalty (encourages convexity).
        * ``'concave'`` — second-difference penalty (encourages concavity).
        * ``'periodic'`` — circular second-difference penalty with
          wrapped basis (enforces f(0)=f(1)).
    """

    __slots__ = (
        'feature', 'n_splines', 'lam', 'spline_order',
        '_edge_knots', 'constraint',
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
        self.feature = feature
        self.n_splines = n_splines
        self.lam = lam
        self.spline_order = spline_order
        self.constraint = constraint
        self.n_coefs = (
            n_splines - spline_order + 1
            if constraint == 'periodic'
            else n_splines
        )
        self._edge_knots: Optional[np.ndarray] = None

    def compile(self, x: np.ndarray) -> None:
        col = np.asarray(x[:, self.feature]).ravel()
        self._edge_knots = np.array([col.min(), col.max()])

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        B = b_spline_basis(
            x[:, self.feature],
            self.n_splines,
            self.spline_order,
            self._edge_knots,  # type: ignore[arg-type]
            periodic=self.constraint == 'periodic',
        )
        return B

    def build_penalty(self, lam: Optional[float] = None) -> np.ndarray:
        l = lam if lam is not None else self.lam
        K_eff = self.n_coefs
        if self.constraint in ('mono_inc', 'mono_dec'):
            return first_diff_penalty(K_eff, l)
        if self.constraint == 'periodic':
            return circular_second_diff_penalty(K_eff, l)
        return second_diff_penalty(K_eff, l)

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        K_eff = self.n_coefs
        if self.constraint in ('mono_inc', 'mono_dec'):
            return [(first_diff_penalty(K_eff, 1.0), 1)]
        if self.constraint == 'periodic':
            return [(circular_second_diff_penalty(K_eff, 1.0), 1)]
        return [(second_diff_penalty(K_eff, 1.0), 1)]


class _TensorTerm(_Term):
    """Tensor-product B-spline interaction term (2+D)."""

    __slots__ = (
        'features', 'd', 'n_splines_per', 'lam_per', 'lam',
        'spline_order', '_edge_knots',
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
        else:
            if len(n_splines) != self.d:
                raise ValueError(
                    f"n_splines length ({len(n_splines)}) must match "
                    f"number of features ({self.d})."
                )
            self.n_splines_per = list(n_splines)

        if isinstance(lam, (int, float)):
            self.lam_per = [float(lam)] * self.d
        else:
            if len(lam) != self.d:
                raise ValueError(
                    f"lam length ({len(lam)}) must match "
                    f"number of features ({self.d})."
                )
            self.lam_per = [float(v) for v in lam]

        self.lam = self.lam_per[0]
        self.spline_order = spline_order
        self.n_coefs = int(np.prod(self.n_splines_per))
        self._edge_knots: List[np.ndarray] = []

    def compile(self, x: np.ndarray) -> None:
        self._edge_knots = []
        for feat in self.features:
            col = np.asarray(x[:, feat]).ravel()
            self._edge_knots.append(np.array([col.min(), col.max()]))

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        marginals = []
        for i, feat in enumerate(self.features):
            B_i = b_spline_basis(
                x[:, feat],
                self.n_splines_per[i],
                self.spline_order,
                self._edge_knots[i],
            )
            marginals.append(B_i)
        return _khatri_rao(marginals)

    def build_penalty(
        self,
        lam: Optional[List[float]] = None,
    ) -> np.ndarray:
        lams = lam if lam is not None else self.lam_per
        if isinstance(lams, (int, float)):
            lams = [float(lams)] * self.d

        marginal_penalties = []
        for i in range(self.d):
            P_i = second_diff_penalty(self.n_splines_per[i], lams[i])
            marginal_penalties.append(P_i)
        return _kron_sum(marginal_penalties)

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        bases: List[Tuple[np.ndarray, int]] = []
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

    __slots__ = ('feature', 'lam', 'coding', '_levels')

    def __init__(
        self,
        feature: int,
        lam: float = 1.0,
        coding: str = 'one-hot',
    ) -> None:
        self.feature = feature
        self.lam = lam
        self.coding = coding
        self._levels: Optional[np.ndarray] = None
        self.n_coefs = 0

    def compile(self, x: np.ndarray) -> None:
        col = np.asarray(x[:, self.feature]).ravel()
        self._levels = np.unique(col)
        self.n_coefs = (
            len(self._levels)
            if self.coding == 'one-hot'
            else len(self._levels) - 1
        )
        if self.n_coefs <= 0:
            raise ValueError(
                f"Factor term feature {self.feature} has no levels."
            )

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        col = np.asarray(x[:, self.feature]).ravel()
        B = (col[:, None] == self._levels[None, :]).astype(float)
        if self.coding == 'dummy':
            B = B[:, 1:]
        return B

    def build_penalty(self, lam: Optional[float] = None) -> np.ndarray:
        return ridge_penalty(
            self.n_coefs,
            lam if lam is not None else self.lam,
        )

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        return [(np.eye(self.n_coefs), 1)]


class _LinearTerm(_Term):
    """Linear (unpenalised by default) term."""

    __slots__ = ('feature', 'lam')

    def __init__(self, feature: int, lam: float = 0.0) -> None:
        self.feature = feature
        self.lam = lam
        self.n_coefs = 1

    def build_columns(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x[:, self.feature]).ravel().reshape(-1, 1)

    def build_penalty(self, lam: Optional[float] = None) -> np.ndarray:
        l = lam if lam is not None else self.lam
        return np.array([[l]], dtype=float)

    def penalty_bases(self) -> List[Tuple[np.ndarray, int]]:
        return [(np.array([[1.0]]), 1)]


def _khatri_rao(marginals: List[np.ndarray]) -> np.ndarray:
    """Column-wise Kronecker (Khatri-Rao) product of marginals."""
    n = marginals[0].shape[0]
    result = marginals[0]
    for i in range(1, len(marginals)):
        result = result[..., :, None]
        B_next = marginals[i][:, None, :]
        result = result * B_next
        result = result.reshape(n, -1)
    return result


def _kron_sum(penalties: List[np.ndarray]) -> np.ndarray:
    """Kronecker sum: sum_i I_0 kron ... kron P_i kron ... kron I_{d-1}."""
    d = len(penalties)
    sizes = [p.shape[0] for p in penalties]
    total = int(np.prod(sizes))
    result = np.zeros((total, total))
    for i, P_i in enumerate(penalties):
        I_pre = np.eye(int(np.prod(sizes[:i])))
        I_post = np.eye(int(np.prod(sizes[i + 1:])))
        result += np.kron(np.kron(I_pre, P_i), I_post)
    return result


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

        if t == 's':
            terms.append(_SplineTerm(
                feats[0],
                kw.pop('n_splines', 10),
                kw.pop('lam', 1.0),
                term_order,
                constraint=kw.pop('constraint', None),
            ))
        elif t == 'te':
            terms.append(_TensorTerm(
                feats,
                kw.pop('n_splines', 5),
                kw.pop('lam', 1.0),
                term_order,
            ))
        elif t == 'f':
            terms.append(_FactorTerm(
                feats[0],
                kw.pop('lam', 1.0),
                kw.pop('coding', 'one-hot'),
            ))
        elif t == 'l':
            terms.append(_LinearTerm(
                feats[0],
                kw.pop('lam', 0.0),
            ))
        else:
            raise ValueError(f"Unknown term type: {t}")

        if kw:
            raise ValueError(
                f"Unrecognised kwargs for {t}(): {list(kw.keys())}"
            )
    return terms