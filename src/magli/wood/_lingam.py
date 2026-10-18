"""Single-term penalised GAM: LinGAM class (Wood backend)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ._gam import GAMCore
from ._terms import _WoodSplineTerm


class LinGAM(GAMCore):
    """Penalised GAM for a single smooth term y = beta_0 + f(x) + eps (Wood backend)."""

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
                f"Unsupported constraint '{constraint}' for Wood backend. "
                "Wood backend natively supports 'periodic' (bs='cc'). For soft shape "
                "constraints (monotonicity, convexity), use backend='eilers'."
            )

        if n_splines < 3:
            raise ValueError("n_splines (k) must be at least 3.")
        if lam is not None and lam < 0:
            raise ValueError(
                "Smoothing parameter 'lam' / 'sp' must be non-negative."
            )

        self.constraint = constraint
        self.edge_knots_: Optional[np.ndarray] = None
        self.knots_: Optional[np.ndarray] = None

        bs_lower = bs.lower()
        m_param = (
            (int(spline_order), 2)
            if bs_lower in ("bs", "b")
            else (int(spline_order) - 1, 2)
            if bs_lower in ("ps", "cp")
            else None
        )
        lam_val = float(lam) if lam is not None else None
        super().__init__(
            f"s(0, k={int(n_splines)}, bs={bs_lower!r}, sp={lam_val!r}, m={m_param!r})",
            method=method,
            gamma=gamma,
            fit_intercept=True,
            spline_order=int(spline_order),
        )

    @property
    def _term(self) -> _WoodSplineTerm:
        return self._terms[0]  # type: ignore[return-value]

    @property
    def n_splines(self) -> int:
        return self._term.k

    @n_splines.setter
    def n_splines(self, val: int) -> None:
        self._term.k = int(val)
        self._term_configs[0]["kwargs"]["k"] = int(val)

    k = n_splines

    @property
    def lam(self) -> Optional[float]:
        return self._term.sp[0] if self._term.sp else None

    @lam.setter
    def lam(self, val: Optional[float]) -> None:
        self._term.sp = [float(val)] if val is not None else None
        self._term_configs[0]["kwargs"]["sp"] = (
            float(val) if val is not None else None
        )

    sp = lam

    @property
    def bs(self) -> str:
        return self._term.bs

    @property
    def n_coefs(self) -> int:
        """Effective number of model coefficients (smooth + intercept)."""
        return len(self.coef_) if self.coef_ is not None else self.n_splines

    @property
    def design_matrix_(self) -> Optional[np.ndarray]:
        return self._design_matrix_

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        sp_str = f"sp={self.lam:.4g}" if self.lam is not None else "sp=auto"
        cons_str = (
            f", constraint='{self.constraint}'" if self.constraint else ""
        )
        return (
            f"LinGAM(k={self.n_splines}, bs='{self.bs}', {sp_str}, "
            f"method='{self.method}'{cons_str}) [{status}]"
        )

    def _diagnostics_title(self) -> str:
        return f"LinGAM Diagnostics (k={self.n_splines}, bs='{self.bs}', sp={self.lam:.4f})"

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        weights: Optional[ArrayLike] = None,
    ) -> "LinGAM":
        """Fit the GAM model using Wood REML or GCV criteria."""
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("Input data contains NaN or infinity.")
        self.edge_knots_ = np.array(
            [float(np.min(x_arr)), float(np.max(x_arr))]
        )
        cfg_sp = self._term_configs[0]["kwargs"].get("sp")
        self._term.sp = [float(cfg_sp)] if cfg_sp is not None else None
        super().fit(x_arr.reshape(-1, 1), y_arr, weights=weights)
        meta = self._term._meta
        xp = meta.get("xp") if meta.get("xp") is not None else meta.get("knots")
        if xp is None:
            xp = meta.get("Xu")
        self.knots_ = (
            np.asarray(xp)
            if xp is not None
            else np.linspace(
                self.edge_knots_[0], self.edge_knots_[1], self.n_splines
            )
        )
        return self

    def gridsearch(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_splines: Optional[ArrayLike] = None,
        lam: Optional[ArrayLike] = None,
        gamma: float = 1.0,
        weights: Optional[ArrayLike] = None,
    ) -> "LinGAM":
        """Select optimal basis dimension k (and optionally lam) via REML or GCV."""
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        k_list = [
            int(v)
            for v in np.atleast_1d(
                np.arange(5, 25, 2) if n_splines is None else n_splines
            )
        ]
        lam_list = (
            [None]
            if lam is None
            else [float(v) for v in np.atleast_1d(lam)]
        )
        crit_key = "GCV" if len(k_list) > 1 else "score"
        best_score = np.inf
        best_cand: Optional[LinGAM] = None

        for k_cand in k_list:
            for l_cand in lam_list:
                try:
                    cand = LinGAM(
                        n_splines=k_cand,
                        lam=l_cand,
                        bs=self.bs,
                        method=self.method,
                        gamma=gamma,
                        spline_order=self.spline_order,
                    ).fit(x_arr, y_arr, weights=weights)
                    if cand.statistics_[crit_key] < best_score:
                        best_score = cand.statistics_[crit_key]
                        best_cand = cand
                except Exception:
                    continue

        if best_cand is not None:
            self._term_configs = best_cand._term_configs
            self._terms = best_cand._terms
            self._coef_slices = best_cand._coef_slices
            self.coef_ = best_cand.coef_
            self.statistics_ = best_cand.statistics_
            self._design_matrix_ = best_cand._design_matrix_
            self.edge_knots_ = best_cand.edge_knots_
            self.knots_ = best_cand.knots_
            return self

        return self.fit(x_arr, y_arr, weights=weights)

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 300,
        figsize: Tuple[int, int] = (16, 5),
        show_weighted: bool = False,
    ) -> Tuple:
        """Plot basis decomposition, coefficients, and overall fit."""
        k_smooth = self._term.n_coefs
        fit_title = (
            "Weighted Basis Functions + Fit"
            if show_weighted
            else "Overall Model Fit"
        )
        return self._plot_lingam_decomposition(
            x,
            y,
            k_smooth=k_smooth,
            prefix="b",
            basis_title=f"Basis Functions (dim={k_smooth})",
            fit_title=fit_title,
            suptitle=f"LinGAM Decomposition (k={self.n_splines}, bs='{self.bs}', sp={self.lam:.4f}, method={self.method})",
            knots=self.knots_,
            show_weighted=show_weighted,
            n_grid=n_grid,
            figsize=figsize,
        )
