"""Single-term penalised B-spline GAM (Eilers & Marx backend)."""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ._gam import GAMCore
from ._terms import _SplineTerm


class LinGAM(GAMCore):
    """Penalised B-spline GAM for a single smooth term y = f(x) + eps."""

    def __init__(
        self,
        n_splines: int = 10,
        lam: float = 1.0,
        spline_order: int = 3,
        constraint: Optional[str] = None,
    ) -> None:
        if lam < 0:
            raise ValueError("Smoothing parameter 'lam' must be non-negative.")
        if spline_order < 1:
            raise ValueError("spline_order must be >= 1.")
        super().__init__(
            f"s(0, n_splines={int(n_splines)}, lam={float(lam)}, constraint={constraint!r})",
            spline_order=int(spline_order),
            fit_intercept=True,
        )

    @property
    def _term(self) -> _SplineTerm:
        return self._terms[0]  # type: ignore[return-value]

    @property
    def n_splines(self) -> int:
        return self._term.n_splines

    @n_splines.setter
    def n_splines(self, val: int) -> None:
        val = int(val)
        self._term.n_splines = val
        self._term.n_coefs = (
            val - self.spline_order + 1
            if self.constraint == "periodic"
            else val
        )
        self._term_configs[0]["kwargs"]["n_splines"] = val

    @property
    def lam(self) -> float:
        return self._term.lam

    @lam.setter
    def lam(self, val: float) -> None:
        self._term.lam = float(val)
        self._term_configs[0]["kwargs"]["lam"] = float(val)

    @property
    def constraint(self) -> Optional[str]:
        return self._term.constraint

    @property
    def n_coefs(self) -> int:
        """Effective number of spline coefficients (reduced for periodic)."""
        return self._term.n_coefs

    @property
    def edge_knots_(self) -> Optional[np.ndarray]:
        return self._term._edge_knots

    @property
    def design_matrix_(self) -> Optional[np.ndarray]:
        return self._design_matrix_

    @property
    def knots_(self) -> Optional[np.ndarray]:
        ek = self.edge_knots_
        if ek is None:
            return None
        scale = (ek[1] - ek[0]) or 1.0
        return (
            np.linspace(0.0, 1.0, 1 + self.n_splines - self.spline_order)
            * scale
            + ek[0]
        )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        cons = f", constraint='{self.constraint}'" if self.constraint else ""
        return (
            f"LinGAM(n_splines={self.n_splines}, lam={self.lam:.4g}, "
            f"spline_order={self.spline_order}{cons}) [{status}]"
        )

    def _diagnostics_title(self) -> str:
        cons = f", constraint='{self.constraint}'" if self.constraint else ""
        return f"LinGAM Diagnostics (K={self.n_splines}, lam={self.lam:.4f}{cons})"

    def gridsearch(
        self,
        x: ArrayLike,
        y: ArrayLike,
        lam: Optional[ArrayLike] = None,
        n_splines: Optional[ArrayLike] = None,
        gamma: float = 1.4,
        n_jobs: int = 1,
    ) -> "LinGAM":
        """Select optimal (lam, n_splines) by minimising GCV."""
        nsp = [np.atleast_1d(n_splines)] if n_splines is not None else None
        lam_g = np.atleast_1d(lam) if lam is not None else None
        super().gridsearch(
            x, y, lam=lam_g, n_splines=nsp, gamma=gamma, n_jobs=n_jobs
        )
        return self

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 300,
        figsize: Tuple[int, int] = (16, 5),
    ) -> Tuple:
        cons_str = f", constraint='{self.constraint}'" if self.constraint else ""
        return self._plot_lingam_decomposition(
            x,
            y,
            k_smooth=self.n_coefs,
            prefix="B",
            basis_title=f"B-Spline Basis (K_eff={self.n_coefs})",
            fit_title="Weighted Basis Functions + Fit",
            suptitle=f"GAM Decomposition (K={self.n_splines}{cons_str}, lam={self.lam:.4f})",
            knots=self.knots_,
            show_weighted=True,
            n_grid=n_grid,
            figsize=figsize,
        )
