"""Multi-term GAM with formula interface, grid search, and diagnostics."""

from __future__ import annotations

from itertools import product as cartesian_product
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.linalg import block_diag, cho_factor, cho_solve

from ._formula import parse_formula
from ._solver import irls_solve, solve_pirls
from ._terms import (
    _FactorTerm,
    _LinearTerm,
    _SplineTerm,
    _TensorTerm,
    _Term,
    instantiate_terms,
)


class GAMCore:
    """Multi-term Generalized Additive Model with formula interface.

    Parameters
    ----------
    formula : str
        Model specification, e.g.
        ``"s(0, n_splines=10) + te(1, 2) + f(3) + l(4)"``.

        Spline terms support a ``constraint`` keyword for shape
        constraints: ``None`` (default, smooth), ``'mono_inc'``,
        ``'mono_dec'``, ``'convex'``, ``'concave'``, ``'periodic'``.

        Example::

            GAMCore("s(0, constraint='mono_inc') + s(1, constraint='periodic')")

    spline_order : int, default 3
        B-spline order (polynomial degree) for all spline/te terms.
    fit_intercept : bool, default True
        Whether to include an unpenalised intercept term.
    """

    def __init__(
        self,
        formula: str,
        spline_order: int = 3,
        fit_intercept: bool = True,
    ) -> None:
        self.formula = formula
        self.spline_order = spline_order
        self.fit_intercept = fit_intercept

        configs = parse_formula(formula)
        self._terms: List[_Term] = instantiate_terms(configs, spline_order)
        self._term_configs = configs

        self._coef_slices: List[Tuple[int, int]] = []
        self.coef_: Optional[np.ndarray] = None
        self.statistics_: Optional[Dict[str, Any]] = None
        self._B_solve: Optional[np.ndarray] = None
        self._design_matrix_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.coef_ is not None

    @property
    def n_terms(self) -> int:
        return len(self._terms)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit() or gridsearch() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"GAMCore(formula='{self.formula}') [{status}]"

    def _compile(self, x: np.ndarray) -> None:
        for term in self._terms:
            term.compile(x)

    def _total_coefs(self) -> int:
        total = sum(t.n_coefs for t in self._terms)
        if self.fit_intercept:
            total += 1
        return total

    def _build_model_matrix(self, x: np.ndarray) -> np.ndarray:
        blocks = [t.build_columns(x) for t in self._terms]
        if self.fit_intercept:
            blocks.append(np.ones((len(x), 1)))
        return np.hstack(blocks)

    def _build_penalty(self) -> np.ndarray:
        blocks = [t.build_penalty() for t in self._terms]
        if self.fit_intercept:
            blocks.append(np.zeros((1, 1)))
        return blocks[0] if len(blocks) == 1 else block_diag(*blocks)

    def _compute_coef_slices(self) -> None:
        slices = []
        offset = 0
        for t in self._terms:
            slices.append((offset, offset + t.n_coefs))
            offset += t.n_coefs
        self._coef_slices = slices

    # ── fit ──────────────────────────────────────────────────────

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        robust: bool = False,
    ) -> "GAMCore":
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if not self.is_fitted:
            self._compile(x_arr)
        self._fit_internal(x_arr, y_arr, robust=robust)
        return self

    def _fit_internal(
        self,
        x: np.ndarray,
        y: np.ndarray,
        robust: bool = False,
    ) -> None:
        n = len(x)
        X = self._build_model_matrix(x)
        self._design_matrix_ = X
        P = self._build_penalty()
        self._compute_coef_slices()

        if robust:
            coef, U1, B_solve, w, _ = irls_solve(X, y, P, n_iter=15)
            y_hat = X @ coef
            rss = np.sum(w * (y - y_hat) ** 2)
        else:
            coef, U1, B_solve = solve_pirls(X, y, P)
            y_hat = X @ coef
            rss = np.sum((y - y_hat) ** 2)

        self.coef_ = coef
        self._B_solve = B_solve

        edof_per_coef = np.sum(U1 ** 2, axis=1)
        edof = np.sum(edof_per_coef)

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

    # ── predict / partial_dependence ──────────────────────────────

    def predict(self, x: ArrayLike) -> np.ndarray:
        self._check_fitted()
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        X = self._build_model_matrix(x_arr)
        return X @ self.coef_

    def partial_dependence(self, term_idx: int, x: ArrayLike) -> np.ndarray:
        self._check_fitted()
        if term_idx < 0 or term_idx >= len(self._terms):
            raise IndexError(
                f"term_idx {term_idx} out of range (0..{len(self._terms) - 1})."
            )
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        B_term = self._terms[term_idx].build_columns(x_arr)
        start, end = self._coef_slices[term_idx]
        return B_term @ self.coef_[start:end]

    # ── intervals ────────────────────────────────────────────────

    def confidence_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        return self._compute_intervals(x, width, prediction=False)

    def prediction_intervals(self, x: ArrayLike, width: float = 0.95) -> np.ndarray:
        return self._compute_intervals(x, width, prediction=True)

    def _compute_intervals(
        self, x: np.ndarray, width: float, prediction: bool,
    ) -> np.ndarray:
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

    # ── statistics ────────────────────────────────────────────────

    def loglikelihood(self) -> float:
        """Gaussian log-likelihood of the fitted model."""
        self._check_fitted()
        n = self.statistics_['n_samples']
        rss = self.statistics_['rss']
        scale = self.statistics_['scale']
        return -0.5 * n * np.log(2 * np.pi * scale ** 2) - 0.5 * rss / scale ** 2

    def aic(self) -> float:
        """Akaike Information Criterion (AIC)."""
        self._check_fitted()
        return -2 * self.loglikelihood() + 2 * (self.statistics_['edof'] + 1)

    def bic(self) -> float:
        """Bayesian Information Criterion (BIC)."""
        self._check_fitted()
        n = self.statistics_['n_samples']
        k = self.statistics_['edof'] + 1
        return -2 * self.loglikelihood() + np.log(n) * k

    def deviance_explained(self) -> float:
        """Fraction of null deviance explained (pseudo R-squared).

        Computed as ``1 - RSS / TSS`` where TSS is the total sum of
        squares around the mean of y.
        """
        self._check_fitted()
        y = self._design_matrix_ @ self.coef_ + self.statistics_['rss'] / self.statistics_['n_samples']
        n = self.statistics_['n_samples']
        X = self._design_matrix_
        y_hat = X @ self.coef_
        y_mean = np.mean(y_hat + np.sqrt(self.statistics_['rss'] / n))
        tss = np.sum((y_hat - y_mean) ** 2) + self.statistics_['rss']
        if tss == 0:
            return 0.0
        return max(0.0, 1.0 - self.statistics_['rss'] / tss)

    # ════════════════════════════════════════════════════════════════
    #  Grid Search
    # ════════════════════════════════════════════════════════════════

    def gridsearch(
        self,
        x: ArrayLike,
        y: ArrayLike,
        lam_grids: Optional[List[ArrayLike]] = None,
        n_splines_grids: Optional[List[ArrayLike]] = None,
        gamma: float = 1.4,
        robust: bool = False,
        search_terms: Optional[Union[List[Union[int, str]], str]] = None,
    ) -> "GAMCore":
        """Select optimal hyper-parameters via GCV grid search.

        Always uses the fast QR+Cholesky path.  When ``robust=True``,
        each candidate is evaluated with Huber IRLS (iterative
        reweighting inside the Cholesky solve).

        Parameters
        ----------
        x, y : array-like
            Training data.
        lam_grids : list of array-like, optional
            Per-slot candidate lambda arrays. Default: logspace(-3, 3, 11).
        n_splines_grids : list of array-like, optional
            Per-slot candidate n_splines arrays (s/te only).
            Default: arange(5, 25, 2) for s(), arange(4, 8) for te().
        gamma : float, default 1.4
            EDF penalty multiplier in GCV denominator.
        robust : bool, default False
            Use Huber IRLS for evaluation.
        search_terms : None | str | list of int/str, optional
            Which terms to search hyper-parameters for.

        Returns
        -------
        GAMCore
            Fitted model with best hyper-parameters.
        """
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        if robust:
            return self._gridsearch_robust(
                x_arr, y_arr,
                lam_grids=lam_grids,
                n_splines_grids=n_splines_grids,
                gamma=gamma,
                search_terms=search_terms,
            )
        return self._gridsearch(
            x_arr, y_arr,
            lam_grids=lam_grids,
            n_splines_grids=n_splines_grids,
            gamma=gamma,
            search_terms=search_terms,
        )

    # ── search_terms resolution ──────────────────────────────────

    def _resolve_search_terms(
        self,
        search_terms: Optional[Union[List[Union[int, str]], str]],
    ) -> Set[int]:
        if search_terms is None:
            return set(range(len(self._terms)))
        if isinstance(search_terms, str):
            if search_terms == "all":
                return set(range(len(self._terms)))
            return self._indices_for_type(search_terms)

        indices: Set[int] = set()
        for item in search_terms:
            if isinstance(item, int):
                if item < 0 or item >= len(self._terms):
                    raise IndexError(
                        f"Term index {item} out of range "
                        f"(0..{len(self._terms) - 1})."
                    )
                indices.add(item)
            elif isinstance(item, str):
                indices.update(self._indices_for_type(item))
            else:
                raise TypeError(
                    f"Invalid search_terms element: {item!r}. "
                    f"Expected int or str."
                )
        return indices

    def _indices_for_type(self, kind: str) -> Set[int]:
        type_map = {
            's': _SplineTerm, 'te': _TensorTerm,
            'f': _FactorTerm, 'l': _LinearTerm,
        }
        kind_lower = kind.lower()
        if kind_lower not in type_map:
            raise ValueError(
                f"Unknown term type '{kind}'. Expected s, te, f, or l."
            )
        cls = type_map[kind_lower]
        return {i for i, t in enumerate(self._terms) if isinstance(t, cls)}

    # ── grid construction ────────────────────────────────────────

    def _build_default_grids(
        self,
        x: np.ndarray,
        lam_grids: Optional[List[ArrayLike]],
        n_splines_grids: Optional[List[ArrayLike]],
        search_terms: Optional[Union[List[Union[int, str]], str]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        search_idx = self._resolve_search_terms(search_terms)

        n_lam_slots = sum(
            t.d if isinstance(t, _TensorTerm) else 1
            for t in self._terms
        )
        n_nsp_slots = sum(
            t.d if isinstance(t, _TensorTerm) else 1
            for t in self._terms
            if isinstance(t, (_SplineTerm, _TensorTerm))
        )

        if lam_grids is None:
            _default_lam = [np.logspace(-3, 3, 11)]
            lam_grids = []
            for t in self._terms:
                n_slots = t.d if isinstance(t, _TensorTerm) else 1
                lam_grids.extend(_default_lam * n_slots)
        elif not isinstance(lam_grids, list):
            lam_grids = [np.atleast_1d(lam_grids)] * n_lam_slots
        elif len(lam_grids) == 1 and n_lam_slots > 1:
            lam_grids = lam_grids * n_lam_slots
        elif len(lam_grids) != n_lam_slots:
            raise ValueError(
                f"lam_grids has {len(lam_grids)} elements, "
                f"but model requires {n_lam_slots} lam slots "
                f"({self.formula})."
            )

        if n_splines_grids is None:
            n_splines_grids = []
            for t in self._terms:
                if isinstance(t, _SplineTerm):
                    n_splines_grids.append(np.arange(5, 25, 2))
                elif isinstance(t, _TensorTerm):
                    n_splines_grids.extend([np.arange(4, 8)] * t.d)
        elif not isinstance(n_splines_grids, list):
            n_splines_grids = [np.atleast_1d(n_splines_grids)] * max(n_nsp_slots, 1)
        elif len(n_splines_grids) == 1 and n_nsp_slots > 1:
            n_splines_grids = n_splines_grids * n_nsp_slots
        elif len(n_splines_grids) != n_nsp_slots:
            raise ValueError(
                f"n_splines_grids has {len(n_splines_grids)} elements, "
                f"but model requires {n_nsp_slots} n_splines slots "
                f"({self.formula})."
            )

        lam_grids = list(lam_grids)
        n_splines_grids = list(n_splines_grids)

        lam_idx = 0
        nsp_idx = 0
        for i, t in enumerate(self._terms):
            if isinstance(t, _TensorTerm):
                n_lam = t.d
                n_nsp = t.d
            elif isinstance(t, _SplineTerm):
                n_lam = 1
                n_nsp = 1
            elif isinstance(t, (_FactorTerm, _LinearTerm)):
                n_lam = 1
                n_nsp = 0
            else:
                n_lam = 1
                n_nsp = 0

            if i not in search_idx:
                if isinstance(t, _TensorTerm):
                    lam_grids[lam_idx:lam_idx + n_lam] = [
                        np.array([v]) for v in t.lam_per
                    ]
                    n_splines_grids[nsp_idx:nsp_idx + n_nsp] = [
                        np.array([v]) for v in t.n_splines_per
                    ]
                else:
                    lam_grids[lam_idx] = np.array([t.lam])
                    if isinstance(t, _SplineTerm):
                        n_splines_grids[nsp_idx] = np.array([t.n_splines])

            lam_idx += n_lam
            nsp_idx += n_nsp

        self._compile(x)
        return lam_grids, n_splines_grids

    # ── fast grid search (Cholesky) ───────────────────────────────

    def _gridsearch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam_grids: Optional[List[ArrayLike]] = None,
        n_splines_grids: Optional[List[ArrayLike]] = None,
        gamma: float = 1.4,
        search_terms: Optional[Union[List[Union[int, str]], str]] = None,
    ) -> "GAMCore":
        lam_grids, n_splines_grids = self._build_default_grids(
            x, lam_grids, n_splines_grids, search_terms,
        )
        n = len(y)

        nsp_idx = 0
        k_combo_elements: List[List] = []
        for t in self._terms:
            if isinstance(t, _SplineTerm):
                grid = np.atleast_1d(n_splines_grids[nsp_idx])
                k_combo_elements.append([int(v) for v in grid])
                nsp_idx += 1
            elif isinstance(t, _TensorTerm):
                grids = [
                    np.atleast_1d(g)
                    for g in n_splines_grids[nsp_idx:nsp_idx + t.d]
                ]
                combos = [
                    tuple(int(v) for v in c)
                    for c in cartesian_product(*grids)
                ]
                k_combo_elements.append(combos)
                nsp_idx += t.d
            else:
                k_combo_elements.append([None])

        all_k_combos = list(cartesian_product(*k_combo_elements))
        lam_combos = list(cartesian_product(*lam_grids))

        def _precompute_k(k_combo):
            configs_copy = [dict(cfg) for cfg in self._term_configs]
            for ci, cfg in enumerate(configs_copy):
                k_val = k_combo[ci]
                if cfg['type'] == 's' and k_val is not None:
                    cfg['kwargs']['n_splines'] = k_val
                elif cfg['type'] == 'te' and k_val is not None:
                    cfg['kwargs']['n_splines'] = (
                        list(k_val) if len(k_val) > 1 else k_val[0]
                    )
            terms_copy = instantiate_terms(configs_copy, self.spline_order)
            for t in terms_copy:
                t.compile(x)
            X_k = _build_matrix_from_terms(terms_copy, x, self.fit_intercept)
            Q, R = np.linalg.qr(X_k)
            RtR = R.T @ R
            Rt_qty = R.T @ (Q.T @ y)
            bases_list, m = _precompute_penalty_bases(
                terms_copy, self.fit_intercept,
            )
            return X_k, RtR, Rt_qty, bases_list, m, terms_copy

        with ThreadPoolExecutor() as executor:
            precomputed = list(executor.map(_precompute_k, all_k_combos))

        n_k = len(precomputed)

        def _eval_k_combo(k_idx):
            X_pc, RtR, Rt_qty, bases_list, m, terms_pc = precomputed[k_idx]
            P_buf = np.zeros((m, m))
            B_buf = np.empty((m, m))
            best_gcv_k = np.inf
            best_l_idx = -1

            for l_idx, l_combo in enumerate(lam_combos):
                _assemble_penalty_from_bases(
                    bases_list, l_combo, m, self.fit_intercept, out=P_buf,
                )
                np.add(RtR, P_buf, out=B_buf)
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
                    B_fallback = RtR + P_buf
                    B_fallback.flat[::m + 1] += 1e-12
                    coef = np.linalg.solve(B_fallback, Rt_qty)
                    edf = float(np.trace(np.linalg.solve(B_fallback, RtR)))

                residuals = y - X_pc @ coef
                rss = float(residuals @ residuals)
                gcv = (n * rss) / max(n - gamma * edf, 1) ** 2

                if gcv < best_gcv_k:
                    best_gcv_k = gcv
                    best_l_idx = l_idx

            return best_gcv_k, k_idx, best_l_idx

        with ThreadPoolExecutor() as executor:
            k_results = list(executor.map(_eval_k_combo, range(n_k)))

        best_gcv = np.inf
        best_k_idx = 0
        best_l_idx = 0
        for gcv_val, k_idx, l_idx in k_results:
            if gcv_val < best_gcv:
                best_gcv = gcv_val
                best_k_idx = k_idx
                best_l_idx = l_idx

        _, _, _, _, _, best_terms = precomputed[best_k_idx]
        best_l_combo = lam_combos[best_l_idx]
        self._terms = best_terms
        _set_lams_from_combo(self._terms, best_l_combo)
        self._compile(x)
        self._fit_internal(x, y, robust=False)
        return self

    # ── robust grid search (IRLS) ─────────────────────────────────

    def _gridsearch_robust(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam_grids: Optional[List[ArrayLike]] = None,
        n_splines_grids: Optional[List[ArrayLike]] = None,
        gamma: float = 1.4,
        search_terms: Optional[Union[List[Union[int, str]], str]] = None,
    ) -> "GAMCore":
        lam_grids, n_splines_grids = self._build_default_grids(
            x, lam_grids, n_splines_grids, search_terms,
        )
        n = len(y)

        nsp_idx = 0
        k_combo_elements: List[List] = []
        for t in self._terms:
            if isinstance(t, _SplineTerm):
                grid = np.atleast_1d(n_splines_grids[nsp_idx])
                k_combo_elements.append([int(v) for v in grid])
                nsp_idx += 1
            elif isinstance(t, _TensorTerm):
                grids = [
                    np.atleast_1d(g)
                    for g in n_splines_grids[nsp_idx:nsp_idx + t.d]
                ]
                combos = [
                    tuple(int(v) for v in c)
                    for c in cartesian_product(*grids)
                ]
                k_combo_elements.append(combos)
                nsp_idx += t.d
            else:
                k_combo_elements.append([None])

        all_k_combos = list(cartesian_product(*k_combo_elements))
        lam_combos = list(cartesian_product(*lam_grids))

        def _precompute_k(k_combo):
            configs_copy = [dict(cfg) for cfg in self._term_configs]
            for ci, cfg in enumerate(configs_copy):
                k_val = k_combo[ci]
                if cfg['type'] == 's' and k_val is not None:
                    cfg['kwargs']['n_splines'] = k_val
                elif cfg['type'] == 'te' and k_val is not None:
                    cfg['kwargs']['n_splines'] = (
                        list(k_val) if len(k_val) > 1 else k_val[0]
                    )
            terms_copy = instantiate_terms(configs_copy, self.spline_order)
            for t in terms_copy:
                t.compile(x)
            X_k = _build_matrix_from_terms(terms_copy, x, self.fit_intercept)
            bases_list, m = _precompute_penalty_bases(
                terms_copy, self.fit_intercept,
            )
            return X_k, bases_list, m, terms_copy

        with ThreadPoolExecutor() as executor:
            precomputed = list(executor.map(_precompute_k, all_k_combos))

        best_gcv = np.inf
        best_result: Optional[tuple] = None

        for k_idx, k_combo in enumerate(all_k_combos):
            X_full, bases_list, m, terms_k = precomputed[k_idx]
            P_buf = np.zeros((m, m))

            def _eval_lam(l_combo):
                _assemble_penalty_from_bases(
                    bases_list, l_combo, m, self.fit_intercept, out=P_buf,
                )
                P_full = P_buf.copy()
                coef, U1, _, _, _ = irls_solve(
                    X_full, y, P_full, n_iter=10,
                )
                residuals = y - X_full @ coef
                rss = float(residuals @ residuals)
                edf = float(np.sum(U1 ** 2))
                gcv = (n * rss) / max(n - gamma * edf, 1) ** 2
                return gcv, coef, edf, l_combo

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(_eval_lam, lam_combos))

            for gcv_val, coef_val, edf_val, l_combo in results:
                if gcv_val < best_gcv:
                    best_gcv = gcv_val
                    best_result = (
                        gcv_val, k_combo, l_combo,
                        coef_val, edf_val, terms_k,
                    )

        if best_result is None:
            raise RuntimeError("Grid search failed: no valid result.")

        _, best_k_combo, best_l_combo, best_coef_vec, best_edf, best_terms = best_result
        self._terms = best_terms
        _set_lams_from_combo(self._terms, best_l_combo)
        self._compile(x)
        self._fit_internal(x, y, robust=True)
        return self

    # ── plotting ─────────────────────────────────────────────────

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 200,
        extrap_frac: float = 0.15,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Tuple:
        self._check_fitted()
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        p_predictors = x_arr.shape[1]
        n_terms = len(self._terms)
        n_term_cols = min(3, n_terms)
        n_term_rows = int(np.ceil(n_terms / n_term_cols)) if n_terms > 0 else 0
        n_rows = 1 + n_term_rows
        n_cols = max(n_term_cols, 1)

        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)

        ax_overall = fig.add_subplot(gs[0, :])

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

        x_overall = np.tile(medians, (n_grid, 1))
        x_overall[:, 0] = feat_grids[0]
        y_overall = self.predict(x_overall)
        sort_idx = np.argsort(x_overall[:, 0])
        ax_overall.plot(
            x_overall[sort_idx, 0], y_overall[sort_idx],
            'C0-', lw=2, label='GAM fit',
        )
        ax_overall.scatter(
            x_arr[:, 0], y_arr, alpha=0.4, color='gray', s=10, label='Data',
        )

        x_lo, x_hi = feat_grids[0][0], feat_grids[0][-1]
        ax_overall.axvspan(x_lo, feat_mins[0], alpha=0.06, color='red')
        ax_overall.axvspan(
            feat_maxs[0], x_hi, alpha=0.06, color='red', label='Extrapolated',
        )
        ax_overall.set_xlabel('x[0]' if p_predictors > 1 else 'x')
        ax_overall.set_ylabel('y')

        constraint_info = ""
        constrained_terms = [
            (i, t) for i, t in enumerate(self._terms)
            if isinstance(t, _SplineTerm) and t.constraint
        ]
        if constrained_terms:
            constraint_info = "  constraints: " + ", ".join(
                f"s({t.feature})={t.constraint}" for _, t in constrained_terms
            )

        ax_overall.set_title(
            f'GAM Fit  (GCV={self.statistics_["GCV"]:.3f}, '
            f'EDF={self.statistics_["edof"]:.1f}, '
            f'n={self.statistics_["n_samples"]}){constraint_info}'
        )
        ax_overall.legend(fontsize=7, loc='best')

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

                label = f's({term.feature})'
                if term.constraint:
                    label += f' [{term.constraint}]'
                ax.plot(
                    fv[sort_idx_i], pdep[sort_idx_i],
                    'C1-', lw=2, label=label,
                )
                ax.axvspan(fv[0], feat_mins[term.feature], alpha=0.06, color='red')
                ax.axvspan(feat_maxs[term.feature], fv[-1], alpha=0.06, color='red')
                ax.axhline(0, color='gray', ls='--', lw=0.5)
                ax.set_xlabel(f'x[{term.feature}]')
                ax.set_ylabel('Contribution')
                title = f's({term.feature})   K={term.n_splines}, lam={term.lam:.3f}'
                if term.constraint:
                    title += f'  [{term.constraint}]'
                ax.set_title(title)
                ax.legend(fontsize=7)

            elif isinstance(term, _TensorTerm):
                if term.d == 2:
                    ng2d = min(n_grid, 75)
                    g0 = np.linspace(
                        feat_mins[term.features[0]]
                        - extrap_frac * (feat_maxs[term.features[0]] - feat_mins[term.features[0]]),
                        feat_maxs[term.features[0]]
                        + extrap_frac * (feat_maxs[term.features[0]] - feat_mins[term.features[0]]),
                        ng2d,
                    )
                    g1 = np.linspace(
                        feat_mins[term.features[1]]
                        - extrap_frac * (feat_maxs[term.features[1]] - feat_mins[term.features[1]]),
                        feat_maxs[term.features[1]]
                        + extrap_frac * (feat_maxs[term.features[1]] - feat_mins[term.features[1]]),
                        ng2d,
                    )
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
                ax.bar(
                    range(len(level_contribs)), level_contribs,
                    color=colors, edgecolor='k', lw=0.5,
                )
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

        fig.suptitle(
            f'GAMCore Decomposition — {self.formula}',
            fontsize=12, fontweight='bold',
        )
        return fig, np.array(fig.axes)


# ═════════════════════════════════════════════════════════════════════
#  Module-level helpers
# ═════════════════════════════════════════════════════════════════════


def _set_lams_from_combo(terms: List[_Term], l_combo: tuple) -> None:
    lam_idx = 0
    for t in terms:
        if isinstance(t, _TensorTerm):
            n_sl = t.d
            t.lam_per = list(l_combo[lam_idx:lam_idx + n_sl])
            lam_idx += n_sl
        else:
            t.lam = float(l_combo[lam_idx])
            lam_idx += 1


def _assemble_penalty_from_bases(
    bases_list: list,
    l_combo: tuple,
    m: int,
    fit_intercept: bool,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if out is None:
        P = np.zeros((m, m))
    else:
        out[:] = 0
        P = out
    lam_idx = 0
    for base_mats, (start, end), n_slots in bases_list:
        for k in range(n_slots):
            P[start:end, start:end] += float(l_combo[lam_idx]) * base_mats[k]
            lam_idx += 1
    return P


def _precompute_penalty_bases(
    terms: List[_Term],
    fit_intercept: bool,
) -> Tuple[list, int]:
    bases_list = []
    off = 0
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
    return bases_list, m


def _build_matrix_from_terms(
    terms: List[_Term],
    x: np.ndarray,
    fit_intercept: bool,
) -> np.ndarray:
    blocks = [t.build_columns(x) for t in terms]
    if fit_intercept:
        blocks.append(np.ones((len(x), 1)))
    return np.hstack(blocks)