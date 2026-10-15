"""Multi-term GAM with formula interface, grid search, and diagnostics."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product as cartesian_product
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.linalg import block_diag, cho_factor, cho_solve

from .._formula import parse_formula
from ._solver import solve_pirls
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
    ) -> "GAMCore":
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("Input data contains NaN or infinity.")
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if not self.is_fitted:
            self._compile(x_arr)
        self._fit_internal(x_arr, y_arr)
        return self

    def _fit_internal(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        n = len(x)
        X = self._build_model_matrix(x)
        self._design_matrix_ = X
        P = self._build_penalty()
        self._compute_coef_slices()

        coef, U1, B_solve = solve_pirls(X, y, P)
        y_hat = X @ coef
        rss = np.sum((y - y_hat) ** 2)

        self.coef_ = coef
        self._B_solve = B_solve

        edof_per_coef = np.sum(U1 ** 2, axis=1)
        edof = np.sum(edof_per_coef)

        null_deviance = float(np.sum((y - np.mean(y)) ** 2))
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
            'null_deviance': null_deviance,
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
        """Fraction of null deviance explained (pseudo R-squared)."""
        self._check_fitted()
        null_dev = self.statistics_.get('null_deviance')
        if null_dev is None or null_dev == 0.0:
            return 0.0
        return max(0.0, 1.0 - self.statistics_['rss'] / null_dev)

    # ════════════════════════════════════════════════════════════════
    #  Grid Search
    # ════════════════════════════════════════════════════════════════

    def gridsearch(
        self,
        x: ArrayLike,
        y: ArrayLike,
        lam: Optional[Union[List[ArrayLike], ArrayLike]] = None,
        n_splines: Optional[Union[List[ArrayLike], ArrayLike]] = None,
        gamma: float = 1.4,
        search_terms: Optional[Union[List[Union[int, str]], str]] = None,
        n_jobs: int = 1,
        **kwargs,
    ) -> "GAMCore":
        """Select optimal hyper-parameters via GCV grid search.

        Replicates official pyGAM's interface, design space, and search philosophy:
        - By default, searches only smoothing parameter `lam` along the diagonal
          (11 logarithmically spaced points where smooth terms share the same lambda).
        - `n_splines` is kept fixed at each term's configured value unless explicitly specified.
        - Avoids sweeping the giant design matrix inside the candidate evaluation loop
          by operating in reduced QR coordinate space (O(M^2) per model).
        - Supports parallel evaluation of candidate configurations via `n_jobs`.

        Parameters
        ----------
        x, y : array-like
            Training data.
        lam : array-like or list of array-like, optional
            Candidate lambda grid (named `lam` matching official pyGAM, alias: `lam_grids`):
            - Default: `np.logspace(-3, 3, 11)` evaluated diagonally across smooth terms.
            - 1D array: Evaluated diagonally across searched smooth terms (len(grid) models).
            - 2D array: Evaluates each row directly across terms (for random/custom search).
            - List of 1D arrays: Cartesian product across the specified subgrids.
        n_splines : array-like or list of array-like, optional
            Candidate n_splines arrays (s/te only, alias: `n_splines_grids`).
            Default: None (keeps current term values, matching pyGAM).
        gamma : float, default 1.4
            EDF penalty multiplier in GCV denominator.
        search_terms : None | str | list of int/str, optional
            Which terms to search hyper-parameters for. Default: all penalised terms.
        n_jobs : int, default 1
            Number of parallel threads for model evaluation. Use -1 for all CPU cores.

        Returns
        -------
        GAMCore
            Fitted model with best hyper-parameters.
        """
        # Support aliases
        if lam is None and 'lam_grids' in kwargs:
            lam = kwargs.pop('lam_grids')
        if n_splines is None and 'n_splines_grids' in kwargs:
            n_splines = kwargs.pop('n_splines_grids')
        if kwargs:
            raise ValueError(f"Unrecognised kwargs for gridsearch(): {list(kwargs.keys())}")

        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        return self._gridsearch(
            x_arr, y_arr,
            lam=lam,
            n_splines=n_splines,
            gamma=gamma,
            search_terms=search_terms,
            n_jobs=n_jobs,
        )

    # ── search_terms resolution ──────────────────────────────────

    def _resolve_search_terms(
        self,
        search_terms: Optional[Union[List[Union[int, str]], str]],
    ) -> Set[int]:
        if search_terms is None:
            penalised = {
                i for i, t in enumerate(self._terms)
                if not isinstance(t, _LinearTerm)
            }
            return penalised if penalised else set(range(len(self._terms)))
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

    def _expand_partial_grids(
        self,
        user_grids: list,
        search_idx: Set[int],
        grid_type: str,
    ) -> list:
        full = []
        user_idx = 0
        for i, t in enumerate(self._terms):
            if grid_type == 'lam':
                n_slots = t.d if isinstance(t, _TensorTerm) else 1
            else:
                n_slots = (
                    t.d if isinstance(t, _TensorTerm)
                    else 1 if isinstance(t, _SplineTerm)
                    else 0
                )

            if n_slots == 0:
                continue

            if i in search_idx:
                full.extend(user_grids[user_idx:user_idx + n_slots])
                user_idx += n_slots
            else:
                if grid_type == 'lam':
                    if isinstance(t, _TensorTerm):
                        full.extend([np.array([v]) for v in t.lam_per])
                    else:
                        full.append(np.array([t.lam]))
                else:
                    if isinstance(t, _TensorTerm):
                        full.extend([np.array([v]) for v in t.n_splines_per])
                    else:
                        full.append(np.array([t.n_splines]))

        return full

    def _build_combos(
        self,
        x: np.ndarray,
        lam_grids: Optional[Union[List[ArrayLike], ArrayLike]],
        n_splines_grids: Optional[Union[List[ArrayLike], ArrayLike]],
        search_terms: Optional[Union[List[Union[int, str]], str]] = None,
    ) -> Tuple[List[Any], List[Tuple[float, ...]]]:
        search_idx = self._resolve_search_terms(search_terms)

        # ── 1. Spline basis grid (all_k_combos) ───────────────────
        # By default (matching pyGAM), n_splines is NOT searched.
        if n_splines_grids is None:
            k_combo = []
            for t in self._terms:
                if isinstance(t, _SplineTerm):
                    k_combo.append(int(t.n_splines))
                elif isinstance(t, _TensorTerm):
                    k_combo.append(tuple(int(k) for k in t.n_splines_per))
                else:
                    k_combo.append(None)
            all_k_combos = [tuple(k_combo)]
        else:
            n_nsp_slots = sum(
                t.d if isinstance(t, _TensorTerm) else 1
                for t in self._terms
                if isinstance(t, (_SplineTerm, _TensorTerm))
            )
            n_nsp_searched = sum(
                t.d if isinstance(t, _TensorTerm) else 1
                for i, t in enumerate(self._terms)
                if i in search_idx and isinstance(t, (_SplineTerm, _TensorTerm))
            )

            if not isinstance(n_splines_grids, list):
                n_splines_grids = [np.atleast_1d(n_splines_grids)] * max(n_nsp_slots, 1)
            elif len(n_splines_grids) == n_nsp_searched and n_nsp_searched != n_nsp_slots:
                n_splines_grids = self._expand_partial_grids(
                    n_splines_grids, search_idx, 'nsp'
                )
            elif len(n_splines_grids) == 1 and n_nsp_slots > 1:
                n_splines_grids = n_splines_grids * n_nsp_slots
            elif len(n_splines_grids) != n_nsp_slots:
                raise ValueError(
                    f"n_splines_grids has {len(n_splines_grids)} elements, "
                    f"but model requires {n_nsp_slots} n_splines slots "
                    f"({self.formula})."
                )

            min_nsp = self.spline_order + 1
            for i, grid in enumerate(n_splines_grids):
                for v in np.atleast_1d(grid):
                    v_int = int(v)
                    if v_int <= self.spline_order:
                        raise ValueError(
                            f"n_splines_grids[{i}] has value {v_int}, "
                            f"must be > spline_order ({self.spline_order}). "
                            f"Use at least {min_nsp}."
                        )

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

        # ── 2. Lambda grid (lam_combos) ───────────────────────────
        slot_term_map: List[Tuple[int, int]] = []
        default_fixed: List[float] = []
        for ti, t in enumerate(self._terms):
            if isinstance(t, _TensorTerm):
                for d_i in range(t.d):
                    slot_term_map.append((ti, d_i))
                    default_fixed.append(float(t.lam_per[d_i]))
            else:
                slot_term_map.append((ti, 0))
                default_fixed.append(float(t.lam))

        total_slots = len(slot_term_map)
        searched_slot_indices = [
            si for si, (ti, _) in enumerate(slot_term_map) if ti in search_idx
        ]

        if lam_grids is None:
            # Official pyGAM default: 11 points diagonally across lambda space
            diag_grid = np.logspace(-3, 3, 11)
            lam_combos = []
            for val in diag_grid:
                combo = list(default_fixed)
                for si in searched_slot_indices:
                    combo[si] = float(val)
                lam_combos.append(tuple(combo))

        elif isinstance(lam_grids, np.ndarray) and lam_grids.ndim == 2:
            n_rows, n_cols = lam_grids.shape
            if n_cols == len(searched_slot_indices):
                lam_combos = []
                for row in lam_grids:
                    combo = list(default_fixed)
                    for si, val in zip(searched_slot_indices, row):
                        combo[si] = float(val)
                    lam_combos.append(tuple(combo))
            elif n_cols == total_slots:
                lam_combos = [tuple(float(v) for v in row) for row in lam_grids]
            else:
                raise ValueError(
                    f"2D lam_grids has {n_cols} columns, expected {len(searched_slot_indices)} "
                    f"for searched slots or {total_slots} for all slots."
                )

        elif isinstance(lam_grids, list) and len(lam_grids) > 1:
            # Multiple subgrids passed: Cartesian product across subgrids
            n_searched = len(searched_slot_indices)
            if len(lam_grids) == n_searched and n_searched != total_slots:
                lam_grids = self._expand_partial_grids(
                    lam_grids, search_idx, 'lam'
                )
            elif len(lam_grids) != total_slots:
                raise ValueError(
                    f"lam_grids has {len(lam_grids)} elements, "
                    f"but model requires {total_slots} lam slots "
                    f"({self.formula})."
                )
            raw_combos = list(cartesian_product(*[np.atleast_1d(g) for g in lam_grids]))
            final_combos = []
            for c in raw_combos:
                combo = list(c)
                for si, (ti, _) in enumerate(slot_term_map):
                    if ti not in search_idx:
                        combo[si] = default_fixed[si]
                final_combos.append(tuple(float(v) for v in combo))
            lam_combos = list(dict.fromkeys(final_combos))

        else:
            # 1D array or single-element list: DIAGONAL SEARCH
            sub = lam_grids[0] if isinstance(lam_grids, list) else lam_grids
            diag_grid = np.atleast_1d(sub)
            lam_combos = []
            for val in diag_grid:
                combo = list(default_fixed)
                for si in searched_slot_indices:
                    combo[si] = float(val)
                lam_combos.append(tuple(combo))

        self._compile(x)
        return all_k_combos, lam_combos

    # ── fast grid search (Cholesky) ───────────────────────────────

    def _gridsearch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam: Optional[Union[List[ArrayLike], ArrayLike]] = None,
        n_splines: Optional[Union[List[ArrayLike], ArrayLike]] = None,
        gamma: float = 1.4,
        search_terms: Optional[Union[List[Union[int, str]], str]] = None,
        n_jobs: int = 1,
    ) -> "GAMCore":
        all_k_combos, lam_combos = self._build_combos(
            x, lam, n_splines, search_terms,
        )
        n = len(y)
        yty = float(np.dot(y, y))

        def _precompute_k(k_combo):
            configs_copy = [{**cfg, 'kwargs': dict(cfg['kwargs'])} for cfg in self._term_configs]
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
            qty = Q.T @ y
            RtR = R.T @ R
            Rt_qty = R.T @ qty
            ss_orth = max(0.0, yty - float(np.dot(qty, qty)))
            bases_list, m = _precompute_penalty_bases(
                terms_copy, self.fit_intercept,
            )
            return R, RtR, Rt_qty, qty, ss_orth, bases_list, m, terms_copy

        n_k = len(all_k_combos)
        if n_jobs != 1 and n_k > 1:
            workers = os.cpu_count() or 4 if n_jobs == -1 else max(1, int(n_jobs))
            with ThreadPoolExecutor(max_workers=min(workers, n_k)) as ex:
                precomputed = list(ex.map(_precompute_k, all_k_combos))
        else:
            precomputed = [_precompute_k(c) for c in all_k_combos]

        def _eval_k_combo(k_idx):
            R, RtR, Rt_qty, qty, ss_orth, bases_list, m, terms_pc = precomputed[k_idx]
            P_buf = np.zeros((m, m))
            B_buf = np.empty((m, m))
            best_gcv_k = np.inf
            best_l_idx = -1

            for l_idx, l_combo in enumerate(lam_combos):
                _assemble_penalty_from_bases(
                    bases_list, l_combo, m, self.fit_intercept, out=P_buf,
                )
                np.add(RtR, P_buf, out=B_buf)
                B_buf.flat[::m + 1] += 1e-8
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
                    B_fallback.flat[::m + 1] += 1e-8
                    coef, _, _, _ = np.linalg.lstsq(B_fallback, Rt_qty, rcond=1e-10)
                    edf = float(np.trace(np.linalg.pinv(B_fallback, rcond=1e-10) @ RtR))

                # Never sweep giant design matrix X: compute RSS strictly in reduced M-coordinate space
                r_diff = qty - R @ coef
                rss = max(0.0, ss_orth + float(np.dot(r_diff, r_diff)))
                gcv = (n * rss) / max(n - gamma * edf, 1) ** 2

                if gcv < best_gcv_k:
                    best_gcv_k = gcv
                    best_l_idx = l_idx

            return best_gcv_k, k_idx, best_l_idx

        if n_jobs != 1 and n_k > 1:
            workers = os.cpu_count() or 4 if n_jobs == -1 else max(1, int(n_jobs))
            with ThreadPoolExecutor(max_workers=min(workers, n_k)) as ex:
                k_results = list(ex.map(_eval_k_combo, range(n_k)))
        else:
            k_results = [_eval_k_combo(i) for i in range(n_k)]

        best_gcv = np.inf
        best_k_idx = 0
        best_l_idx = 0
        for gcv_val, k_idx, l_idx in k_results:
            if gcv_val < best_gcv:
                best_gcv = gcv_val
                best_k_idx = k_idx
                best_l_idx = l_idx

        best_terms = precomputed[best_k_idx][-1]
        best_l_combo = lam_combos[best_l_idx]
        self._terms = best_terms
        _set_lams_from_combo(self._terms, best_l_combo)
        self._compile(x)
        self._fit_internal(x, y)
        return self

    # ── plotting ─────────────────────────────────────────────────

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 200,
        extrap_frac: float = 0.15,
        figsize: Optional[Tuple[int, int]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple:
        self._check_fitted()
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        def _fname(idx):
            if feature_names is not None and idx < len(feature_names):
                return feature_names[idx]
            return f'x[{idx}]'

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
        ax_overall.set_xlabel(_fname(0) if p_predictors > 1 else 'x')
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
                ax.set_xlabel(_fname(term.feature))
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
                    ax.set_xlabel(_fname(term.features[0]))
                    ax.set_ylabel(_fname(term.features[1]))
                else:
                    x_sweep = np.tile(medians, (n_grid, 1))
                    x_sweep[:, term.features[0]] = feat_grids[term.features[0]]
                    pdep = self.partial_dependence(i, x_sweep)
                    fv = feat_grids[term.features[0]]
                    ax.plot(fv, pdep, 'C2-', lw=1.5)
                    ax.set_xlabel(f'{_fname(term.features[0])} (projection)')
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
                ax.set_xlabel(_fname(term.feature))
                ax.set_ylabel('Contribution')
                ax.set_title(f'l({term.feature})   beta={beta:.4f}')

        fig.suptitle(
            f'GAMCore Decomposition — {self.formula}',
            fontsize=12, fontweight='bold',
        )
        return fig, np.array(fig.axes)

    def plot_diagnostics(
        self,
        x: ArrayLike,
        y: ArrayLike,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Tuple:
        """Diagnostic plots for model validation.

        Produces a 2×2 grid with:

        * Observed vs Fitted (with 1:1 line and :math:`R^2`)
        * Residuals vs Fitted
        * Normal Q-Q plot of residuals
        * Histogram of residuals with normal overlay

        Parameters
        ----------
        x, y : array-like
            Data used for validation (typically the training set).
        figsize : tuple of int, optional
            Figure size in inches. Default is ``(10, 9)``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of matplotlib.axes.Axes
        """
        self._check_fitted()
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        y_fit = self.predict(x_arr)
        residuals = y_arr - y_fit
        r2 = self.deviance_explained()

        figsize = figsize or (10, 9)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()

        # 1. Observed vs Fitted
        ax = axes[0]
        ax.scatter(y_fit, y_arr, alpha=0.5, edgecolors='k', s=20)
        lims = [
            np.min([y_arr.min(), y_fit.min()]),
            np.max([y_arr.max(), y_fit.max()]),
        ]
        ax.plot(lims, lims, 'r--', lw=1.5, label='1:1')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Observed values')
        ax.set_title(f'Observed vs Fitted ($R^2$={r2:.3f})')
        ax.legend()

        # 2. Residuals vs Fitted
        ax = axes[1]
        ax.scatter(y_fit, residuals, alpha=0.5, edgecolors='k', s=20)
        ax.axhline(0, color='r', linestyle='--', lw=1.5)
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted')

        # 3. Normal Q-Q
        ax = axes[2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q')
        ax.get_lines()[0].set_markerfacecolor('steelblue')
        ax.get_lines()[0].set_markeredgecolor('k')
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[1].set_color('r')

        # 4. Histogram of residuals
        ax = axes[3]
        ax.hist(
            residuals, bins='auto', density=True,
            color='steelblue', edgecolor='k', alpha=0.7,
        )
        sigma = np.std(residuals, ddof=1)
        x_norm = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(
            x_norm, stats.norm.pdf(x_norm, 0.0, sigma),
            'r-', lw=2, label='Normal approx.',
        )
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.legend()

        fig.suptitle(
            f'GAMCore Diagnostics — {self.formula}',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig, axes


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