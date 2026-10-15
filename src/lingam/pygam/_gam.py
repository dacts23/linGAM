"""Multi-term GAM with formula interface, grid search, and diagnostics."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product as cartesian_product
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag, cho_factor, cho_solve

from .._base import _GAMBase
from .._formula import parse_formula
from ._solver import irls_solve, solve_pirls
from ._terms import (
    _FactorTerm,
    _LinearTerm,
    _SplineTerm,
    _TensorTerm,
    _Term,
    instantiate_terms,
)


class GAMCore(_GAMBase):
    """Multi-term Generalized Additive Model with formula interface."""

    _scale_is_var = False

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
        self._design_matrix_: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"GAMCore(formula='{self.formula}') [{status}]"

    def _build_penalty(self) -> np.ndarray:
        blocks = [t.build_penalty() for t in self._terms]
        if self.fit_intercept:
            blocks.append(np.zeros((1, 1)))
        return blocks[0] if len(blocks) == 1 else block_diag(*blocks)

    def fit(self, x: ArrayLike, y: ArrayLike, robust: bool = False) -> "GAMCore":
        """Fit the model."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("Input data contains NaN or infinity.")
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        self._compile(x_arr)
        self._fit_internal(x_arr, y_arr, robust=robust)
        return self

    def _fit_internal(
        self, x: np.ndarray, y: np.ndarray, robust: bool = False
    ) -> None:
        n = len(x)
        X = self._build_model_matrix(x)
        self._design_matrix_ = X
        P = self._build_penalty()
        self._compute_coef_slices()

        if robust:
            coef, U1, B_solve, w = irls_solve(X, y, P, n_iter=15)
            rss = np.sum(w * (y - X @ coef) ** 2)
        else:
            coef, U1, B_solve = solve_pirls(X, y, P)
            rss = np.sum((y - X @ coef) ** 2)

        self.coef_ = coef
        edof_per_coef = np.sum(U1**2, axis=1)
        edof = np.sum(edof_per_coef)

        null_deviance = float(np.sum((y - np.mean(y)) ** 2))
        scale = np.sqrt(rss / max(n - edof, 1))
        cov = (B_solve @ B_solve.T) * scale**2

        self.statistics_ = {
            "edof": edof,
            "edof_per_coef": edof_per_coef,
            "scale": scale,
            "cov": cov,
            "se": np.sqrt(np.diag(cov)),
            "rss": rss,
            "null_deviance": null_deviance,
            "n_samples": n,
            "GCV": (n * rss) / max(n - 1.4 * edof, 1) ** 2,
        }

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
        """Select optimal hyper-parameters via GCV grid search."""
        if lam is None and "lam_grids" in kwargs:
            lam = kwargs.pop("lam_grids")
        if n_splines is None and "n_splines_grids" in kwargs:
            n_splines = kwargs.pop("n_splines_grids")
        if kwargs:
            raise ValueError(
                f"Unrecognised kwargs for gridsearch(): {list(kwargs.keys())}"
            )

        x_arr = np.asarray(x)
        y_arr = np.asarray(y).ravel()
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        return self._gridsearch(
            x_arr,
            y_arr,
            lam=lam,
            n_splines=n_splines,
            gamma=gamma,
            search_terms=search_terms,
            n_jobs=n_jobs,
        )

    def _resolve_search_terms(
        self,
        search_terms: Optional[Union[List[Union[int, str]], str]],
    ) -> Set[int]:
        if search_terms is None:
            penalised = {
                i
                for i, t in enumerate(self._terms)
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
                        f"Term index {item} out of range (0..{len(self._terms) - 1})."
                    )
                indices.add(item)
            elif isinstance(item, str):
                indices.update(self._indices_for_type(item))
            else:
                raise TypeError(
                    f"Invalid search_terms element: {item!r}. Expected int or str."
                )
        return indices

    def _indices_for_type(self, kind: str) -> Set[int]:
        type_map = {
            "s": _SplineTerm,
            "te": _TensorTerm,
            "f": _FactorTerm,
            "l": _LinearTerm,
        }
        kind_lower = kind.lower()
        if kind_lower not in type_map:
            raise ValueError(
                f"Unknown term type '{kind}'. Expected s, te, f, or l."
            )
        cls = type_map[kind_lower]
        return {i for i, t in enumerate(self._terms) if isinstance(t, cls)}

    def _expand_partial_grids(
        self,
        user_grids: list,
        search_idx: Set[int],
        grid_type: str,
    ) -> list:
        full = []
        user_idx = 0
        for i, t in enumerate(self._terms):
            if grid_type == "lam":
                n_slots = t.d if isinstance(t, _TensorTerm) else 1
            else:
                n_slots = (
                    t.d
                    if isinstance(t, _TensorTerm)
                    else 1
                    if isinstance(t, _SplineTerm)
                    else 0
                )
            if n_slots == 0:
                continue
            if i in search_idx:
                full.extend(user_grids[user_idx : user_idx + n_slots])
                user_idx += n_slots
            elif grid_type == "lam":
                vals = t.lam_per if isinstance(t, _TensorTerm) else [t.lam]
                full.extend([np.array([v]) for v in vals])
            else:
                vals = (
                    t.n_splines_per
                    if isinstance(t, _TensorTerm)
                    else [t.n_splines]
                )
                full.extend([np.array([v]) for v in vals])
        return full

    def _build_combos(
        self,
        x: np.ndarray,
        lam_grids: Optional[Union[List[ArrayLike], ArrayLike]],
        n_splines_grids: Optional[Union[List[ArrayLike], ArrayLike]],
        search_terms: Optional[Union[List[Union[int, str]], str]] = None,
    ) -> Tuple[List[Any], List[Tuple[float, ...]]]:
        search_idx = self._resolve_search_terms(search_terms)

        if n_splines_grids is None:
            k_combo = [
                int(t.n_splines)
                if isinstance(t, _SplineTerm)
                else tuple(int(k) for k in t.n_splines_per)
                if isinstance(t, _TensorTerm)
                else None
                for t in self._terms
            ]
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
                n_splines_grids = [np.atleast_1d(n_splines_grids)] * max(
                    n_nsp_slots, 1
                )
            elif (
                len(n_splines_grids) == n_nsp_searched
                and n_nsp_searched != n_nsp_slots
            ):
                n_splines_grids = self._expand_partial_grids(
                    n_splines_grids, search_idx, "nsp"
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
                    if int(v) <= self.spline_order:
                        raise ValueError(
                            f"n_splines_grids[{i}] has value {int(v)}, "
                            f"must be > spline_order ({self.spline_order}). "
                            f"Use at least {min_nsp}."
                        )

            nsp_idx = 0
            k_combo_elements: List[List] = []
            for t in self._terms:
                if isinstance(t, _SplineTerm):
                    k_combo_elements.append(
                        [int(v) for v in np.atleast_1d(n_splines_grids[nsp_idx])]
                    )
                    nsp_idx += 1
                elif isinstance(t, _TensorTerm):
                    grids = [
                        np.atleast_1d(g)
                        for g in n_splines_grids[nsp_idx : nsp_idx + t.d]
                    ]
                    k_combo_elements.append(
                        [tuple(int(v) for v in c) for c in cartesian_product(*grids)]
                    )
                    nsp_idx += t.d
                else:
                    k_combo_elements.append([None])

            all_k_combos = list(cartesian_product(*k_combo_elements))

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
            lam_grids = np.logspace(-3, 3, 11)

        if isinstance(lam_grids, np.ndarray) and lam_grids.ndim == 2:
            n_cols = lam_grids.shape[1]
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
            n_searched = len(searched_slot_indices)
            if len(lam_grids) == n_searched and n_searched != total_slots:
                lam_grids = self._expand_partial_grids(
                    lam_grids, search_idx, "lam"
                )
            elif len(lam_grids) != total_slots:
                raise ValueError(
                    f"lam_grids has {len(lam_grids)} elements, "
                    f"but model requires {total_slots} lam slots "
                    f"({self.formula})."
                )
            raw_combos = list(
                cartesian_product(*[np.atleast_1d(g) for g in lam_grids])
            )
            final_combos = []
            for c in raw_combos:
                combo = list(c)
                for si, (ti, _) in enumerate(slot_term_map):
                    if ti not in search_idx:
                        combo[si] = default_fixed[si]
                final_combos.append(tuple(float(v) for v in combo))
            lam_combos = list(dict.fromkeys(final_combos))
        else:
            sub = lam_grids[0] if isinstance(lam_grids, list) else lam_grids
            lam_combos = []
            for val in np.atleast_1d(sub):
                combo = list(default_fixed)
                for si in searched_slot_indices:
                    combo[si] = float(val)
                lam_combos.append(tuple(combo))

        self._compile(x)
        return all_k_combos, lam_combos

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
            x, lam, n_splines, search_terms
        )
        n = len(y)
        yty = float(np.dot(y, y))

        def _precompute_k(k_combo):
            configs_copy = [
                {**cfg, "kwargs": dict(cfg["kwargs"])}
                for cfg in self._term_configs
            ]
            for ci, cfg in enumerate(configs_copy):
                k_val = k_combo[ci]
                if cfg["type"] == "s" and k_val is not None:
                    cfg["kwargs"]["n_splines"] = k_val
                elif cfg["type"] == "te" and k_val is not None:
                    cfg["kwargs"]["n_splines"] = (
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
                terms_copy, self.fit_intercept
            )
            return R, RtR, Rt_qty, qty, ss_orth, bases_list, m, terms_copy

        n_k = len(all_k_combos)
        workers = os.cpu_count() or 4 if n_jobs == -1 else max(1, int(n_jobs))
        if n_jobs != 1 and n_k > 1:
            with ThreadPoolExecutor(max_workers=min(workers, n_k)) as ex:
                precomputed = list(ex.map(_precompute_k, all_k_combos))
        else:
            precomputed = [_precompute_k(c) for c in all_k_combos]

        def _eval_k_combo(k_idx):
            R, RtR, Rt_qty, qty, ss_orth, bases_list, m, _ = precomputed[k_idx]
            P_buf = np.zeros((m, m))
            B_buf = np.empty((m, m))
            best_gcv_k = np.inf
            best_l_idx = 0

            for l_idx, l_combo in enumerate(lam_combos):
                _assemble_penalty_from_bases(bases_list, l_combo, m, out=P_buf)
                np.add(RtR, P_buf, out=B_buf)
                B_buf.flat[:: m + 1] += 1e-8
                try:
                    c_chol, lower = cho_factor(
                        B_buf, overwrite_a=True, check_finite=False
                    )
                    coef = cho_solve(
                        (c_chol, lower), Rt_qty, check_finite=False
                    )
                    edf = float(
                        np.trace(
                            cho_solve((c_chol, lower), RtR, check_finite=False)
                        )
                    )
                except Exception:
                    B_fallback = RtR + P_buf
                    B_fallback.flat[:: m + 1] += 1e-8
                    coef, _, _, _ = np.linalg.lstsq(
                        B_fallback, Rt_qty, rcond=1e-10
                    )
                    edf = float(
                        np.trace(
                            np.linalg.pinv(B_fallback, rcond=1e-10) @ RtR
                        )
                    )

                r_diff = qty - R @ coef
                rss = max(0.0, ss_orth + float(np.dot(r_diff, r_diff)))
                gcv = (n * rss) / max(n - gamma * edf, 1) ** 2
                if gcv < best_gcv_k:
                    best_gcv_k = gcv
                    best_l_idx = l_idx

            return best_gcv_k, k_idx, best_l_idx

        if n_jobs != 1 and n_k > 1:
            with ThreadPoolExecutor(max_workers=min(workers, n_k)) as ex:
                k_results = list(ex.map(_eval_k_combo, range(n_k)))
        else:
            k_results = [_eval_k_combo(i) for i in range(n_k)]

        _, best_k_idx, best_l_idx = min(k_results, key=lambda r: r[0])
        self._terms = precomputed[best_k_idx][-1]
        _set_lams_from_combo(self._terms, lam_combos[best_l_idx])
        self._compile(x)
        self._fit_internal(x, y)
        return self

    def plot_decomposition(
        self,
        x: ArrayLike,
        y: ArrayLike,
        n_grid: int = 200,
        extrap_frac: float = 0.15,
        figsize: Optional[Tuple[int, int]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple:
        return self._plot_gamcore_decomposition(
            x,
            y,
            n_grid=n_grid,
            extrap_frac=extrap_frac,
            figsize=figsize,
            feature_names=feature_names,
        )


def _set_lams_from_combo(terms: List[_Term], l_combo: tuple) -> None:
    lam_idx = 0
    for t in terms:
        if isinstance(t, _TensorTerm):
            t.lam_per = list(l_combo[lam_idx : lam_idx + t.d])
            lam_idx += t.d
        else:
            t.lam = float(l_combo[lam_idx])
            lam_idx += 1


def _assemble_penalty_from_bases(
    bases_list: list,
    l_combo: tuple,
    m: int,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    P = np.zeros((m, m)) if out is None else out
    if out is not None:
        P[:] = 0
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
        pb = t.penalty_bases()
        bases_list.append(
            ([b[0] for b in pb], (off, off + t.n_coefs), sum(b[1] for b in pb))
        )
        off += t.n_coefs
    return bases_list, off + (1 if fit_intercept else 0)


def _build_matrix_from_terms(
    terms: List[_Term],
    x: np.ndarray,
    fit_intercept: bool,
) -> np.ndarray:
    blocks = [t.build_columns(x) for t in terms]
    if fit_intercept:
        blocks.append(np.ones((len(x), 1)))
    return np.hstack(blocks)
