# linGAM Examples

Self-contained scripts demonstrating the library's features.
Run any script from this directory:

```bash
python 01_basic_lingam.py
```

Each script generates synthetic data, fits a model, prints results to the console,
and displays matplotlib figures with `plt.show()`.

Examples **01**, **03**, and **08** include an optional **pyGAM comparison** block
that runs automatically if `pygam` is installed, showing that linGAM produces
numerically equivalent results to pyGAM's `LinearGAM`.

## Scripts

| # | File | What it shows | Backends compared |
|---|------|---------------|-------------------|
| 01 | `01_basic_lingam.py` | Single smooth term, grid search, prediction, confidence/prediction intervals, decomposition plot | pyGAM + mgcv |
| 02 | `02_multi_term_formula.py` | `GAMCore` formula interface with `s()` + `l()` terms, AIC/BIC/R^2 | pyGAM + mgcv |
| 03 | `03_robust_fitting.py` | Standard vs Huber IRLS robust fit with artificial outliers | pyGAM robust vs std, mgcv std ref (robust is pyGAM-only) |
| 04 | `04_shape_constraints.py` | `mono_inc`, `mono_dec`, `convex`, `concave`, `periodic` constraints | pyGAM all five; mgcv for `periodic` only |
| 05 | `05_tensor_interaction.py` | 2-D tensor interaction `te()` vs additive-only model, partial grid search | pyGAM + mgcv |
| 06 | `06_categorical_factors.py` | Categorical variable via `f()` term | pyGAM + mgcv (different `f()` coding defaults) |
| 07 | `07_custom_grid_search.py` | Selective term search and custom lambda / n_splines grids | pyGAM grids vs mgcv REML + k-search |
| 08 | `08_intervals_and_diagnostics.py` | Confidence intervals, prediction intervals, AIC, BIC, deviance explained | pyGAM + mgcv (incl. CI/PI widths) |
| 09 | `09_plotting.py` | Partial dependence and decomposition plots for multi-term models | pyGAM + mgcv (partial-dependence agreement) |
| 10 | `10_engineering_case_study.py` | Full workflow: s+te+f+l formula, shape constraints, partial grid search, robust fit, intervals, decomposition with named features -- see [EXAMPLE_10_README.md](EXAMPLE_10_README.md) | pyGAM full; mgcv for unconstrained formula |
| 11 | `11_mgcv_backend.py` | R mgcv backend: REML auto-tuning, `summary()` F-tests/p-values, `bs='tp'/'cr'/'ps'/'cc'`, tensor + linear terms | mgcv + pyGAM |
| 12 | `12_concrete_demo.py` | Concrete strength GAM: EDA plots, s+te+f+l formula, selective grid search, key-age predictions | pyGAM + mgcv |
| 13 | `13_pygam_vs_mgcv_comparison.py` | Head-to-head comparison with plots: univariate + multivariate fits, EDF/RSS/R²/timings | pyGAM vs mgcv |

Every example ends with a console **backend comparison** block fitting the same
data on both backends, except where a feature exists on one backend only:

- **pyGAM-only:** Huber IRLS `robust=True`, soft shape constraints
  (`mono_inc`, `mono_dec`, `convex`, `concave`), discrete lam/n_splines grids
  with `search_terms`.
- **mgcv-only:** automatic REML smoothing, `summary()` with approximate F-tests
  and p-values, basis menu (`bs='tp'`, `'cr'`, `'ps'`, `'cc'`, …).
