# gamkit Examples

Self-contained scripts demonstrating the library's features.
Run any script from this directory:

```bash
python 01_basic_lingam.py
```

Each script generates synthetic data, fits a model, prints results to the console,
and displays matplotlib figures with `plt.show()`.

Examples **01**, **03**, and **08** include an optional comparison block
that can run automatically if reference packages are installed.

## Scripts

| # | File | What it shows | Backends compared |
|---|------|---------------|-------------------|
| 01 | `01_basic_lingam.py` | Single smooth term, grid search, prediction, confidence/prediction intervals, decomposition plot | Eilers-Marx + Wood |
| 02 | `02_multi_term_formula.py` | `GAMCore` formula interface with `s()` + `l()` terms, AIC/BIC/R^2 | Eilers-Marx + Wood |
| 03 | `03_robust_fitting.py` | Robust fitting via Huber & Tukey biweight observation `weights` on outlier-corrupted data | Eilers-Marx + Wood |
| 04 | `04_shape_constraints.py` | `mono_inc`, `mono_dec`, `convex`, `concave`, `periodic` constraints | Eilers-Marx all five; Wood for `periodic` only |
| 05 | `05_tensor_interaction.py` | 2-D tensor interaction `te()` vs additive-only model, partial grid search | Eilers-Marx + Wood |
| 06 | `06_categorical_factors.py` | Categorical variable via `f()` term | Eilers-Marx + Wood (different `f()` coding defaults) |
| 07 | `07_custom_grid_search.py` | Selective term search and custom lambda / n_splines grids | Eilers-Marx grids vs Wood REML + k-search |
| 08 | `08_intervals_and_diagnostics.py` | Confidence intervals, prediction intervals, AIC, BIC, deviance explained | Eilers-Marx + Wood (incl. CI/PI widths) |
| 09 | `09_plotting.py` | Partial dependence and decomposition plots for multi-term models | Eilers-Marx + Wood (partial-dependence agreement) |
| 10 | `10_engineering_case_study.py` | Full workflow: s+te+f+l formula, shape constraints, partial grid search, weighted fit, intervals, decomposition with named features -- see [EXAMPLE_10_README.md](EXAMPLE_10_README.md) | Eilers-Marx full; Wood for unconstrained formula |
| 11 | `11_wood_backend.py` | Wood (2011) backend: REML auto-tuning, `summary()` F-tests/p-values, `bs='tp'/'cr'/'ps'/'cc'`, tensor + linear terms | Wood + Eilers-Marx |
| 12 | `12_concrete_demo.py` | Concrete strength GAM: EDA plots, s+te+f+l formula, selective grid search, key-age predictions | Eilers-Marx + Wood |
| 13 | `13_eilers_vs_wood_comparison.py` | Head-to-head comparison with plots: univariate + multivariate fits, EDF/RSS/R²/timings | Eilers-Marx vs Wood |
| 14 | `14_wood_basis_decomposition_plots.py` | Simon Wood's y vs x test function: all plot types, basis decomposition (unweighted/weighted/cumulative), penalty effects, basis comparison | Wood + Eilers-Marx |
| 15 | `15_eilers_basis_decomposition_plots.py` | Eilers-Marx (P-spline) y vs x: uniform B-spline decomposition, partition of unity, shape constraints, visual basis comparison vs Wood | Eilers-Marx + Wood |

Every example ends with a console **backend comparison** block fitting the same
data on both backends, except where a feature exists on one backend only:

- **Eilers-Marx-only:** soft shape constraints
  (`mono_inc`, `mono_dec`, `convex`, `concave`), discrete lam/n_splines grids
  with `search_terms`.
- **Wood-only:** automatic REML smoothing, `summary()` with approximate F-tests
  and p-values, basis menu (`bs='tp'`, `'cr'`, `'ps'`, `'cc'`, …).
