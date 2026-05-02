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

| # | File | What it shows |
|---|------|---------------|
| 01 | `01_basic_lingam.py` | Single smooth term, grid search, prediction, confidence/prediction intervals, decomposition plot |
| 02 | `02_multi_term_formula.py` | `GAMCore` formula interface with `s()` + `l()` terms, AIC/BIC/R² |
| 03 | `03_robust_fitting.py` | Standard vs Huber IRLS robust fit with artificial outliers |
| 04 | `04_shape_constraints.py` | `mono_inc`, `mono_dec`, `convex`, `concave`, `periodic` constraints |
| 05 | `05_tensor_interaction.py` | 2-D tensor interaction `te()` vs additive-only model |
| 06 | `06_categorical_factors.py` | Categorical variable via `f()` term |
| 07 | `07_custom_grid_search.py` | Selective term search and custom lambda / n_splines grids |
| 08 | `08_intervals_and_diagnostics.py` | Confidence intervals, prediction intervals, AIC, BIC, deviance explained |
| 09 | `09_plotting.py` | Partial dependence and decomposition plots for multi-term models |
