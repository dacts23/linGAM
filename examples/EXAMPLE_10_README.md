# Example 10 -- Engineering Case Study README

## What This Example Does

This script simulates a turbine blade life prediction problem and
demonstrates how to use **all major linGAM features** together in a
realistic workflow.  It generates synthetic data where you *know the
ground truth* so you can see exactly what the model recovers.

---

## Quick Start

```bash
python examples/10_engineering_case_study.py
```

Each section prints interpretation to the console, then shows one or more
matplotlib plots.  **Close each plot window to advance** to the next section.

---

## The Physical Setup

| Column | Variable | Unit | Expected shape |
|--------|----------|------|----------------|
| x[0] | Turbine inlet temperature | deg C | Monotonic-increasing degradation |
| x[1] | Shaft speed | 1000 rpm | Concave (optimal at 12 krpm) |
| x[2] | Blade vibration amplitude | mm/s | Synergistic with temperature |
| x[3] | Cooling mass flow rate | kg/s | Linear protective effect |
| x[4] | Blade coating type | {A,B,C} | Baseline shift (+6, 0, -8) |

The ground truth includes a **temperature x vibration interaction** --
high temperature is worse at high vibration, and vice versa.  5% of
points are artificial "sensor outliers" (shifted by sigma=15) to
demonstrate robust fitting.

---

## Section-by-Section Guide

### Section 1 -- Data Summary

Console output showing data dimensions, feature names, coating level
counts, and the known ground truth.  The model does **not** see this
ground truth -- it only sees `x` and `y`.

### Section 2 -- Baseline Fit

The model is built with this formula:

```
s(0, constraint='mono_inc') + s(1, constraint='concave')
  + te(0, 2, n_splines=6) + f(4, coding='dummy') + l(3)
```

| Term | Meaning | Why this constraint |
|------|---------|---------------------|
| `s(0, mono_inc)` | Smooth effect of temperature | Physics: higher temp always degrades |
| `s(1, concave)` | Smooth effect of shaft speed | Engineering: optimum RPM exists |
| `te(0,2)` | 2-D tensor interaction | Temp x vibration synergy |
| `f(4, dummy)` | Categorical coating type | Levels B, C relative to baseline A |
| `l(3)` | Linear cooling effect | More flow -> linearly longer life |

The baseline fit uses **default lambda and n_splines** values in a
single-shot pIRLS solve.  It's fast but not optimised.

**Key diagnostic values:**
- **GCV** (Generalised Cross-Validation): Lower = better balance of fit vs complexity.
- **EDF** (Effective Degrees of Freedom): Roughly "how many parameters the model uses."
- **R^2** (Deviance explained): Pseudo R-squared, 0-1, higher is better.
- **AIC / BIC**: Information criteria; lower is better.  BIC penalises complexity more than AIC.

### Section 3 -- Grid Search (with new partial-grid flexibility)

This is where the **new `search_terms` flexibility** shines.  We tell
the grid search: "only optimise the spline terms `s(0)` and `s(1)`."

```python
model.gridsearch(
    x, y,
    search_terms=['s'],                     # only spline terms
    lam_grids=[logspace(-2, 2, 8),          # 8 candidate lambdas for s(0)
               logspace(-2, 2, 8)],         # 8 candidate lambdas for s(1)
    n_splines_grids=[arange(7, 22, 2),      # 8 candidates for s(0)
                     arange(7, 22, 2)],     # 8 candidates for s(1)
)
```

Notice: we provide **only 2 grids** even though the model has 5 terms
(which collectively need 4 lambda slots and 4 n_splines slots).
Non-searched terms (`te`, `f`, `l`) are automatically pinned to their
current fitted values.  The old API required providing grids for
**every** term, including dummy values that were immediately
overwritten.

The grid search evaluates **4096 GCV scores** (64 lambda combos x 64
n_splines combos) and selects the best pair.

**Why search only spline terms?**
- Tensor terms have many coefficients; their optimal n_splines/lambda
  depend on the interaction structure, which is already estimated.
- Factor and linear terms have at most 1 penalty parameter --
  exhaustive search isn't worth the cost.
- You can focus the search on the terms whose nonlinear shape is most
  uncertain.

### Section 4 -- Robust Fitting

The same formula is fit with `robust=True`, which uses 10 iterations
of Huber IRLS.  Huber loss down-weights points with large residuals,
making the fit resistant to the 5% sensor outliers.

**Interpreting the comparison:**
- Robust GCV is *lower* (5.99 vs 13.81) despite similar EDF.
- This means the standard fit's GCV is inflated by outlier variance.
- In practice, run robust when you suspect data quality issues.

### Section 5 -- Constrained vs Unconstrained

We fit the same data with **shape constraints removed**:

```python
"s(0) + s(1) + te(0,2) + f(4) + l(3)"   # no mono_inc, no concave
```

The constrained model achieves nearly identical R^2 with **fewer
effective degrees of freedom** (16.8 vs 19.7).  The constraints inject
domain knowledge (temperature degradation is monotonic, RPM has an
optimum) that regularises the fit, reducing the risk of overfitting to
noise-driven wiggles.

**When to use constraints:**
- `mono_inc` / `mono_dec`: When physics guarantees monotonicity.
- `convex` / `concave`: When you expect a single peak or valley.
- `periodic`: When the feature wraps around (angles, time-of-day).
- These are **soft** constraints -- they shape the penalty, not hard-clip the curve.

### Section 6 -- Confidence & Prediction Intervals

- **Confidence Interval (CI)**: How precisely we know the *mean*
  response at each point.  Narrow CI = we're confident about the
  average behaviour.
- **Prediction Interval (PI)**: The range where a *single new
  observation* is expected to fall.  Wide PI = high process
  variability.

For this dataset, CI ~ 2.4 cycles, PI ~ 14.4 cycles.  This is
expected -- we can estimate the mean accurately, but individual blades
vary substantially.

### Section 7 -- Partial Dependence Ranges

Partial dependence isolates one term's contribution by holding all
other predictors at their medians:

```python
pd = model.partial_dependence(term_index, x)
```

The **span** (max - min) tells you each term's impact magnitude.  In
this example:

| Term | Span | Interpretation |
|------|------|----------------|
| `te(0,2)` temp x vib | 49.9 | Largest effect: the interaction dominates |
| `s(1)` shaft speed | 19.2 | Strong main effect: RPM matters a lot |
| `l(3)` cooling flow | 17.8 | Cooling is critical |
| `f(4)` coating type | 13.5 | Material choice shifts life meaningfully |
| `s(0)` temperature | 7.9 | Smallest main effect (but enables interaction) |

**Caveat:** Spans don't tell the full story.  The temperature main
effect is modest because most of its influence flows through the
interaction term.  Always check the decomposition plot to see
the *shape*, not just the magnitude.

### Section 8 -- Diagnostics Plots

**[Plot 1]** Two standard model-checking panels:

1. **Observed vs Fitted** (left): Points on the red 1:1 line = perfect
   prediction.  Systematic deviation from the line indicates model
   bias.

2. **Residuals vs Fitted** (right): Should look like random scatter
   around zero.  Patterns (funnel shapes, curvature) indicate
   heteroscedasticity or missing terms.

### Section 9 -- Decomposition Plot

**[Plot 2]** This is the most important visualisation.  It shows how
the model **decomposes** blade life into its components.

**Row 0 -- Overall fit:**
The model's predicted y when sweeping x[0] (temperature) with all
other predictors at their medians.  Gray dots = actual data.  Red
shaded edges = extrapolation (model is less reliable there).

**Row 1+ -- Per-term panels:**

| Panel shows | How to read it |
|-------------|---------------|
| `s()` curve | The **shape** of the predictor's effect.  Is it linear, saturating, peaked?  Red shading = extrapolation. |
| `te()` heatmap | The 2-D interaction surface.  Blue = below-average life, red = above-average.  Look for diagonal patterns (synergy) or checkerboards (antagonism). |
| `f()` bar chart | Baseline shift per category.  Bar above zero = higher than reference level, below = lower. |
| `l()` line | Linear slope.  The `beta` coefficient is shown in the title. |

Every panel shows **isolated contributions** -- the effect of changing
only that term while everything else stays constant.  The sum of all
term contributions (plus intercept) equals the full prediction.

**How to spot issues:**
- A smooth term that looks like noise (zig-zag) = under-smoothed (lambda too small).
- A smooth term that's nearly flat = over-smoothed (lambda too large) or the term is unnecessary.
- Extrapolation regions (red) show uncertainty: the model has no data there, treat with caution.

### Section 10 -- Robust vs Standard Comparison

**[Plot 3]** Shows how the 25 outlier points are handled differently:

- **Orange X marks**: Where the *standard* fit places outlier
  predictions.  They're pulled away from the 1:1 line because the
  outliers distort the fit.
- **Green circles**: Where the *robust* fit places those same points.
  They're closer to the 1:1 line because Huber loss down-weighted the
  outliers during fitting.
- **Blue dots**: Clean points, both fits agree on them.

---

## How to Adapt This for Your Own Data

1. **Define your formula** based on domain knowledge:
   - Continuous predictors with nonlinear effects -> `s(col_index)`
   - Two continuous predictors that interact -> `te(col_a, col_b)`
   - Categorical predictors -> `f(col_index, coding='dummy')`
   - Known linear effects -> `l(col_index)`

2. **Add shape constraints** where physics justifies them
   (mono_inc, concave, etc.).  They regularise the fit at no cost.

3. **Run grid search** on the terms whose shape you're least certain
   about.  Use `search_terms=['s']` to focus only on spline terms, or
   pass integer indices `search_terms=[0, 2]` for specific terms.

4. **Inspect the decomposition plot** -- it's your primary tool for
   understanding what the model learned.

5. **Check diagnostics** (GCV, R^2, residual plots) to confirm the
   model is adequate.  If GCV is high or residuals show patterns,
   consider adding interaction terms or different constraints.

6. **Use robust fitting** (`robust=True`) if your data might contain
   sensor errors, recording mistakes, or other outliers.

---

## Key Takeaways

| Concept | What it means |
|---------|--------------|
| GCV | Lower = better fit-penalty balance.  Your primary model selection criterion. |
| EDF | Model complexity.  More EDF = more wiggles.  EDF << n is good. |
| R^2 | Fraction of variance explained.  0.9+ is strong; 0.5 suggests missing terms. |
| Partial dependence | What one term contributes, holding others fixed.  The "explanation" of the model. |
| Decomposition plot | The visual summary of *everything the model learned*.  Read it first. |
| Shape constraints | Inject physics knowledge.  They reduce EDF without hurting fit. |
| Grid search | Automatically find best lambda and n_splines.  Use `search_terms` to limit scope. |
| Robust fitting | Immune to outliers.  Use when data quality is uncertain. |
| Confidence interval | How sure we are about the mean prediction.  Narrow = precise. |
| Prediction interval | Where new observations should fall.  Wide = high variability. |
