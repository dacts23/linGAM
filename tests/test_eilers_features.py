"""Tests for Eilers-Marx features: extrapolation, constraints, and validation."""

import numpy as np
import pytest
from gamkit import GAMCore, LinGAM



def test_bspline_linear_extrapolation():
    """Verify that predictions extrapolate linearly beyond the edge knots."""
    np.random.seed(42)
    x = np.linspace(0.0, 1.0, 50)
    y = 2.0 * x + 1.0

    m = LinGAM(n_splines=6, lam=1.0, backend="eilers")
    m.fit(x, y)

    # Predict outside [0, 1]
    x_extrap = np.array([-0.5, -0.2, 1.2, 1.5])
    preds = m.predict(x_extrap)

    assert np.all(np.isfinite(preds))
    # Linearity check outside bounds: (f(1.5) - f(1.2)) / 0.3 should equal slope
    slope_right = (preds[3] - preds[2]) / 0.3
    slope_left = (preds[1] - preds[0]) / 0.3
    assert np.isclose(slope_right, 2.0, atol=0.1)
    assert np.isclose(slope_left, 2.0, atol=0.1)


def test_nan_input_validation():
    """Verify that NaN and inf values raise ValueError across both backends."""
    x = np.array([0.1, 0.2, np.nan, 0.4])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    for backend in ("eilers", "wood"):
        m = LinGAM(backend=backend)
        with pytest.raises(ValueError, match="Input data contains NaN or infinity"):
            m.fit(x, y)

        g = GAMCore("s(0)", backend=backend)
        with pytest.raises(ValueError, match="Input data contains NaN or infinity"):
            g.fit(x.reshape(-1, 1), y)

    # Test Inf
    x_inf = np.array([0.1, 0.2, np.inf, 0.4])
    for backend in ("eilers", "wood"):
        m = LinGAM(backend=backend)
        with pytest.raises(ValueError, match="Input data contains NaN or infinity"):
            m.fit(x_inf, y)


def test_deviance_explained_accuracy():
    """Verify that deviance_explained strictly matches 1 - RSS / TSS."""
    np.random.seed(42)
    x = np.linspace(0, 1, 80)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 80)

    # Single-term LinGAM
    m = LinGAM(n_splines=10, lam=0.1, backend="eilers")
    m.fit(x, y)

    tss = np.sum((y - np.mean(y)) ** 2)
    expected_r2 = 1.0 - m.statistics_["rss"] / tss
    assert np.isclose(m.deviance_explained(), expected_r2, atol=1e-7)

    # Multi-term GAMCore
    g = GAMCore("s(0, n_splines=10)", backend="eilers")
    g.fit(x.reshape(-1, 1), y)
    expected_r2_g = 1.0 - g.statistics_["rss"] / tss
    assert np.isclose(g.deviance_explained(), expected_r2_g, atol=1e-7)


def test_eilers_gridsearch_diagonal_and_design_space():
    """Verify that GAMCore gridsearch in Eilers-Marx backend searches diagonal and custom design spaces."""
    np.random.seed(42)
    n = 100
    X = np.random.uniform(-1, 1, (n, 3))
    y = np.sin(np.pi * X[:, 0]) + 0.5 * (X[:, 1] ** 2) + 1.2 * X[:, 2] + np.random.normal(0, 0.2, n)

    # 1. Default gridsearch: 11 points diagonally across lambda space, n_splines fixed
    g = GAMCore("s(0, n_splines=10) + s(1, n_splines=8) + l(2)", backend="eilers")
    g.gridsearch(X, y)
    assert g.is_fitted
    assert g._terms[0].n_splines == 10
    assert g._terms[1].n_splines == 8
    assert g._terms[2].lam == 0.0  # Linear term remains unpenalised
    assert g._terms[0].lam == g._terms[1].lam  # Diagonal search shares lambda

    # 2. 2D array row-wise custom search
    custom_lams = np.array([
        [0.01, 10.0],
        [1.0, 0.1],
        [5.0, 5.0],
    ])
    g2 = GAMCore("s(0) + s(1)", backend="eilers")
    g2.gridsearch(X, y, lam_grids=custom_lams)
    assert g2.is_fitted
    found_row = any(
        np.allclose([g2._terms[0].lam, g2._terms[1].lam], row)
        for row in custom_lams
    )
    assert found_row

    # 3. Keyword alias support ('lam' and 'n_splines') and parallel evaluation
    g3 = GAMCore("s(0) + s(1)", backend="eilers")
    g3.gridsearch(X, y, lam=np.logspace(-2, 2, 5), n_jobs=2)
    assert g3.is_fitted

    # 4. LinGAM parallel evaluation with n_jobs
    m = LinGAM(backend="eilers")
    m.gridsearch(X[:, 0], y, lam=[0.01, 1.0], n_splines=[6, 8], n_jobs=-1)
    assert m.is_fitted

