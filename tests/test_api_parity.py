"""API parity and usability tests across pyGAM and mgcv backends."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lingam import (
    GAMCore,
    LinGAM,
    MgcvGAMCore,
    MgcvLinGAM,
    PygamGAMCore,
    PygamLinGAM,
)
from lingam.mgcv import GAMCore as DirectMgcvGAMCore
from lingam.mgcv import LinGAM as DirectMgcvLinGAM


def test_import_and_factory():
    """Verify that imports and backend factories work identically."""
    # Factory tests
    m_py = LinGAM(n_splines=10, backend="pygam")
    assert isinstance(m_py, PygamLinGAM)

    m_mg = LinGAM(k=10, backend="mgcv")
    assert isinstance(m_mg, MgcvLinGAM)

    # Direct import tests
    m_direct = DirectMgcvLinGAM(k=10)
    assert isinstance(m_direct, MgcvLinGAM)

    g_py = GAMCore("s(0)", backend="pygam")
    assert isinstance(g_py, PygamGAMCore)

    g_mg = GAMCore("s(0)", backend="mgcv")
    assert isinstance(g_mg, MgcvGAMCore)

    g_direct = DirectMgcvGAMCore("s(0)")
    assert isinstance(g_direct, MgcvGAMCore)

    with pytest.raises(ValueError, match="Unknown backend"):
        LinGAM(backend="invalid_backend")

    with pytest.raises(ValueError, match="Unknown backend"):
        GAMCore("s(0)", backend="invalid_backend")


def test_lingam_lifecycle_and_methods():
    """Verify that both backends provide identical lifecycle methods and attributes."""
    np.random.seed(42)
    x = np.linspace(0, 1, 60)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 60)
    x_test = np.linspace(0.1, 0.9, 20)

    for backend in ("pygam", "mgcv"):
        m = LinGAM(n_splines=8, backend=backend)
        assert not m.is_fitted
        assert repr(m).startswith("LinGAM")

        with pytest.raises(RuntimeError):
            m.predict(x_test)

        # Fit
        m.fit(x, y)
        assert m.is_fitted
        assert m.coef_ is not None
        assert m.design_matrix_ is not None

        # Predict
        preds = m.predict(x_test)
        assert preds.shape == (20,)

        # Intervals
        ci = m.confidence_intervals(x_test, width=0.95)
        pi = m.prediction_intervals(x_test, width=0.95)
        assert ci.shape == (20, 2)
        assert pi.shape == (20, 2)
        assert np.all(ci[:, 0] <= ci[:, 1])
        assert np.all(pi[:, 0] <= pi[:, 1])
        # Prediction interval must be wider than confidence interval
        assert np.all((pi[:, 1] - pi[:, 0]) >= (ci[:, 1] - ci[:, 0]))

        # Common statistics
        stats = m.statistics_
        assert isinstance(stats, dict)
        for expected_key in (
            "edof",
            "edof_per_coef",
            "scale",
            "cov",
            "se",
            "rss",
            "n_samples",
            "GCV",
        ):
            assert expected_key in stats, f"Missing key '{expected_key}' in backend '{backend}'"

        # Information criteria and deviance
        ll = m.loglikelihood()
        aic = m.aic()
        bic = m.bic()
        dev_expl = m.deviance_explained()
        assert isinstance(ll, float)
        assert isinstance(aic, float)
        assert isinstance(bic, float)
        assert 0.0 <= dev_expl <= 1.0

        # Plotting
        fig, axes = m.plot_decomposition(x, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig, axes = m.plot_diagnostics(x, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_gamcore_lifecycle_and_methods():
    """Verify that GAMCore formula models work identically across both backends."""
    np.random.seed(42)
    n = 80
    x0 = np.random.uniform(0, 1, n)
    x1 = np.random.uniform(0, 1, n)
    X = np.column_stack([x0, x1])
    y = np.sin(2 * np.pi * x0) + 1.8 * x1 + np.random.normal(0, 0.1, n)
    X_test = np.column_stack([np.linspace(0.1, 0.9, 15), np.linspace(0.2, 0.8, 15)])

    formula = "s(0, n_splines=8) + l(1)"

    for backend in ("pygam", "mgcv"):
        gam = GAMCore(formula, backend=backend)
        assert not gam.is_fitted
        assert gam.n_terms == 2

        with pytest.raises(RuntimeError):
            gam.predict(X_test)

        gam.fit(X, y)
        assert gam.is_fitted
        assert gam.coef_ is not None

        preds = gam.predict(X_test)
        assert preds.shape == (15,)

        pdep0 = gam.partial_dependence(0, X_test)
        pdep1 = gam.partial_dependence(1, X_test)
        assert pdep0.shape == (15,)
        assert pdep1.shape == (15,)

        ci = gam.confidence_intervals(X_test)
        pi = gam.prediction_intervals(X_test)
        assert ci.shape == (15, 2)
        assert pi.shape == (15, 2)
        assert np.all(ci[:, 0] <= ci[:, 1])
        assert np.all(pi[:, 0] <= pi[:, 1])

        # Plotting
        fig, axes = gam.plot_decomposition(X, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig, axes = gam.plot_diagnostics(X, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_gridsearch_both_backends():
    """Verify that gridsearch works on both backends."""
    np.random.seed(42)
    x = np.linspace(0, 1, 50)
    y = np.cos(2 * np.pi * x) + np.random.normal(0, 0.1, 50)

    # PyGAM grid search
    m_py = LinGAM(backend="pygam")
    m_py.gridsearch(x, y, n_splines=[6, 8, 10], lam=[0.01, 0.1, 1.0])
    assert m_py.is_fitted
    assert m_py.n_splines in [6, 8, 10]

    # mgcv grid search
    m_mg = LinGAM(backend="mgcv")
    m_mg.gridsearch(x, y, n_splines=[6, 8, 10])
    assert m_mg.is_fitted
    assert m_mg.n_splines in [6, 8, 10]
