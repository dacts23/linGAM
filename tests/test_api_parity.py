"""API parity and usability tests across Eilers-Marx and Wood backends."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from magli import (
    GAM,
    GAMCore,
    LinGAM,
    EilersGAMCore,
    EilersLinGAM,
    WoodGAMCore,
    WoodLinGAM,
)
from magli.wood import GAMCore as DirectWoodGAMCore
from magli.wood import LinGAM as DirectWoodLinGAM, GAM as DirectWoodGAM
from magli.eilers import GAMCore as DirectEilersGAMCore
from magli.eilers import LinGAM as DirectEilersLinGAM, GAM as DirectEilersGAM


def test_import_and_factory():
    """Verify that imports and backend factories work identically."""
    # Default backend is wood
    assert isinstance(GAM(k=10), WoodLinGAM)
    assert isinstance(LinGAM(k=10), WoodLinGAM)
    assert isinstance(GAMCore("s(0)"), WoodGAMCore)

    # Explicit backend selection ('wood' and 'eilers' / 'eilers')
    m_em = GAM(n_splines=10, backend="eilers")
    assert isinstance(m_em, EilersLinGAM)

    m_wd = GAM(k=10, backend="wood")
    assert isinstance(m_wd, WoodLinGAM)

    m_lingam = LinGAM(n_splines=10, backend="eilers")
    assert isinstance(m_lingam, EilersLinGAM)

    # Direct import tests
    m_direct_wd = DirectWoodGAM(k=10)
    assert isinstance(m_direct_wd, WoodLinGAM)

    m_direct_em = DirectEilersGAM(n_splines=10)
    assert isinstance(m_direct_em, EilersLinGAM)

    g_em = GAMCore("s(0)", backend="eilers")
    assert isinstance(g_em, EilersGAMCore)

    g_wd = GAMCore("s(0)", backend="wood")
    assert isinstance(g_wd, WoodGAMCore)

    g_direct_wd = DirectWoodGAMCore("s(0)")
    assert isinstance(g_direct_wd, WoodGAMCore)

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

    for backend in ("eilers", "wood"):
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

    for backend in ("eilers", "wood"):
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

    # Eilers-Marx grid search
    m_em = LinGAM(backend="eilers")
    m_em.gridsearch(x, y, n_splines=[6, 8, 10], lam=[0.01, 0.1, 1.0])
    assert m_em.is_fitted
    assert m_em.n_splines in [6, 8, 10]

    # Wood grid search
    m_wd = LinGAM(backend="wood")
    m_wd.gridsearch(x, y, n_splines=[6, 8, 10])
    assert m_wd.is_fitted
    assert m_wd.n_splines in [6, 8, 10]

    X2 = np.column_stack([x, np.random.uniform(0, 1, 50)])
    g_wd = GAMCore("s(0, bs='cr', k=10) + s(1, bs='cr', k=10)", backend="wood")
    g_wd.gridsearch(X2, y, n_splines=[6, 8], search_terms=[0])
    assert g_wd.is_fitted
    assert g_wd._terms[0].k in [6, 8]
    assert g_wd._terms[1].k == 10


def test_weights_both_backends():
    """Verify observation weights across fit() and gridsearch() on both backends."""
    np.random.seed(42)
    x = np.linspace(0, 10, 150)
    y_clean = np.sin(x)
    y = y_clean + np.random.normal(0, 0.15, len(x))
    outlier_idx = [20, 75, 120]
    y[20] += 8.0
    y[75] -= 8.0
    y[120] += 7.0

    w = np.ones_like(y)
    w[outlier_idx] = 1e-4

    for backend in ("eilers", "wood"):
        # Unit weights match unweighted fit
        m_unw = LinGAM(n_splines=14, backend=backend).fit(x, y)
        m_ones = LinGAM(n_splines=14, backend=backend).fit(
            x, y, weights=np.ones_like(y)
        )
        assert np.allclose(m_unw.coef_, m_ones.coef_)

        # Downweighting outliers recovers clean curve
        m_w = LinGAM(n_splines=14, backend=backend).fit(x, y, weights=w)
        rmse_unw = np.sqrt(np.mean((m_unw.predict(x) - y_clean) ** 2))
        rmse_w = np.sqrt(np.mean((m_w.predict(x) - y_clean) ** 2))
        assert rmse_w < rmse_unw * 0.6

        g_unw = GAMCore("s(0, k=14)", backend=backend).fit(x, y)
        g_w = GAMCore("s(0, k=14)", backend=backend).fit(x, y, weights=w)
        assert np.sqrt(np.mean((g_w.predict(x) - y_clean) ** 2)) < np.sqrt(
            np.mean((g_unw.predict(x) - y_clean) ** 2)
        ) * 0.6

        # Gridsearch with weights
        gs_w = LinGAM(backend=backend).gridsearch(
            x, y, n_splines=[10, 14], weights=w
        )
        assert np.sqrt(np.mean((gs_w.predict(x) - y_clean) ** 2)) < rmse_unw * 0.6

        # Invalid weights validation
        with pytest.raises(ValueError, match="weights length"):
            LinGAM(backend=backend).fit(x, y, weights=np.ones(5))
        with pytest.raises(ValueError, match="weights must be finite"):
            LinGAM(backend=backend).fit(x, y, weights=-np.ones_like(y))
        with pytest.raises(ValueError, match="weights must be finite"):
            LinGAM(backend=backend).fit(x, y, weights=np.zeros_like(y))

