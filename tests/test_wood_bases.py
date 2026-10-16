"""Tests for all Wood basis families: tp, ts, cr, cs, cc, ps, cp, bs, re, and te."""

import numpy as np
import pytest
from gamkit.wood import GAMCore, LinGAM



@pytest.mark.parametrize("basis", ["tp", "ts", "cr", "cs", "cc", "ps", "cp", "bs", "re"])
def test_all_univariate_bases(basis):
    """Test that every basis can fit, predict, and compute intervals without error."""
    np.random.seed(123)
    n = 60
    if basis == "re":
        # Categorical covariate for random effect
        x = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n)
        y = 2.0 * x + np.random.normal(0, 0.2, n)
        x_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    else:
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
        x_test = np.linspace(0.05, 0.95, 25)

    model = LinGAM(k=8, bs=basis, method="REML")
    model.fit(x, y)
    assert model.is_fitted
    assert model.coef_ is not None

    preds = model.predict(x_test)
    assert preds.shape == (len(x_test),)
    assert np.all(np.isfinite(preds))

    ci = model.confidence_intervals(x_test)
    pi = model.prediction_intervals(x_test)
    assert ci.shape == (len(x_test), 2)
    assert pi.shape == (len(x_test), 2)
    assert np.all(ci[:, 0] <= ci[:, 1])
    assert np.all(pi[:, 0] <= pi[:, 1])

    # Summary
    s = model.summary()
    assert isinstance(s, str)
    assert "Parametric coefficients" in s
    assert "Approximate significance of smooth terms" in s


def test_tensor_product_basis():
    """Test 2D tensor product te(0, 1) interaction term."""
    np.random.seed(42)
    n = 100
    x0 = np.random.uniform(0, 1, n)
    x1 = np.random.uniform(0, 1, n)
    X = np.column_stack([x0, x1])
    y = np.sin(np.pi * x0) * np.cos(np.pi * x1) + np.random.normal(0, 0.05, n)

    gam = GAMCore("te(0, 1, bs='cr', k=4)")
    gam.fit(X, y)
    assert gam.is_fitted

    preds = gam.predict(X[:10])
    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))

    ci = gam.confidence_intervals(X[:10])
    assert ci.shape == (10, 2)
    assert np.all(ci[:, 0] <= ci[:, 1])


def test_periodic_constraint():
    """Test periodic smooth term constraint='periodic' and cyclic P-spline 'cp'."""
    np.random.seed(42)
    n = 60
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + np.random.normal(0, 0.1, n)

    for m in (LinGAM(k=8, constraint="periodic"), LinGAM(k=8, bs="cp")):
        m.fit(x, y)
        assert m.is_fitted
        pred_0 = m.predict([0.0])[0]
        pred_2pi = m.predict([2 * np.pi])[0]
        assert np.isclose(pred_0, pred_2pi, atol=1e-6)


def test_ts_shrinks_null_space():
    from gamkit.wood._smooths import build_tp_basis

    x = np.linspace(0, 1, 50)
    _, [S_tp], _ = build_tp_basis(x, k=8, shrink=False)
    _, [S_ts], _ = build_tp_basis(x, k=8, shrink=True)
    ev_tp = np.linalg.eigvalsh(S_tp)
    ev_ts = np.linalg.eigvalsh(S_ts)
    assert np.allclose(ev_tp[:2], 0.0, atol=1e-12)
    assert np.allclose(ev_ts[:2], ev_tp[2] * 0.1, rtol=1e-6)
    assert np.allclose(ev_ts[2:], ev_tp[2:], rtol=1e-6)


def test_gcv_method():
    """Test GCV.Cp smoothing parameter estimation method."""
    np.random.seed(42)
    x = np.linspace(0, 1, 80)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 80)

    m = LinGAM(k=10, bs="cr", method="GCV.Cp")
    m.fit(x, y)
    assert m.is_fitted
    assert m.statistics_["method"] == "GCV.CP"
    assert m.statistics_["score"] > 0


def test_multi_penalty_reml_and_gcv():
    """Verify multi-penalty REML and GCV optimization across disjoint and overlapping penalties."""
    np.random.seed(42)
    n = 80
    X = np.random.uniform(0, 1, (n, 3))
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1]) + np.random.normal(0, 0.08, n)

    for method in ("REML", "GCV.Cp"):
        gam = GAMCore("s(0, bs='cr', k=8) + te(1, 2, bs='cr', k=5)", method=method).fit(X, y)
        assert gam.is_fitted
        assert len(gam.statistics_["sp"]) == 3
        assert gam.deviance_explained() > 0.9
        # x2 is pure noise so its smoothing parameter should be much larger than x1's
        assert gam.statistics_["sp"][2] > gam.statistics_["sp"][1] * 10
        assert np.allclose(gam.predict(X), gam.predict(X + 0.0))
