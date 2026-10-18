"""Tests for Eilers-Marx features: extrapolation, constraints, validation, and pyGAM parity."""

import numpy as np
import pytest
from scipy import stats

from magli import GAMCore, LinGAM
from magli.eilers._bspline import b_spline_basis


def test_bspline_linear_extrapolation():
    """Verify that predictions extrapolate linearly beyond the edge knots."""
    np.random.seed(42)
    x = np.linspace(0.0, 1.0, 50)
    y = 2.0 * x + 1.0

    m = LinGAM(n_splines=6, lam=1.0, backend="eilers")
    m.fit(x, y)

    x_extrap = np.array([-0.5, -0.2, 1.2, 1.5])
    preds = m.predict(x_extrap)

    assert np.all(np.isfinite(preds))
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

    m = LinGAM(n_splines=10, lam=0.1, backend="eilers")
    m.fit(x, y)

    tss = np.sum((y - np.mean(y)) ** 2)
    expected_r2 = 1.0 - m.statistics_["rss"] / tss
    assert np.isclose(m.deviance_explained(), expected_r2, atol=1e-7)

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

    g = GAMCore("s(0, n_splines=10) + s(1, n_splines=8) + l(2)", backend="eilers")
    g.gridsearch(X, y)
    assert g.is_fitted
    assert g._terms[0].n_splines == 10
    assert g._terms[1].n_splines == 8
    assert g._terms[2].lam == 0.0
    assert g._terms[0].lam == g._terms[1].lam

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

    g3 = GAMCore("s(0) + s(1)", backend="eilers")
    g3.gridsearch(X, y, lam=np.logspace(-2, 2, 5), n_jobs=2)
    assert g3.is_fitted

    m = LinGAM(backend="eilers")
    m.gridsearch(X[:, 0], y, lam=[0.01, 1.0], n_splines=[6, 8], n_jobs=-1)
    assert m.is_fitted


def test_periodic_bspline_partition_of_unity_and_continuity():
    """Verify periodic B-spline wraps k columns, sums to 1, and satisfies B(0) == B(1)."""
    x = np.linspace(0.0, 1.0, 101)
    B = b_spline_basis(x, edge_knots=np.array([0.0, 1.0]), n_splines=10, spline_order=3, periodic=True)
    assert B.shape == (101, 7)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-14)
    assert np.allclose(B[0], B[-1], atol=1e-14)


def test_directional_shape_constraints_enforced():
    """Verify mono_inc, mono_dec, convex, and concave enforce their target shapes on non-conforming data."""
    np.random.seed(42)
    x = np.linspace(0.0, 1.0, 80)
    y = np.sin(2.0 * np.pi * x) + np.random.normal(0.0, 0.1, 80)
    xg = np.linspace(0.0, 1.0, 200)

    p_inc = LinGAM(n_splines=12, lam=0.6, constraint="mono_inc", backend="eilers").fit(x, y).predict(xg)
    p_dec = LinGAM(n_splines=12, lam=0.6, constraint="mono_dec", backend="eilers").fit(x, y).predict(xg)
    p_cvx = LinGAM(n_splines=12, lam=0.6, constraint="convex", backend="eilers").fit(x, y).predict(xg)
    p_ccv = LinGAM(n_splines=12, lam=0.6, constraint="concave", backend="eilers").fit(x, y).predict(xg)

    assert np.min(np.diff(p_inc)) >= -1e-6
    assert np.max(np.diff(p_dec)) <= 1e-6
    assert np.min(np.diff(p_cvx, n=2)) >= -1e-5
    assert np.max(np.diff(p_ccv, n=2)) <= 1e-5
    assert not np.allclose(p_inc, p_dec)
    assert not np.allclose(p_cvx, p_ccv)


def test_eilers_vs_pygam_parity():
    """Verify exact numerical parity against pyGAM for unconstrained and constrained splines."""
    pygam = pytest.importorskip("pygam")
    np.random.seed(42)
    x = np.linspace(0.0, 1.0, 60)
    y = np.sin(2.0 * np.pi * x) + np.random.normal(0.0, 0.15, 60)

    pg = pygam.LinearGAM(pygam.s(0, n_splines=10, lam=0.6, spline_order=3), fit_intercept=True).fit(x, y)
    eg = LinGAM(n_splines=10, lam=0.6, spline_order=3, backend="eilers").fit(x, y)

    assert np.allclose(eg.predict(x), pg.predict(x), atol=1e-12)
    assert np.isclose(eg.statistics_["edof"], pg.statistics_["edof"], atol=1e-12)
    assert np.isclose(eg.statistics_["scale"], pg.statistics_["scale"], atol=1e-12)
    assert np.isclose(eg.statistics_["GCV"], pg.statistics_["GCV"], atol=1e-12)
    assert np.allclose(eg.statistics_["cov"], pg.statistics_["cov"], atol=1e-12)

    for c_pg, c_mg in [
        ("monotonic_inc", "mono_inc"),
        ("monotonic_dec", "mono_dec"),
        ("convex", "convex"),
        ("concave", "concave"),
    ]:
        pg_c = pygam.LinearGAM(
            pygam.s(0, n_splines=10, lam=0.6, constraints=c_pg), fit_intercept=True
        ).fit(x, y)
        eg_c = LinGAM(n_splines=10, lam=0.6, constraint=c_mg, backend="eilers").fit(x, y)
        assert np.allclose(eg_c.predict(x), pg_c.predict(x), atol=1e-6)
        assert np.isclose(eg_c.statistics_["edof"], pg_c.statistics_["edof"], atol=1e-5)


def test_eilers_n_less_than_m_and_custom_gamma():
    """Verify solve_pirls and GAMCore in the n < m regime and custom gamma preservation."""
    np.random.seed(42)
    x = np.linspace(0.0, 1.0, 12)
    y = np.sin(2.0 * np.pi * x)

    m = LinGAM(n_splines=20, lam=1.0, backend="eilers").fit(x, y)
    X = m.design_matrix_
    P = m._build_penalty()
    eps = np.finfo(float).eps
    beta_exact = np.linalg.solve(X.T @ X + P + np.sqrt(eps) * np.eye(X.shape[1]), X.T @ y)

    assert np.allclose(m.predict(x), X @ beta_exact, atol=1e-12)
    assert np.allclose(m.coef_, beta_exact, atol=1e-7)
    assert len(m.statistics_["edof_per_coef"]) == X.shape[1]

    m_gs = LinGAM(backend="eilers").gridsearch(x, y, lam=[0.1, 1.0], n_splines=[8], gamma=2.0)
    n = len(x)
    rss = m_gs.statistics_["rss"]
    edof = m_gs.statistics_["edof"]
    assert np.isclose(m_gs.statistics_["GCV"], (n * rss) / max(n - 2.0 * edof, 1.0) ** 2, atol=1e-12)


def test_weighted_loglikelihood_includes_log_weights():
    """Verify Gaussian loglikelihood includes 0.5 * sum(log(w)) when observation weights are used."""
    np.random.seed(42)
    x = np.linspace(0.0, 1.0, 50)
    y = np.sin(2.0 * np.pi * x) + np.random.normal(0.0, 0.1, 50)
    w = np.linspace(0.5, 2.0, 50)

    for backend in ("eilers", "wood"):
        m = LinGAM(n_splines=8, backend=backend).fit(x, y, weights=w)
        mu = m.predict(x)
        s = m.statistics_["scale"]
        sigma = np.sqrt(s) if backend == "wood" else s
        expected_ll = float(np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma / np.sqrt(w))))
        assert np.isclose(m.loglikelihood(), expected_ll, atol=1e-10)
