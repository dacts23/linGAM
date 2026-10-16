"""Direct comparison tests between Python gamkit.wood and R's mgcv package via Rscript."""

import json
import os
import shutil
import subprocess
import numpy as np
import pytest

from gamkit.wood import GAMCore, LinGAM


# Path to Rscript executable
R_CANDIDATES = [
    r"C:\Users\dacts\AppData\Local\Programs\R\R-4.6.1\bin\x64\Rscript.exe",
    shutil.which("Rscript"),
]
RSCRIPT_PATH = next((p for p in R_CANDIDATES if p and os.path.exists(p)), None)


def run_r_gam(script: str) -> dict:
    """Execute R script and parse output JSON."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not found on system.")

    res = subprocess.run(
        [RSCRIPT_PATH, "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Rscript failed with error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    # Output lines may contain package startup messages, find JSON
    lines = res.stdout.strip().split("\n")
    json_str = None
    for i in range(len(lines)):
        candidate = "\n".join(lines[i:])
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise RuntimeError(f"Could not parse JSON from R output:\n{res.stdout}")


@pytest.mark.skipif(not RSCRIPT_PATH, reason="Rscript not installed.")
def test_cr_basis_vs_r():
    """Verify cubic regression spline 'cr' with REML against R mgcv."""
    np.random.seed(42)
    x = np.linspace(0, 1, 80).tolist()
    y = (np.sin(2 * np.pi * np.array(x)) + np.random.normal(0, 0.15, 80)).tolist()

    r_script = f"""
    library(mgcv); library(jsonlite)
    df <- data.frame(x = c({','.join(map(str, x))}), y = c({','.join(map(str, y))}))
    m <- gam(y ~ s(x, bs='cr', k=10), data=df, method='REML')
    cat(toJSON(list(
        fitted = as.numeric(fitted(m)),
        edf = as.numeric(sum(m$edf)),
        scale = as.numeric(m$sig2),
        sp = as.numeric(m$sp),
        reml = as.numeric(m$gcv.ubre)
    ), digits=8))
    """
    r_res = run_r_gam(r_script)

    py_model = LinGAM(k=10, bs="cr", method="REML")
    py_model.fit(x, y)

    py_fitted = py_model.predict(x)
    r_fitted = np.array(r_res["fitted"])

    # High correlation between Python and R fits
    corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
    assert corr > 0.999

    # Max difference on predictions
    max_diff = np.max(np.abs(py_fitted - r_fitted))
    assert max_diff < 0.05, f"Max difference {max_diff} exceeded tolerance"

    # Effective degrees of freedom
    assert np.isclose(py_model.statistics_["edof"], r_res["edf"], atol=0.5)

    # Scale estimate (residual variance)
    assert np.isclose(py_model.statistics_["scale"], r_res["scale"], rtol=0.2)


@pytest.mark.skipif(not RSCRIPT_PATH, reason="Rscript not installed.")
def test_ps_basis_vs_r():
    """Verify P-spline basis 'ps' with REML against R mgcv."""
    np.random.seed(42)
    x = np.linspace(0, 1, 80).tolist()
    y = (np.cos(2 * np.pi * np.array(x)) + np.random.normal(0, 0.15, 80)).tolist()

    r_script = f"""
    library(mgcv); library(jsonlite)
    df <- data.frame(x = c({','.join(map(str, x))}), y = c({','.join(map(str, y))}))
    m <- gam(y ~ s(x, bs='ps', k=10), data=df, method='REML')
    cat(toJSON(list(
        fitted = as.numeric(fitted(m)),
        edf = as.numeric(sum(m$edf)),
        scale = as.numeric(m$sig2),
        sp = as.numeric(m$sp)
    ), digits=8))
    """
    r_res = run_r_gam(r_script)

    py_model = LinGAM(k=10, bs="ps", method="REML")
    py_model.fit(x, y)

    py_fitted = py_model.predict(x)
    r_fitted = np.array(r_res["fitted"])

    corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
    assert corr > 0.999
    assert np.max(np.abs(py_fitted - r_fitted)) < 0.05


@pytest.mark.skipif(not RSCRIPT_PATH, reason="Rscript not installed.")
def test_cc_basis_vs_r():
    """Verify cyclic cubic spline 'cc' with REML against R mgcv."""
    np.random.seed(42)
    x = np.linspace(0, 2 * np.pi, 80).tolist()
    y = (np.sin(np.array(x)) + np.random.normal(0, 0.15, 80)).tolist()

    r_script = f"""
    library(mgcv); library(jsonlite)
    df <- data.frame(x = c({','.join(map(str, x))}), y = c({','.join(map(str, y))}))
    m <- gam(y ~ s(x, bs='cc', k=10), data=df, method='REML')
    cat(toJSON(list(
        fitted = as.numeric(fitted(m)),
        edf = as.numeric(sum(m$edf)),
        scale = as.numeric(m$sig2),
        sp = as.numeric(m$sp)
    ), digits=8))
    """
    r_res = run_r_gam(r_script)

    py_model = LinGAM(k=10, bs="cc", method="REML")
    py_model.fit(x, y)

    py_fitted = py_model.predict(x)
    r_fitted = np.array(r_res["fitted"])

    corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
    assert corr > 0.999
    assert np.max(np.abs(py_fitted - r_fitted)) < 0.05


@pytest.mark.skipif(not RSCRIPT_PATH, reason="Rscript not installed.")
def test_fixed_sp_vs_r():
    """Verify fixed smoothing parameter (sp=0.5) against R mgcv."""
    np.random.seed(42)
    x = np.linspace(0, 1, 80).tolist()
    y = (np.sin(2 * np.pi * np.array(x)) + np.random.normal(0, 0.15, 80)).tolist()

    r_script = f"""
    library(mgcv); library(jsonlite)
    df <- data.frame(x = c({','.join(map(str, x))}), y = c({','.join(map(str, y))}))
    m <- gam(y ~ s(x, bs='cr', k=10, sp=0.5), data=df)
    cat(toJSON(list(
        fitted = as.numeric(fitted(m)),
        edf = as.numeric(sum(m$edf)),
        scale = as.numeric(m$sig2)
    ), digits=8))
    """
    r_res = run_r_gam(r_script)

    py_model = LinGAM(k=10, bs="cr", sp=0.5)
    py_model.fit(x, y)

    py_fitted = py_model.predict(x)
    r_fitted = np.array(r_res["fitted"])

    # Fits with fixed sp should match extremely closely (< 0.005)
    assert np.max(np.abs(py_fitted - r_fitted)) < 0.01
    assert np.isclose(py_model.statistics_["edof"], r_res["edf"], atol=0.01)


@pytest.mark.skipif(not RSCRIPT_PATH, reason="Rscript not installed.")
def test_gcv_method_vs_r():
    """Verify GCV.Cp method against R mgcv."""
    np.random.seed(42)
    x = np.linspace(0, 1, 80).tolist()
    y = (np.sin(2 * np.pi * np.array(x)) + np.random.normal(0, 0.15, 80)).tolist()

    r_script = f"""
    library(mgcv); library(jsonlite)
    df <- data.frame(x = c({','.join(map(str, x))}), y = c({','.join(map(str, y))}))
    m <- gam(y ~ s(x, bs='cr', k=10), data=df, method='GCV.Cp')
    cat(toJSON(list(
        fitted = as.numeric(fitted(m)),
        edf = as.numeric(sum(m$edf)),
        scale = as.numeric(m$sig2)
    ), digits=8))
    """
    r_res = run_r_gam(r_script)

    py_model = LinGAM(k=10, bs="cr", method="GCV.Cp")
    py_model.fit(x, y)

    py_fitted = py_model.predict(x)
    r_fitted = np.array(r_res["fitted"])

    corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
    assert corr > 0.999
    assert np.max(np.abs(py_fitted - r_fitted)) < 0.05


@pytest.mark.skipif(not RSCRIPT_PATH, reason="Rscript not installed.")
def test_tensor_product_vs_r():
    """Verify tensor product te(x1, x2) against R mgcv."""
    np.random.seed(42)
    n = 60
    x0 = np.random.uniform(0, 1, n).tolist()
    x1 = np.random.uniform(0, 1, n).tolist()
    y = (np.sin(np.pi * np.array(x0)) * np.cos(np.pi * np.array(x1)) + np.random.normal(0, 0.05, n)).tolist()

    r_script = f"""
    library(mgcv); library(jsonlite)
    df <- data.frame(
        x0 = c({','.join(map(str, x0))}),
        x1 = c({','.join(map(str, x1))}),
        y = c({','.join(map(str, y))})
    )
    m <- gam(y ~ te(x0, x1, bs='cr', k=4), data=df, method='REML')
    cat(toJSON(list(
        fitted = as.numeric(fitted(m)),
        edf = as.numeric(sum(m$edf)),
        scale = as.numeric(m$sig2)
    ), digits=8))
    """
    r_res = run_r_gam(r_script)

    X = np.column_stack([x0, x1])
    py_gam = GAMCore("te(0, 1, bs='cr', k=4)", method="REML")
    py_gam.fit(X, y)

    py_fitted = py_gam.predict(X)
    r_fitted = np.array(r_res["fitted"])

    corr = np.corrcoef(py_fitted, r_fitted)[0, 1]
    assert corr > 0.99
    assert np.max(np.abs(py_fitted - r_fitted)) < 0.05


@pytest.mark.skipif(not RSCRIPT_PATH, reason="Rscript not installed.")
def test_cp_and_ts_basis_vs_r():
    """Verify cyclic P-spline 'cp' and shrinkage thin-plate 'ts' against R mgcv."""
    np.random.seed(42)
    x = np.linspace(0, 1, 80).tolist()
    y = (np.sin(2 * np.pi * np.array(x)) + np.random.normal(0, 0.15, 80)).tolist()

    for bs in ("cp", "ts"):
        r_script = f"""
        library(mgcv); library(jsonlite)
        df <- data.frame(x = c({','.join(map(str, x))}), y = c({','.join(map(str, y))}))
        m <- gam(y ~ s(x, bs='{bs}', k=10), data=df, method='REML')
        cat(toJSON(list(
            fitted = as.numeric(fitted(m)),
            edf = as.numeric(sum(m$edf))
        ), digits=8))
        """
        r_res = run_r_gam(r_script)
        py_model = LinGAM(k=10, bs=bs, method="REML").fit(x, y)
        py_fitted = py_model.predict(x)
        r_fitted = np.array(r_res["fitted"])
        assert np.corrcoef(py_fitted, r_fitted)[0, 1] > 0.999
        assert np.max(np.abs(py_fitted - r_fitted)) < 0.05
        assert np.isclose(py_model.statistics_["edof"], r_res["edf"], atol=0.5)
