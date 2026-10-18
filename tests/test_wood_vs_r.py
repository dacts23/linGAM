"""Direct numerical parity tests between magli.wood and R's mgcv package via Rscript."""

import json
import os
import shutil
import subprocess
import numpy as np
import pytest

from magli.wood import GAMCore, LinGAM
from magli.wood._smooths import build_smooth_basis

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
        raise RuntimeError(
            f"Rscript failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )

    lines = res.stdout.strip().split("\n")
    for i in range(len(lines)):
        try:
            return json.loads("\n".join(lines[i:]))
        except Exception:
            continue
    raise RuntimeError(f"Could not parse JSON from R output:\n{res.stdout}")


@pytest.fixture(scope="module")
def r_mgcv_suite():
    """Run a single batched Rscript call for all bases and models."""
    if not RSCRIPT_PATH:
        pytest.skip("Rscript not installed.")

    np.random.seed(42)
    n = 60
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + np.random.normal(0.0, 0.15, n)

    np.random.seed(42)
    x0 = np.random.uniform(0.0, 1.0, n)
    x1 = np.random.uniform(0.0, 1.0, n)
    y_2d = (
        np.sin(np.pi * x0) * np.cos(np.pi * x1)
        + np.random.normal(0.0, 0.05, n)
    )
    y_add = (
        np.sin(2.0 * np.pi * x0)
        + 0.5 * np.cos(2.0 * np.pi * x1)
        + np.random.normal(0.0, 0.1, n)
    )

    x_s = ",".join(f"{v:.15g}" for v in x)
    y_s = ",".join(f"{v:.15g}" for v in y)
    x0_s = ",".join(f"{v:.15g}" for v in x0)
    x1_s = ",".join(f"{v:.15g}" for v in x1)
    y2d_s = ",".join(f"{v:.15g}" for v in y_2d)
    yadd_s = ",".join(f"{v:.15g}" for v in y_add)

    r_script = f"""
    suppressPackageStartupMessages(library(mgcv))
    suppressPackageStartupMessages(library(jsonlite))
    df <- data.frame(x = c({x_s}), y = c({y_s}))
    df2 <- data.frame(x0 = c({x0_s}), x1 = c({x1_s}), y2d = c({y2d_s}), yadd = c({yadd_s}))

    bases <- c("tp", "ts", "cr", "cs", "cc", "ps", "cp", "bs")
    smooth_con <- list()
    fits <- list()

    for (bs in bases) {{
      sm <- smoothCon(s(x, bs=bs, k=8), data=df, absorb.cons=FALSE, scale.penalty=FALSE)[[1]]
      smooth_con[[bs]] <- list(
        X = sm$X,
        S = sm$S[[1]],
        S_ev = sort(eigen(sm$S[[1]], symmetric=TRUE, only.values=TRUE)$values)
      )
      for (method in c("REML", "GCV.Cp")) {{
        m <- gam(y ~ s(x, bs=bs, k=8), data=df, method=method)
        fits[[paste0(bs, "_", method)]] <- list(
          fitted = as.numeric(fitted(m)),
          edf = as.numeric(sum(m$edf)),
          scale = as.numeric(m$sig2),
          sp = as.numeric(m$sp),
          score = as.numeric(m$gcv.ubre)
        )
      }}
    }}

    m_fixed <- gam(y ~ s(x, bs="cr", k=10, sp=0.5), data=df, method="REML")
    fixed_res <- list(
      fitted = as.numeric(fitted(m_fixed)),
      edf = as.numeric(sum(m_fixed$edf)),
      scale = as.numeric(m_fixed$sig2)
    )

    m_add <- gam(yadd ~ s(x0, bs="tp", k=8) + s(x1, bs="cr", k=8), data=df2, method="REML")
    add_res <- list(
      fitted = as.numeric(fitted(m_add)),
      edf = as.numeric(sum(m_add$edf)),
      scale = as.numeric(m_add$sig2),
      sp = as.numeric(m_add$sp)
    )

    m_te <- gam(y2d ~ te(x0, x1, bs="cr", k=4), data=df2, method="REML")
    te_res <- list(
      fitted = as.numeric(fitted(m_te)),
      edf = as.numeric(sum(m_te$edf)),
      scale = as.numeric(m_te$sig2)
    )

    cat(toJSON(list(
      smooth_con = smooth_con,
      fits = fits,
      fixed = fixed_res,
      add = add_res,
      te = te_res
    ), digits=15, auto_unbox=TRUE))
    """
    return {
        "x": x,
        "y": y,
        "x0": x0,
        "x1": x1,
        "y_2d": y_2d,
        "y_add": y_add,
        "r": run_r_gam(r_script),
    }


@pytest.mark.parametrize("bs", ["tp", "ts", "cr", "cs", "cc", "ps", "cp", "bs"])
def test_smoothcon_basis_and_penalty_vs_r(r_mgcv_suite, bs):
    """Verify raw basis matrix X and penalty matrix S against mgcv::smoothCon."""
    x = r_mgcv_suite["x"]
    r_sm = r_mgcv_suite["r"]["smooth_con"][bs]
    X_r = np.array(r_sm["X"])
    S_r = np.array(r_sm["S"])
    ev_r = np.array(r_sm["S_ev"])

    X_py, [S_py], _ = build_smooth_basis(x, bs=bs, k=8)
    ev_py = np.linalg.eigvalsh(S_py)

    assert np.allclose(ev_py, ev_r, rtol=1e-4, atol=1e-8)
    if bs in ("tp", "ts"):
        # Eigenvector sign conventions can differ by +-1; column spaces must match
        proj_r = X_r @ np.linalg.pinv(X_r)
        assert np.allclose(proj_r @ X_py, X_py, atol=1e-10)
    elif bs == "cs":
        # cs assigns distinct weights (0.1, 0.01) to the 2D degenerate null space of cr
        assert np.allclose(X_py, X_r, atol=1e-12)
        assert np.allclose(S_py, S_r, rtol=1e-3, atol=2.0)
    else:
        assert np.allclose(X_py, X_r, atol=1e-12)
        assert np.allclose(S_py, S_r, rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize("bs", ["tp", "ts", "cr", "cs", "cc", "ps", "cp", "bs"])
@pytest.mark.parametrize("method", ["REML", "GCV.Cp"])
def test_all_bases_reml_and_gcv_vs_r(r_mgcv_suite, bs, method):
    """Verify fitted values, EDF, scale, and smoothing parameter sp against R mgcv."""
    x = r_mgcv_suite["x"]
    y = r_mgcv_suite["y"]
    r_res = r_mgcv_suite["r"]["fits"][f"{bs}_{method}"]

    py_model = LinGAM(k=8, bs=bs, method=method).fit(x, y)
    py_fitted = py_model.predict(x)
    r_fitted = np.array(r_res["fitted"])

    tol_fit = 3e-4 if bs == "cs" else 1e-4
    tol_edf = 2e-3 if bs == "cs" else 1e-3
    assert np.max(np.abs(py_fitted - r_fitted)) < tol_fit
    assert np.isclose(py_model.statistics_["edof"], r_res["edf"], atol=tol_edf)
    assert np.isclose(py_model.statistics_["scale"], r_res["scale"], rtol=1e-4)
    assert np.isclose(py_model.statistics_["sp"][0], r_res["sp"], rtol=5e-3)


def test_fixed_sp_vs_r(r_mgcv_suite):
    """Verify fixed smoothing parameter (sp=0.5) matches R mgcv to machine precision."""
    x = r_mgcv_suite["x"]
    y = r_mgcv_suite["y"]
    r_res = r_mgcv_suite["r"]["fixed"]

    py_model = LinGAM(k=10, bs="cr", sp=0.5).fit(x, y)
    py_fitted = py_model.predict(x)
    r_fitted = np.array(r_res["fitted"])

    assert np.max(np.abs(py_fitted - r_fitted)) < 1e-10
    assert np.isclose(py_model.statistics_["edof"], r_res["edf"], atol=1e-10)
    assert np.isclose(py_model.statistics_["scale"], r_res["scale"], rtol=1e-10)


def test_multiterm_reml_vs_r(r_mgcv_suite):
    """Verify multi-term additive REML fit and total null-space dimension M_p against R mgcv."""
    X = np.column_stack([r_mgcv_suite["x0"], r_mgcv_suite["x1"]])
    y = r_mgcv_suite["y_add"]
    r_res = r_mgcv_suite["r"]["add"]

    py_gam = GAMCore("s(0, bs='tp', k=8) + s(1, bs='cr', k=8)", method="REML").fit(X, y)
    py_fitted = py_gam.predict(X)
    r_fitted = np.array(r_res["fitted"])

    assert np.max(np.abs(py_fitted - r_fitted)) < 1e-4
    assert np.isclose(py_gam.statistics_["edof"], r_res["edf"], atol=1e-3)
    assert np.isclose(py_gam.statistics_["scale"], r_res["scale"], rtol=1e-4)
    assert np.allclose(py_gam.statistics_["sp"], r_res["sp"], rtol=2e-3)


def test_tensor_product_vs_r(r_mgcv_suite):
    """Verify tensor product te(x0, x1) against R mgcv."""
    X = np.column_stack([r_mgcv_suite["x0"], r_mgcv_suite["x1"]])
    y = r_mgcv_suite["y_2d"]
    r_res = r_mgcv_suite["r"]["te"]

    py_gam = GAMCore("te(0, 1, bs='cr', k=4)", method="REML").fit(X, y)
    py_fitted = py_gam.predict(X)
    r_fitted = np.array(r_res["fitted"])

    assert np.max(np.abs(py_fitted - r_fitted)) < 1e-6
    assert np.isclose(py_gam.statistics_["edof"], r_res["edf"], atol=1e-4)
    assert np.isclose(py_gam.statistics_["scale"], r_res["scale"], rtol=1e-5)
