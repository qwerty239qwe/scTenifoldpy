"""Heat-kernel tools and predicted direction (scTenifoldKnk >= 2.0.0 in R).

The parity references are written by fixtures/r_reference/generate_heat.R.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scTenifold import heat_kernel, hk_manifold_alignment, knockout_direction, scTenifoldKnk
from scTenifold.core._QC import sc_QC

FIXTURES = Path(__file__).parent / "fixtures" / "r_reference"


def _read(name):
    return pd.read_csv(FIXTURES / name, index_col=0)


def _simulate_modules(n_genes=60, n_cells=600, seed=1):
    # Counts with three gene modules (as in the R tests)
    rng = np.random.default_rng(seed)
    modules = rng.gamma(2, size=(3, n_cells))
    loadings = np.zeros((n_genes, 3))
    loadings[np.arange(n_genes), np.arange(n_genes) % 3] = rng.uniform(0.5, 2, n_genes)
    X = rng.poisson(5 * loadings @ modules)
    return pd.DataFrame(X, index=[f"g{i + 1}" for i in range(n_genes)],
                        columns=[f"c{i + 1}" for i in range(n_cells)])


def _module(genes):
    return np.array([(int(g[1:]) - 1) % 3 for g in genes])


def test_heat_kernel_identity_symmetry_and_eigendecomposition():
    rng = np.random.default_rng(1)
    A = pd.DataFrame(rng.normal(size=(6, 6)), index=list("abcdef"), columns=list("abcdef"))
    np.testing.assert_allclose(heat_kernel(A, t=0).to_numpy(), np.eye(6), atol=1e-12)
    H = heat_kernel(A, t=3)
    np.testing.assert_allclose(H.to_numpy(), H.to_numpy().T, atol=1e-12)
    assert list(H.index) == list(A.index)
    with pytest.raises(ValueError):
        heat_kernel(A, t=-1)
    with pytest.raises(ValueError):
        heat_kernel(A, symmetric=False)


def test_knockout_direction_module_goes_down():
    X = _simulate_modules()
    D = knockout_direction(X, "g1")
    assert D.shape == (1, X.shape[0])
    m = _module(X.index)
    same = [g for g, k in zip(X.index, m) if k == 0 and g != "g1"]
    assert (D.loc["g1", same] < 0).all()
    assert list(knockout_direction(X, [["g1", "g4"]]).index) == ["g1+g4"]


def test_regressing_library_size_removes_the_depth_axis():
    X = _simulate_modules()
    rng = np.random.default_rng(2)
    depth = np.exp(rng.normal(0, 0.8, X.shape[1]))
    Xd = pd.DataFrame(rng.poisson(0.3 * (X.to_numpy() + 0.5) * depth), index=X.index, columns=X.columns)
    m = _module(X.index)
    same = [g for g, k in zip(X.index, m) if k == 0 and g != "g1"]
    other = [g for g, k in zip(X.index, m) if k != 0]
    # without the regression (default) the depth axis dominates and every gene is predicted down
    D0 = knockout_direction(Xd, "g1")
    assert (D0.drop(columns="g1").to_numpy() < 0).all()
    # with it only the module of the knocked-out gene goes down
    D = knockout_direction(Xd, "g1", regress_lib_size=True)
    assert (D.loc["g1", same] < 0).all()
    assert (D.loc["g1", other] > 0).mean() > 0.8
    # the pipeline exposes the option
    def run(**kws):
        return scTenifoldKnk(Xd, ko_genes=["g1"], qc_kws={"min_lib_size": 0},
                             nc_kws={"n_nets": 3, "n_samp_cells": 300}, **kws).build()
    assert (run()["direction"] == "down").all()
    assert (run(dr_direction_regress_lib_size=True)["direction"] == "up").mean() > 0.5


# Parity with R ---------------------------------------------------------------

@pytest.fixture(scope="module")
def qc_counts():
    return sc_QC(_read("X_counts.csv.gz").astype(float), min_lib_size=30)


@pytest.fixture(scope="module")
def wt():
    return _read("knk_tensor_WT.csv.gz")


def test_fixture_versions():
    assert "scTenifoldKnk 2.0.0" in (FIXTURES / "heat_versions.txt").read_text()


def test_heat_kernel_matches_r(wt):
    np.testing.assert_allclose(heat_kernel(wt, t=10).to_numpy(), _read("heat_kernel_WT.csv.gz").to_numpy(),
                               rtol=0, atol=1e-10)


def test_hk_manifold_alignment_matches_r(wt, qc_counts):
    D = hk_manifold_alignment(wt, qc_counts, ko_genes=["ng10", ["ng10", "ng20"]], t=10)
    R = _read("heat_hk_distances.csv.gz")
    assert list(D.index) == list(R.index) and list(D.columns) == list(R.columns)
    np.testing.assert_allclose(D.to_numpy(), R.to_numpy(), rtol=1e-8, atol=1e-12)


def test_knockout_direction_matches_r(wt, qc_counts):
    R = _read("heat_direction.csv.gz")
    kos = ["ng10", ["ng10", "ng20"]]
    D1 = knockout_direction(qc_counts, kos, genes=wt.index, regress_lib_size=True)
    D0 = knockout_direction(qc_counts, kos, genes=wt.index)
    np.testing.assert_allclose(D1.to_numpy(), R.loc[["ng10", "ng10+ng20"]].to_numpy(), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(D0.to_numpy(), R.loc[["ng10_noregress", "ng10+ng20_noregress"]].to_numpy(),
                               rtol=1e-6, atol=1e-10)


def _assert_direction_columns(actual, expected):
    a = actual.set_index("Gene").loc[expected["gene"]]
    assert list(a["direction"]) == list(expected["direction"])
    np.testing.assert_allclose(a["direction score"].to_numpy(dtype=float),
                               expected["directionScore"].to_numpy(dtype=float), rtol=1e-6, atol=1e-10)


def test_pipeline_direction_matches_r():
    sc = scTenifoldKnk(_read("X_counts.csv.gz").astype(float), ko_genes=["ng10"], qc_kws={"min_lib_size": 30})
    dr = sc.build()
    assert list(dr.columns[-2:]) == ["direction", "direction score"]
    _assert_direction_columns(dr, _read("knk_direction_dregulation.csv.gz").reset_index())
    # dr_direction=False restores the previous output
    sc0 = scTenifoldKnk(_read("X_counts.csv.gz").astype(float), ko_genes=["ng10"], qc_kws={"min_lib_size": 30},
                        dr_direction=False)
    assert "direction" not in sc0.build().columns


def test_pipeline_heat_alignment_matches_r():
    sc = scTenifoldKnk(_read("X_counts.csv.gz").astype(float), ko_genes=["ng10"], qc_kws={"min_lib_size": 30},
                       ma_method="heat")
    dr = sc.build()
    R = _read("knk_heat_dregulation.csv.gz").reset_index()
    assert list(dr["Gene"]) == list(R["gene"])
    for py_col, r_col in [("Distance", "distance"), ("FC", "FC"), ("p-value", "p.value"),
                          ("adjusted p-value", "p.adj")]:
        np.testing.assert_allclose(dr[py_col].to_numpy(), R[r_col].to_numpy(), rtol=1e-8, atol=1e-12)
    _assert_direction_columns(dr, R)


def test_transcriptome_wide_matches_r():
    sc = scTenifoldKnk(_read("X_counts.csv.gz").astype(float), qc_kws={"min_lib_size": 30})
    for step in ("qc", "nc", "td"):
        sc.run_step(step)
    out = sc.transcriptome_wide(["ng10", "ng20"])
    np.testing.assert_allclose(out["distances"].to_numpy(), _read("tw_distances.csv.gz").to_numpy(),
                               rtol=1e-8, atol=1e-12)
    np.testing.assert_array_equal(out["directions"].to_numpy(), _read("tw_directions.csv.gz").to_numpy())
