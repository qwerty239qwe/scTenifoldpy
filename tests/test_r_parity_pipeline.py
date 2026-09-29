"""Checks the whole pipelines against outputs of the R packages.

The reference files in fixtures/r_reference are written by
fixtures/r_reference/generate.R; versions.txt records the R versions used.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scTenifold import scTenifoldKnk, scTenifoldNet
from scTenifold.core._QC import sc_QC

FIXTURES = Path(__file__).parent / "fixtures" / "r_reference"


def _read(name):
    return pd.read_csv(FIXTURES / name, index_col=0)


def _counts(name):
    return _read(name).astype(float)


def _assert_manifold_close(actual, expected):
    # Each column of the alignment is an eigenvector, defined up to its sign
    assert list(actual.index) == [i.replace("y_", "Y_", 1) for i in expected.index]
    actual, expected = actual.to_numpy(), expected.to_numpy()
    assert actual.shape == expected.shape
    signs = np.sign((actual * expected).sum(axis=0))
    np.testing.assert_allclose(actual * signs, expected, rtol=0, atol=1e-8)


def _assert_dregulation_close(actual, expected, z_atol=1e-7):
    assert list(actual["Gene"]) == list(expected["gene"])
    for py_col, r_col in [("Distance", "distance"), ("FC", "FC"),
                          ("p-value", "p.value"), ("adjusted p-value", "p.adj")]:
        np.testing.assert_allclose(actual[py_col].to_numpy(), expected[r_col].to_numpy(),
                                   rtol=1e-8, atol=1e-12, err_msg=py_col)
    # The Z-scores of genes at noise level (p = 1) are not meaningful
    moved = expected["p.value"].to_numpy() < 1
    np.testing.assert_allclose(actual["Z"].to_numpy()[moved], expected["Z"].to_numpy()[moved],
                               rtol=0, atol=z_atol)


def test_fixture_versions():
    versions = (FIXTURES / "versions.txt").read_text()
    assert "scTenifoldNet 1.4.3" in versions
    assert "scTenifoldKnk 1.1.5" in versions


def test_qc_keeps_the_same_cells():
    with gzip.open(FIXTURES / "X_qc_cells.txt.gz", "rt") as f:
        expected = f.read().split()
    kept = sc_QC(_counts("X_counts.csv.gz"), min_lib_size=30)
    assert list(kept.columns) == expected


@pytest.fixture(scope="module")
def net():
    sc = scTenifoldNet(_counts("X_counts.csv.gz"), _counts("Y_counts.csv.gz"), "X", "Y",
                       qc_kws={"min_lib_size": 30})
    sc.build()
    return sc


@pytest.fixture(scope="module")
def knk():
    sc = scTenifoldKnk(_counts("X_counts.csv.gz"), ko_genes=["ng10"], qc_kws={"min_lib_size": 30})
    sc.build()
    return sc


@pytest.fixture(scope="module")
def knk_multi():
    sc = scTenifoldKnk(_counts("X_counts.csv.gz"), ko_genes=["ng10", "ng20"], qc_kws={"min_lib_size": 30})
    sc.build()
    return sc


@pytest.mark.parametrize("label", ["X", "Y"])
def test_net_tensor_networks(net, label):
    expected = _read(f"net_tensor_{label}.csv.gz")
    # R returns the networks without self-loops
    actual = net.tensor_dict[label].loc[expected.index, expected.columns].to_numpy().copy()
    np.fill_diagonal(actual, 0)
    np.testing.assert_allclose(actual, expected.to_numpy(), rtol=0, atol=1e-12)


def test_net_manifold_alignment(net):
    _assert_manifold_close(net.manifold, _read("net_manifold.csv.gz"))


def test_net_differential_regulation(net):
    expected = pd.read_csv(FIXTURES / "net_dregulation.csv.gz")
    # Two genes have distances at the level of floating-point noise (p = 1 in
    # both implementations). They change with the last bits of the
    # arithmetic, even between runs of the same code, and so do the Box-Cox
    # power and the mean and SD used for every Z-score; the p-values do not
    # depend on them. A tenfold change of those distances moves the other
    # Z-scores by about 0.025
    assert (expected["p.value"] == 1).sum() == 2
    _assert_dregulation_close(net.d_regulation, expected, z_atol=0.05)


def test_knk_tensor_network(knk):
    expected = _read("knk_tensor_WT.csv.gz")
    actual = knk.tensor_dict["WT"].loc[expected.index, expected.columns]
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)


def test_knk_manifold_alignment(knk):
    _assert_manifold_close(knk.manifold, _read("knk_manifold.csv.gz"))


def test_knk_differential_regulation(knk):
    _assert_dregulation_close(knk.d_regulation, pd.read_csv(FIXTURES / "knk_dregulation.csv.gz"))


def test_knk_multi_manifold_alignment(knk_multi):
    _assert_manifold_close(knk_multi.manifold, _read("knk_multi_manifold.csv.gz"))


def test_knk_multi_differential_regulation(knk_multi):
    _assert_dregulation_close(knk_multi.d_regulation, pd.read_csv(FIXTURES / "knk_multi_dregulation.csv.gz"))
