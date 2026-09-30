"""The knocked-out genes are left out of the expectation of d_regulation.

They move by construction and have the largest distances; left in the mean
squared distance used for the fold-changes, they hide every other gene
(cailab-tamu/scTenifoldKnk#45). Same as ``gKO`` in dRegulation of the R
package scTenifoldKnk (>= 1.1.5).
"""
import numpy as np
import pandas as pd
import pytest

from scTenifold import scTenifoldKnk
from scTenifold.core._networks import d_regulation


def _manifold(moved=(), n_genes=200, d=2, scale=None):
    """Manifold alignment output where only the ``moved`` genes change."""
    rng = np.random.default_rng(1)
    genes = [f"g{i}" for i in range(1, n_genes + 1)]
    X = rng.normal(size=(n_genes, d))
    Y = X + rng.normal(scale=1e-3, size=(n_genes, d))
    shift = {g: 1.0 for g in moved} if scale is None else scale
    for g, s in shift.items():
        Y[genes.index(g)] += s
    return pd.DataFrame(np.vstack([X, Y]), index=[f"X_{g}" for g in genes] + [f"Y_{g}" for g in genes])


def _expected_fc(dr, ko_genes):
    d = dr.set_index("Gene")["Distance"]
    return dr["Distance"].to_numpy() ** 2 / np.mean(d[~d.index.isin(ko_genes)] ** 2)


@pytest.mark.parametrize("ko_genes", [["g1"], ["g1", "g50", "g100"]])
def test_ko_genes_are_left_out_of_the_expectation(ko_genes):
    dr = d_regulation(_manifold(ko_genes), ko_genes=ko_genes)
    np.testing.assert_allclose(dr["FC"].to_numpy(), _expected_fc(dr, ko_genes))
    assert dr.loc[~dr["Gene"].isin(ko_genes), "FC"].mean() == pytest.approx(1)
    assert set(dr["Gene"].iloc[:len(ko_genes)]) == set(ko_genes)
    assert (dr.loc[dr["Gene"].isin(ko_genes), "adjusted p-value"] < 0.05).all()
    assert len(dr) == 200


def test_a_single_gene_name_is_accepted():
    ma = _manifold(["g1"])
    pd.testing.assert_frame_equal(d_regulation(ma, ko_genes="g1"), d_regulation(ma, ko_genes=["g1"]))


def test_ko_gene_in_the_expectation_hides_the_other_genes():
    # g2 moves a tenth of what g1 moves
    ma = _manifold(scale={"g1": 1.0, "g2": 0.1})
    with_ko = d_regulation(ma).set_index("Gene")
    without_ko = d_regulation(ma, ko_genes=["g1"]).set_index("Gene")
    assert with_ko.loc["g2", "adjusted p-value"] >= 0.05
    assert without_ko.loc["g2", "adjusted p-value"] < 0.05
    # Distances and Z do not depend on ko_genes
    pd.testing.assert_frame_equal(with_ko.sort_index()[["Distance", "Z"]],
                                  without_ko.sort_index()[["Distance", "Z"]])


def test_without_ko_genes_the_expectation_uses_all_genes():
    dr = d_regulation(_manifold(["g1"]))
    np.testing.assert_allclose(dr["FC"].to_numpy(), _expected_fc(dr, []))


def test_n_ko_genes_still_leaves_out_the_largest_distances():
    ma = _manifold(["g1", "g2"])
    pd.testing.assert_frame_equal(d_regulation(ma, n_ko_genes=2), d_regulation(ma, ko_genes=["g1", "g2"]))


def test_invalid_ko_genes_are_rejected():
    ma = _manifold(["g1"], n_genes=5)
    with pytest.raises(ValueError, match="not present in the manifold alignment"):
        d_regulation(ma, ko_genes=["absent"])
    with pytest.raises(ValueError, match="At least one gene"):
        d_regulation(ma, ko_genes=[f"g{i}" for i in range(1, 6)])


def _counts(n_genes=60, n_cells=600):
    """Counts with gene modules so that the networks have strong edges."""
    rng = np.random.default_rng(1)
    modules = rng.gamma(2, size=(3, n_cells))
    loadings = np.zeros((n_genes, 3))
    loadings[np.arange(n_genes), np.arange(n_genes) % 3] = rng.uniform(0.5, 2, n_genes)
    return pd.DataFrame(rng.poisson(5 * loadings @ modules),
                        index=[f"g{i}" for i in range(1, n_genes + 1)],
                        columns=[f"c{j}" for j in range(n_cells)])


@pytest.fixture(scope="module")
def counts():
    return _counts()


def _knk(counts, ko_genes, **kws):
    sc = scTenifoldKnk(counts, ko_genes=ko_genes, qc_kws={"min_lib_size": 0},
                       nc_kws={"n_nets": 3, "n_samp_cells": 300}, **kws)
    sc.build()
    return sc


@pytest.mark.parametrize("ko_genes", ["g1", ["g1"], ["g1", "g2", "g4"]])
def test_knk_leaves_the_ko_genes_out_of_the_expectation(counts, ko_genes):
    sc = _knk(counts, ko_genes)
    names = [ko_genes] if isinstance(ko_genes, str) else ko_genes
    dr = sc.d_regulation
    assert dr.loc[~dr["Gene"].isin(names), "FC"].mean() == pytest.approx(1)
    pd.testing.assert_frame_equal(dr, d_regulation(sc.manifold, ko_genes=names))


def test_knk_uses_the_genes_of_the_ko_step(counts):
    sc = scTenifoldKnk(counts, ko_genes=["g1"], qc_kws={"min_lib_size": 0},
                       nc_kws={"n_nets": 3, "n_samp_cells": 300})
    for step in ["qc", "nc", "td"]:
        sc.run_step(step)
    sc.run_step("ko", ko_genes=["g2"])
    sc.run_step("ma")
    sc.run_step("dr")
    pd.testing.assert_frame_equal(sc.d_regulation, d_regulation(sc.manifold, ko_genes=["g2"]))


def test_knk_respects_n_ko_genes_in_dr_kws(counts):
    sc = _knk(counts, ["g1"], dr_kws={"n_ko_genes": 0})
    pd.testing.assert_frame_equal(sc.d_regulation, d_regulation(sc.manifold))


def test_loaded_knk_uses_the_genes_of_the_ko_step(counts, tmp_path):
    sc = scTenifoldKnk(counts, ko_genes=["g1"], qc_kws={"min_lib_size": 0},
                       nc_kws={"n_nets": 3, "n_samp_cells": 300})
    for step in ["qc", "nc", "td"]:
        sc.run_step(step)
    sc.run_step("ko", ko_genes=["g2"])
    sc.run_step("ma")
    sc.save(tmp_path / "knk", verbose=False)
    loaded = scTenifoldKnk.load(tmp_path / "knk")
    loaded.run_step("dr")
    pd.testing.assert_frame_equal(loaded.d_regulation, d_regulation(sc.manifold, ko_genes=["g2"]),
                                  check_exact=False, rtol=1e-12)
