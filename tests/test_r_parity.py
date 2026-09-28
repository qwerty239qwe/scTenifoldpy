"""Checks that the steps give the same results as the R packages.

The reference values were computed with R 4.5 (scTenifoldNet 1.4.3,
scTenifoldKnk 1.1.4, MASS 7.3-65).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scTenifold import scTenifoldKnk, scTenifoldNet
from scTenifold.core._QC import _fivenum, sc_QC
from scTenifold.core._decomposition import cp_decomposition
from scTenifold.core._networks import _boxcox_lambda, d_regulation, make_networks, pc_net
from scTenifold.core._rng import RRandom


def test_rng_matches_r():
    # set.seed(1); runif(3)
    np.testing.assert_array_equal(RRandom(1).unif_rand(3),
                                  [0.26550866314209998, 0.37212389963679016, 0.57285336335189641])
    # set.seed(42); rnorm(3)
    np.testing.assert_allclose(RRandom(42).rnorm(3),
                               [1.3709584471466685, -0.56469817139608869, 0.3631284113373392],
                               rtol=0, atol=1e-15)
    # set.seed(1); sample(1:1000, 8, replace = TRUE)
    np.testing.assert_array_equal(RRandom(1).sample(1000, 8) + 1, [836, 679, 129, 930, 509, 471, 299, 270])
    # set.seed(1); sample(1:100, 5) and sample(1:10, 8)
    np.testing.assert_array_equal(RRandom(1).sample(100, 5, replace=False) + 1, [68, 39, 1, 34, 87])
    np.testing.assert_array_equal(RRandom(1).sample(10, 8, replace=False) + 1, [9, 4, 7, 1, 2, 5, 3, 10])
    # set.seed(1); sample(1:20, 10) and set.seed(42); sample(1:6, 3): size <= n / 2 with
    # repeated indices, where R's partial shuffle and the sample2() hash path disagree
    np.testing.assert_array_equal(RRandom(1).sample(20, 10, replace=False) + 1,
                                  [4, 7, 1, 2, 13, 19, 11, 17, 14, 3])
    np.testing.assert_array_equal(RRandom(42).sample(6, 3, replace=False) + 1, [1, 5, 6])


def test_outlier_cells_use_boxplot_stats_hinges():
    lib_sizes = np.array([5, 1, 9, 30, 2, 7, 8, 3, 100, 6], dtype=float)
    np.testing.assert_array_equal(_fivenum(lib_sizes), [1, 3, 6.5, 9, 100])
    # boxplot.stats(x)$out is c(30, 100)
    X = pd.DataFrame([lib_sizes], index=["g1"], columns=[f"c{i}" for i in range(10)])
    kept = sc_QC(X, min_lib_size=0, min_percent=0, max_mito_ratio=1)
    assert sorted(X.columns.difference(kept.columns)) == ["c3", "c8"]


def test_boxcox_lambda_matches_mass():
    y = np.array([0.3, 1.2, 0.05, 2.5, 0.8, 0.4, 1.9])
    assert _boxcox_lambda(y) == pytest.approx(0.3583583583583585, abs=1e-15)


def _pc_net_by_svd(X, n_comp):
    # Reference: one exact SVD per gene
    Xs = X.T.astype(float)
    Xs = (Xs - Xs.mean(axis=0)) / Xs.std(axis=0, ddof=1)
    n = Xs.shape[1]
    out = np.zeros((n, n))
    for k in range(n):
        others = np.delete(np.arange(n), k)
        U, s, VT = np.linalg.svd(Xs[:, others], full_matrices=False)
        out[k, others] = VT[:n_comp].T @ ((U[:, :n_comp].T @ Xs[:, k]) / s[:n_comp])
    return out


@pytest.mark.parametrize("shape", [(40, 25), (15, 80)])
def test_pc_net_is_exact(shape):
    X = np.random.default_rng(0).poisson(3, size=shape)
    expected = _pc_net_by_svd(X, 3)
    np.testing.assert_allclose(pc_net(X, 3, scale_scores=False), expected, atol=1e-10)


def test_pc_net_constant_genes_have_no_edges():
    X = np.random.default_rng(1).poisson(3, size=(20, 30)).astype(float)
    X[4] = 0
    X[7] = 5
    net = pc_net(X, 3, scale_scores=False)
    assert not net[[4, 7]].any() and not net[:, [4, 7]].any()
    keep = np.setdiff1d(np.arange(20), [4, 7])
    np.testing.assert_allclose(net[np.ix_(keep, keep)], pc_net(X[keep], 3, scale_scores=False), atol=1e-12)


def test_pc_net_prior_network_keeps_edges():
    X = np.random.default_rng(2).poisson(3, size=(20, 60))
    genes = [f"g{i}" for i in range(20)]
    unfiltered = pc_net(X, 3)
    weakest = np.unravel_index(np.argmin(np.abs(unfiltered) + np.eye(20)), unfiltered.shape)
    prior = pd.DataFrame({"regulator": [genes[weakest[0]]], "target": [genes[weakest[1]]]})
    filtered = pc_net(X, 3, q=0.9)
    with_prior = pc_net(X, 3, q=0.9, prior_network=prior, gene_names=genes)
    assert filtered[weakest] == 0
    assert with_prior[weakest] == unfiltered[weakest]


def test_make_networks_is_reproducible_across_backends():
    X = pd.DataFrame(np.random.default_rng(3).poisson(2, size=(15, 40)),
                     index=[f"g{i}" for i in range(15)])
    serial = make_networks(X, n_nets=2, n_samp_cells=30, q=0.9, verbosity=0)
    threaded = make_networks(X, n_nets=2, n_samp_cells=30, q=0.9, backend="joblib-threading",
                             n_jobs=2, verbosity=0)
    for a, b in zip(serial, threaded):
        np.testing.assert_array_equal(a.toarray(), b.toarray())


def test_cp_decomposition_recovers_low_rank_tensor():
    rng = np.random.default_rng(4)
    A, B, C = rng.normal(size=(12, 2)), rng.normal(size=(12, 2)), rng.normal(size=(5, 2))
    tensor = np.einsum("ir,jr,kr->ijk", A, B, C)
    result = cp_decomposition(tensor, K=2, max_iter=500, tol=1e-12)
    np.testing.assert_allclose(result["slice_sum"], tensor.sum(axis=-1), atol=1e-6)


def test_d_regulation_sets_noise_level_distances_to_one():
    genes = [f"g{i}" for i in range(6)]
    x = np.random.default_rng(5).normal(size=(6, 2))
    y = x.copy()
    y[0] += 1.0
    y[1] += 1e-17  # floating point noise
    data = pd.DataFrame(np.vstack([x, y]), index=[f"X_{g}" for g in genes] + [f"Y_{g}" for g in genes])
    result = d_regulation(data, verbosity=0).set_index("Gene")
    assert (result.loc[genes[1:], "p-value"] == 1).all()
    assert result.loc["g0", "p-value"] < 1
    assert result.index[0] == "g0"


def test_d_regulation_warns_when_nothing_moves():
    data = pd.DataFrame(np.ones((4, 2)), index=["X_a", "X_b", "Y_a", "Y_b"])
    with pytest.warns(UserWarning, match="beyond numerical noise"):
        result = d_regulation(data, verbosity=0)
    assert (result["p-value"] == 1).all()


def test_pipeline_defaults_match_r():
    assert scTenifoldNet.list_kws("td_kws")["K"] == 3
    assert scTenifoldNet.list_kws("td_kws")["n_decimal"] == 1
    assert scTenifoldNet.list_kws("nc_kws")["q"] == 0.95
    assert scTenifoldNet.list_kws("ma_kws")["d"] == 30
    assert scTenifoldKnk.list_kws("td_kws")["n_decimal"] == 3
    assert scTenifoldKnk.list_kws("nc_kws")["q"] == 0.9
    assert scTenifoldKnk.list_kws("ma_kws")["d"] == 2
    # User values override the defaults, which are kept for the other keys
    sc = scTenifoldKnk(pd.DataFrame(), td_kws={"max_iter": 10})
    assert sc.td_kws == {"K": 3, "n_decimal": 3, "max_iter": 10}


def test_d_regulation_negative_power_with_zero_distances():
    data = pd.DataFrame([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2], [0.0, 0.0], [3.0, 3.0], [0.1, 0.9]],
                        index=["X_g1", "X_g2", "X_g3", "Y_g1", "Y_g2", "Y_g3"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = d_regulation(data, boxcox_kws={"lmbda": -0.5}, verbosity=0).set_index("Gene")
    # 1 / (0 ^ -0.5) is 0, as in R
    assert result.loc["g1", "boxcox-transformed distance"] == 0
    assert np.isfinite(result[["boxcox-transformed distance", "Z", "FC", "p-value"]].to_numpy()).all()


@pytest.mark.parametrize("cls, name", [(scTenifoldNet, "net_config.yml"), (scTenifoldKnk, "knk_config.yml")])
def test_shipped_configs_use_pipeline_defaults(cls, name):
    config = yaml.safe_load((Path(__file__).parents[1] / "config" / name).read_text())
    defaults = cls.get_empty_config()
    for step in ["nc_kws", "td_kws", "ma_kws"]:
        for key, value in config[step].items():
            if key in defaults[step] and key != "n_cpus":
                assert value == defaults[step][key], (step, key)
