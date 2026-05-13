import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from scTenifold.plotting._plotting import (
    plot_network_graph,
    plot_network_heatmap,
    plot_qqplot,
    plot_embedding,
    plot_hist,
)
from scTenifold.plotting._dim_reduction import prepare_PCA_dfs, prepare_embedding_dfs


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.random((30, 12)),
        index=[f"G{i}" for i in range(30)],
        columns=[f"C{i}" for i in range(12)],
    )


@pytest.fixture
def network():
    rng = np.random.default_rng(1)
    n = rng.random((20, 20))
    n = (n + n.T) / 2
    return n


def test_plot_network_heatmap(network):
    plot_network_heatmap(network, figsize=(4, 4))


def test_plot_network_graph(network):
    plot_network_graph(network, weight_thres=0.1, con_thres=0)


def test_plot_qqplot():
    rng = np.random.default_rng(2)
    n = 30
    df = pd.DataFrame({
        "FC": rng.normal(size=n),
        "adjusted p-value": rng.random(n),
    })
    plot_qqplot(df, plot_qqline=True, sig_threshold=0.5)
    plot_qqplot(df, plot_qqline=False)


def test_prepare_PCA_dfs(df):
    feat, var, comp = prepare_PCA_dfs(df, n_components=3)
    assert feat.shape == (12, 3)
    assert comp.shape == (30, 3)


def test_prepare_PCA_dfs_no_components(df):
    feat, var, comp = prepare_PCA_dfs(df, standardize=False)
    assert feat.shape[0] == 12


def test_prepare_embedding_tsne(df):
    out = prepare_embedding_dfs(df, reducer="TSNE", n_components=2, perplexity=3)
    assert out.shape == (12, 2)


def test_prepare_embedding_mds(df):
    out = prepare_embedding_dfs(df, reducer="MDS", n_components=2)
    assert out.shape == (12, 2)


def test_plot_embedding_pca(df):
    groups = {"g1": list(df.columns[:6]), "g2": list(df.columns[6:])}
    plot_embedding(df, groups=groups, method="PCA", plot_2D=True, title="t")


def test_plot_embedding_default_groups(df):
    plot_embedding(df, groups=None, method="PCA", plot_2D=True, n_components=3)


def test_plot_embedding_3d(df):
    groups = {"all": list(df.columns)}
    plot_embedding(df, groups=groups, method="PCA", plot_2D=False, n_components=3)


def test_plot_hist_single(df):
    plot_hist(df, df_1_name="A", sum_axis=0)


def test_plot_hist_two(df):
    df2 = df.copy() + 1
    plot_hist(df, df_1_name="A", df_2=df2, df_2_name="B", sum_axis=1)


def test_plot_hist_invalid_axis(df):
    with pytest.raises(ValueError):
        plot_hist(df, df_1_name="A", sum_axis=2)
