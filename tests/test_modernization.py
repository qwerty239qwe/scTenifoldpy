from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from typer.testing import CliRunner

import scTenifold
from scTenifold import compare_networks, virtual_knockout
from scTenifold.__main__ import app
from scTenifold.core._networks import anndata_to_dataframe, d_regulation, make_networks, manifold_alignment
from scTenifold.core._norm import cpm_norm
from scTenifold.data import TestDataGenerator as _TestDataGenerator, fetch_data, get_test_df


def test_version_metadata():
    assert scTenifold.__version__ == "0.2.0"


@pytest.mark.parametrize("backend", ["serial", "joblib-loky", "joblib-threading"])
def test_make_networks_backends(backend):
    df = get_test_df(n_genes=12, n_cells=16, random_state=42)
    networks = make_networks(
        df,
        n_nets=2,
        n_samp_cells=12,
        n_comp=3,
        q=0,
        backend=backend,
        n_jobs=2,
        verbosity=0,
    )
    assert len(networks) == 2
    assert networks[0].shape == (12, 12)


def test_n_cpus_alias_warns():
    df = get_test_df(n_genes=12, n_cells=16, random_state=42)
    with pytest.warns(DeprecationWarning, match="n_cpus is deprecated"):
        make_networks(df, n_nets=1, n_samp_cells=12, n_comp=3, q=0, n_cpus=1, verbosity=0)


def test_anndata_like_dense_and_sparse_conversion():
    dense = np.arange(12).reshape(3, 4)
    adata = SimpleNamespace(
        X=dense,
        layers={"counts": sparse.csr_matrix(dense + 1)},
        obs_names=["c1", "c2", "c3"],
        var_names=["g1", "g2", "g3", "g4"],
    )
    x_df = anndata_to_dataframe(adata)
    layer_df = anndata_to_dataframe(adata, layer="counts")
    assert list(x_df.index) == ["g1", "g2", "g3", "g4"]
    assert list(x_df.columns) == ["c1", "c2", "c3"]
    assert layer_df.loc["g1", "c1"] == 1


def test_high_level_apis_return_dataframes():
    x_df = get_test_df(n_genes=24, n_cells=28, random_state=1)
    y_df = get_test_df(n_genes=24, n_cells=28, random_state=2)
    result = compare_networks(
        x_df,
        y_df,
        qc_kws={"min_lib_size": 1, "min_percent": 0, "max_mito_ratio": 1, "min_exp_avg": 0, "min_exp_sum": 0, "remove_outlier_cells": False, "plot": False},
        network_kws={"n_nets": 1, "n_samp_cells": 20, "n_comp": 3, "q": 0},
        td_kws={"K": 2, "max_iter": 50},
        ma_kws={"d": 2},
        backend="serial",
    )
    assert isinstance(result, pd.DataFrame)

    knockout = virtual_knockout(
        x_df,
        ko_genes=[x_df.index[0]],
        qc_kws={"min_lib_size": 1, "min_percent": 0, "max_mito_ratio": 1, "min_exp_avg": 0, "min_exp_sum": 0, "remove_outlier_cells": False, "plot": False},
        network_kws={"n_nets": 1, "n_samp_cells": 20, "n_comp": 3, "q": 0},
        td_kws={"K": 2, "max_iter": 50},
        backend="serial",
    )
    assert isinstance(knockout, pd.DataFrame)


def test_shared_gene_order_is_deterministic():
    x = pd.DataFrame(np.eye(4), index=["g1", "g2", "g3", "g4"], columns=["g1", "g2", "g3", "g4"])
    y = pd.DataFrame(np.eye(4), index=["g3", "g2", "g1", "g5"], columns=["g3", "g2", "g1", "g5"])
    aligned = manifold_alignment(x, y, d=1, verbosity=0)
    assert list(aligned.index[:3]) == ["X_g1", "X_g2", "X_g3"]
    assert list(aligned.index[3:]) == ["Y_g1", "Y_g2", "Y_g3"]
    assert np.isrealobj(aligned.to_numpy())


def test_cli_config_uses_requested_path(tmp_path):
    runner = CliRunner()
    net_config = tmp_path / "net_config.yml"
    result = runner.invoke(app, ["config", "-t", "1", "-p", str(net_config)])
    assert result.exit_code == 0
    assert net_config.is_file()


def test_save_load_roundtrip(tmp_path):
    from scTenifold import scTenifoldNet

    x_df = get_test_df(n_genes=20, n_cells=24, random_state=3)
    y_df = get_test_df(n_genes=20, n_cells=24, random_state=4)
    sc = scTenifoldNet(
        x_df,
        y_df,
        "X",
        "Y",
        qc_kws={"min_lib_size": 1, "min_percent": 0, "max_mito_ratio": 1, "remove_outlier_cells": False, "plot": False},
        nc_kws={"n_nets": 1, "n_samp_cells": 18, "n_comp": 3, "q": 0, "backend": "serial"},
        td_kws={"K": 2, "max_iter": 50},
        ma_kws={"d": 2},
    )
    sc.build()
    out_dir = tmp_path / "saved_net"
    sc.save(out_dir, verbose=False)
    loaded = scTenifoldNet.load(out_dir)
    assert loaded.x_label == "X"
    assert "X" in loaded.network_dict
    assert isinstance(loaded.QC_dict["X"], pd.DataFrame)
    assert loaded.QC_dict["X"].index[0] == sc.QC_dict["X"].index[0]
    assert isinstance(loaded.manifold, pd.DataFrame)
    assert isinstance(loaded.d_regulation, pd.DataFrame)
    assert isinstance(loaded.step_comps["ma"], pd.DataFrame)

    resaved_dir = tmp_path / "resaved_net"
    loaded.save(resaved_dir, verbose=False)
    assert (resaved_dir / "ma" / "manifold_alignment.csv").is_file()
    assert (resaved_dir / "dr" / "d_regulation.csv").is_file()


def test_invalid_ko_method_raises():
    from scTenifold import scTenifoldKnk

    sc = scTenifoldKnk(get_test_df(n_genes=12, n_cells=16, random_state=5), ko_method="invalid")
    with pytest.raises(ValueError, match="No such method"):
        sc._get_ko_tensor([])


def test_d_regulation_handles_zero_distances():
    data = pd.DataFrame(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [3.0, 3.0]],
        index=["X_g1", "X_g2", "Y_g1", "Y_g2"],
    )
    result = d_regulation(data, verbosity=0)
    assert result.shape[0] == 2
    assert result["boxcox-transformed distance"].notna().all()


def test_propagation_ko_defaults_degree():
    from scTenifold import scTenifoldKnk

    sc = scTenifoldKnk(
        get_test_df(n_genes=12, n_cells=16, random_state=6),
        ko_method="propagation",
        nc_kws={"n_samp_cells": 12, "n_comp": 3, "q": 0},
        td_kws={"K": 2, "max_iter": 50},
    )
    sc.QC_dict["WT"] = get_test_df(n_genes=12, n_cells=16, random_state=6)
    sc.shared_gene_names = sc.QC_dict["WT"].index.to_list()
    sc.network_dict["WT"] = make_networks(sc.QC_dict["WT"], n_nets=1, n_samp_cells=12, n_comp=3, q=0, verbosity=0)
    sc.tensor_dict["WT"] = pd.DataFrame(np.eye(12), index=sc.shared_gene_names, columns=sc.shared_gene_names)
    sc._get_ko_tensor([sc.shared_gene_names[0]])
    assert "KO" in sc.tensor_dict


def test_plot_embedding_standardize_false():
    from scTenifold.plotting._dim_reduction import prepare_embedding_dfs

    df = pd.DataFrame(np.arange(30).reshape(5, 6), index=[f"g{i}" for i in range(5)])
    result = prepare_embedding_dfs(df, reducer="MDS", standardize=False, random_state=1)
    assert result.shape == (6, 2)


def test_qc_does_not_mutate_input():
    from scTenifold.core._QC import sc_QC

    df = pd.DataFrame([[-1, 2], [3, 4]], index=["g1", "g2"])
    original = df.copy()
    sc_QC(df, min_lib_size=0, min_percent=0, max_mito_ratio=1)
    pd.testing.assert_frame_equal(df, original)


def test_sim_helpers_small_and_empty_negative_targets():
    df = get_test_df(n_genes=5, n_cells=3, random_state=1)
    assert df.shape == (5, 3)
    generator = _TestDataGenerator(n_genes=8, n_samples=10, target_neg=[], random_state=1)
    assert generator.X.shape == (8, 10)


def test_cpm_norm_zero_library_size():
    df = pd.DataFrame([[0, 1], [0, 2]], index=["g1", "g2"])
    result = cpm_norm(df)
    assert result.iloc[:, 0].eq(0).all()
    assert np.isfinite(result.to_numpy()).all()


def test_fetch_data_validates_dataset_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_data("missing", dataset_path=tmp_path)
