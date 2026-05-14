import numpy as np
import pandas as pd
import pytest

from scTenifold.cell_cycle.UCell import (
    _check_features,
    calc_auc,
    calc_U_stat_df,
    cal_Uscore,
)
from scTenifold.cell_cycle.scoring import (
    adobo_score,
    cell_cycle_score,
    _get_assigned_bins,
    _get_ctrl_use,
)
from scTenifold.data._sim import TestDataGenerator, DEFAULT_POS, DEFAULT_NEG


@pytest.fixture
def gen():
    return TestDataGenerator(n_genes=200, n_samples=40, neg_eff_ratio=0.1,
                            target_neg=DEFAULT_NEG, n_bins=10, n_ctrl=10, random_state=0)


def test_check_features_warns_on_invalid():
    df = pd.DataFrame(np.zeros((3, 2)), index=["A", "B", "C"])
    with pytest.warns(UserWarning, match="invalid features"):
        valid = _check_features(df, ["A", "Z"])
    assert valid == {"A"}


def test_calc_auc_all_insignificant():
    s = pd.Series([100, 200, 300])
    assert calc_auc(s, max_rank=10) == 0


def test_calc_auc_partial():
    s = pd.Series([1.0, 2.0, 50.0])
    auc = calc_auc(s.copy(), max_rank=5)
    assert 0 < auc <= 1


def test_calc_U_stat_df_pos_and_neg():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((10, 4)), index=[f"G{i}" for i in range(10)])
    ranked = df.rank(ascending=False)
    out = calc_U_stat_df(features=["G0", "G1", "G2"],
                        df=ranked,
                        neg_features=["G3"],
                        max_rank=5)
    assert out.shape == (4,)


def test_cal_Uscore(gen):
    X = pd.DataFrame(gen.n_X, index=gen.gene_list, columns=gen.samples)
    out = cal_Uscore(X, pos_genes=DEFAULT_POS, neg_genes=DEFAULT_NEG, max_rank=50)
    assert out.shape == (40, 1)


def test_get_assigned_bins():
    data_avg = np.sort(np.linspace(0, 1, 20))
    bins = _get_assigned_bins(data_avg, cluster_len=20, n_bins=5)
    assert bins.shape == (20,)


def test_get_ctrl_use():
    rng = np.random.default_rng(1)
    gene_arr = np.array([f"G{i}" for i in range(20)])
    bins = np.array([i % 4 for i in range(20)])
    out = _get_ctrl_use(bins, gene_arr, {"Pos": ["G1", "G2"]}, n_ctrl=3, random_state=rng)
    assert len(out) > 0


def test_cell_cycle_score_default(gen, tmp_path):
    kwargs = gen.get_data("numpy", True)
    out_file = tmp_path / "scores.csv"
    kwargs["file_path"] = out_file
    scores = cell_cycle_score(**kwargs)
    assert scores.shape == (gen.n_samples,)
    assert out_file.is_file()


def test_cell_cycle_score_custom_target(gen):
    kwargs = gen.get_data("numpy", True)
    kwargs["target_dict"] = {"Pos": [g.lower() for g in DEFAULT_POS], "Neg": []}
    scores = cell_cycle_score(**kwargs)
    assert scores.shape == (gen.n_samples,)


def test_cell_cycle_score_no_match_raises(gen):
    kwargs = gen.get_data("numpy", True)
    kwargs["target_dict"] = {"Pos": ["NOPE1", "NOPE2"], "Neg": []}
    with pytest.raises(ValueError, match="No feature genes"):
        cell_cycle_score(**kwargs)


def test_adobo_score(gen, tmp_path):
    kwargs = gen.get_data("pandas", True)
    out_file = tmp_path / "adobo.csv"
    kwargs["file_path"] = out_file
    scores = adobo_score(**kwargs)
    assert scores.shape == (gen.n_samples,)
    assert out_file.is_file()


def test_adobo_score_empty_raises(gen):
    kwargs = gen.get_data("pandas", True)
    kwargs["genes"] = []
    with pytest.raises(ValueError, match="empty"):
        adobo_score(**kwargs)
