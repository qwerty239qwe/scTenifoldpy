import pandas as pd
import pytest

from scTenifold.data import get_test_df
from scTenifold import scTenifoldNet, scTenifoldKnk


def test_scTenifoldNet():
    df_1, df_2 = get_test_df(n_cells=100), get_test_df(n_cells=100)
    sc = scTenifoldNet(df_1, df_2, "X", "Y", qc_kws={"min_lib_size": 10})
    result = sc.build()
    assert isinstance(result, pd.DataFrame)
    # sc.save(file_dir=".")
    # sc2 = scTenifoldNet.load(file_dir=".")


def test_scTenifoldKnk_method1():
    df = get_test_df(n_cells=100)
    sc = scTenifoldKnk(data=df,
                       ko_genes=["NG-1"],  # the gene you wants to knock out
                       qc_kws={"min_lib_size": 10})
    result = sc.build()
    assert isinstance(result, pd.DataFrame)
    # sc.save(file_dir=".")
    # sc2 = scTenifoldNet.load(file_dir=".")


@pytest.mark.test
def test_scTenifoldKnk_method2():
    df = get_test_df(n_genes=100, n_cells=100)
    sc = scTenifoldKnk(data=df,
                       ko_method="propagation",
                       ko_genes=["NG-1"],  # the gene you wants to knock out
                       qc_kws={"min_lib_size": 10, "min_percent": 0.001},
                       ko_kws={"degree": 10})
    result = sc.build()
    assert isinstance(result, pd.DataFrame)

# data not uploaded yet
# def test_scTenifoldNet_2(control_data, treated_data):
#     sc = scTenifoldNet(control_data, treated_data, "X", "Y")
#     result = sc.build()
