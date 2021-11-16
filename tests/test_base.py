import pandas as pd
import numpy as np
import pytest

from scTenifold.data import get_test_df
from scTenifold import scTenifoldNet, scTenifoldKnk


def test_scTenifoldNet():
    df_1, df_2 = get_test_df(n_cells=100, n_genes=100), get_test_df(n_cells=100, n_genes=100)
    sc = scTenifoldNet(df_1, df_2, "X", "Y", qc_kws={"min_lib_size": 1}, nc_kws={"n_cpus": 1})
    result = sc.build()
    print(result)
    assert isinstance(result, pd.DataFrame)
    sc.save(file_dir=".")
    sc2 = scTenifoldNet.load(file_dir=".")


def test_scTenifoldKnk_method1():
    df = get_test_df(n_cells=100, n_genes=100, random_state=42)
    sc = scTenifoldKnk(data=df,
                       ko_genes=["NG-1"],
                       qc_kws={"min_lib_size": 1})
    sc.run_step("qc")
    sc.run_step("nc", n_cpus=1)
    sc.run_step("td")
    sc.run_step("ko", ko_genes=[sc.tensor_dict["WT"].index.to_list()[0]])
    sc.run_step("ma")
    sc.run_step("dr")
    assert isinstance(sc.d_regulation, pd.DataFrame)
    sc.save(file_dir="./saved_knk")
    sc2 = scTenifoldKnk.load(file_dir="./saved_knk")
    np.array_equal(sc.tensor_dict["WT"], sc2.tensor_dict["WT"])


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
