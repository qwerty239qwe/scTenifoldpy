import pandas as pd
from scTenifold.core.utils import get_test_df
from scTenifold.core.base import scTenifoldNet


def test_scTenifoldNet():
    df_1, df_2 = get_test_df(n_cells=1000), get_test_df(n_cells=1000)
    sc = scTenifoldNet(df_1, df_2, "X", "Y", qc_kws={"min_lib_size": 10})
    result = sc.build()
    assert isinstance(result, pd.DataFrame)


# data not uploaded yet
# def test_scTenifoldNet_2(control_data, treated_data):
#     sc = scTenifoldNet(control_data, treated_data, "X", "Y")
#     result = sc.build()
