import pandas as pd
from core.utils import get_test_df
from core.base import scTenifoldNet


def test_scTenifoldNet():
    df_1, df_2 = get_test_df(n_cells=1000), get_test_df(n_cells=1000)
    sc = scTenifoldNet(df_1, df_2, "X", "Y", qc_kws={"min_lib_size": 1})
    result = sc.build()
    assert isinstance(result, pd.DataFrame)