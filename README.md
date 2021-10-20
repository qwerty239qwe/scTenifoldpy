# scTenifoldpy

### Installation
```
pip install sctenifoldpy
```


### Usages
```python
from scTenifold.core.utils import get_test_df
from scTenifold.core.base import scTenifoldNet

df_1, df_2 = get_test_df(n_cells=1000), get_test_df(n_cells=1000)
sc = scTenifoldNet(df_1, df_2, "X", "Y", qc_kws={"min_lib_size": 10})
result = sc.build()
```