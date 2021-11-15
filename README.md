# scTenifoldpy
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/biodbs.svg)](https://pypi.python.org/pypi/biodbs/)
[![Pattern](https://img.shields.io/badge/DOI-10.1016/j.patter.2020.100139-blue)](https://www.sciencedirect.com/science/article/pii/S2666389920301872#bib48)
[![GitHub license](https://img.shields.io/github/license/qwerty239qwe/scTenifoldpy.svg)](https://github.com/qwerty239qwe/scTenifoldpy/blob/master/LICENSE)

This package is a Python version of [scTenifoldNet](https://github.com/cailab-tamu/scTenifoldNet) 
and [scTenifoldKnk](https://github.com/cailab-tamu/scTenifoldKnk). If you are a R/MATLAB user, 
please install them to use their functions. 
Also, please [cite](https://www.sciencedirect.com/science/article/pii/S2666389920301872) the original paper properly 
if you are using this in a scientific publication. Thank you!

### Installation
```
pip install sctenifoldpy
```


### Usages

#### scTenifoldNet
```python
from scTenifold.data import get_test_df
from scTenifold import scTenifoldNet

df_1, df_2 = get_test_df(n_cells=1000), get_test_df(n_cells=1000)
sc = scTenifoldNet(df_1, df_2, "X", "Y", qc_kws={"min_lib_size": 10})
result = sc.build()
```

#### scTenifoldKnk
```python
from scTenifold.data import get_test_df
from scTenifold import scTenifoldKnk

df = get_test_df(n_cells=1000)
sc = scTenifoldKnk(data=df,
                   ko_method="default",
                   ko_genes=["NG-1"],  # the gene you wants to knock out
                   qc_kws={"min_lib_size": 10, "min_percent": 0.001},
                   # ko_kws={"degree": 10}
                   )
result = sc.build()
```