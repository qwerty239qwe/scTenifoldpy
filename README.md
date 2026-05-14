# scTenifoldpy

[![CI](https://github.com/qwerty239qwe/scTenifoldpy/actions/workflows/ci.yml/badge.svg)](https://github.com/qwerty239qwe/scTenifoldpy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/qwerty239qwe/scTenifoldpy/graph/badge.svg?token=H16KYU2K7M)](https://codecov.io/gh/qwerty239qwe/scTenifoldpy)
[![PyPI](https://img.shields.io/pypi/v/scTenifoldpy.svg)](https://pypi.org/project/scTenifoldpy/)
[![Python](https://img.shields.io/pypi/pyversions/scTenifoldpy.svg)](https://pypi.org/project/scTenifoldpy/)
[![License](https://img.shields.io/github/license/qwerty239qwe/scTenifoldpy.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.patter.2020.100139-blue)](https://www.sciencedirect.com/science/article/pii/S2666389920301872)

`scTenifoldpy` is a Python implementation of scTenifoldNet and scTenifoldKnk workflows.

## Installation

```bash
uv venv
uv add scTenifoldpy
```

or:

```bash
pip install scTenifoldpy
```

Optional extras:

```bash
# use uv
uv venv
uv add "scTenifoldpy[scanpy]"
uv add "scTenifoldpy[parallel-ray]"

# or use pip
pip install "scTenifoldpy[scanpy]"
pip install "scTenifoldpy[parallel-ray]"
```


## Docker

Build the default runtime image:

```bash
docker build -t sctenifoldpy .
```

Run the CLI from the container, mounting the current directory as the working directory:

```bash
docker run --rm -v "$PWD:/workspace" sctenifoldpy scTenifold --help
```

PowerShell:

```powershell
docker run --rm -v "${PWD}:/workspace" sctenifoldpy scTenifold --help
```

Optional extras can be included at build time:

```bash
docker build --build-arg EXTRAS=scanpy -t sctenifoldpy:scanpy .
docker build --build-arg EXTRAS=parallel-ray -t sctenifoldpy:ray .
```

## Class API

```python
from scTenifold.data import get_test_df
from scTenifold import scTenifoldNet

df_1 = get_test_df(n_cells=1000)
df_2 = get_test_df(n_cells=1000)

sc = scTenifoldNet(
    df_1,
    df_2,
    "X",
    "Y",
    qc_kws={"min_lib_size": 10},
    nc_kws={"backend": "serial", "n_jobs": 1},
)
result = sc.build()
```

## High-Level API

```python
from scTenifold import compare_networks, virtual_knockout

result = compare_networks(
    df_1,
    df_2,
    qc_kws={"min_lib_size": 10, "plot": False},
    network_kws={"n_nets": 3, "n_samp_cells": 100},
    backend="joblib-threading",
    n_jobs=4,
)

knockout = virtual_knockout(
    df_1,
    ko_genes=["NG-1"],
    qc_kws={"min_lib_size": 10, "min_percent": 0.001},
)
```

## Parallel Backends

Network construction defaults to deterministic serial execution:

```python
from scTenifold import make_networks

networks = make_networks(df_1, backend="serial", n_jobs=1)
networks = make_networks(df_1, backend="joblib-loky", n_jobs=4)
```

Supported backends are `serial`, `joblib-loky`, `joblib-threading`, and `ray`. Ray is optional and requires `scTenifoldpy[parallel-ray]`.

## CLI

```bash
scTenifold config -t 1 -p ./net_config.yml
scTenifold net -c ./net_config.yml -o ./output_folder
scTenifold knk -c ./knk_config.yml -o ./output_folder
```

## Citation

If you use `scTenifoldpy` in scientific work, please cite this software
using the metadata in [`CITATION.cff`](CITATION.cff), and cite the
underlying method paper that matches your analysis:

**scTenifoldNet**

Osorio, D., Zhong, Y., Li, G., Huang, J. Z., & Cai, J. J. (2020).
scTenifoldNet: A machine learning workflow for constructing and comparing
transcriptome-wide gene regulatory networks from single-cell data.
*Patterns*, 1(9), Article 100139.
https://doi.org/10.1016/j.patter.2020.100139

**scTenifoldKnk**

Osorio, D., Zhong, Y., Li, G., Xu, Q., Yang, Y., Tian, Y., Chapkin, R. S.,
Huang, J. Z., & Cai, J. J. (2022). scTenifoldKnk: An efficient virtual
knockout tool for gene function predictions via single-cell gene regulatory
network perturbation. *Patterns*, 3(3), Article 100434.
https://doi.org/10.1016/j.patter.2022.100434
