# Quickstart

Two end-to-end examples plus the step-wise class API.

For notebook-style walkthroughs, see the [Tutorials](source/1_data.ipynb).

## 1. Compare Two Conditions: ``compare_networks``

```python
from scTenifold import compare_networks
from scTenifold.data import get_test_df

x = get_test_df(n_cells=200, n_genes=300, random_state=0)
y = get_test_df(n_cells=200, n_genes=300, random_state=1)

result = compare_networks(
    x, y,
    x_label="ctrl",
    y_label="cond",
    qc_kws={"min_lib_size": 1},
    network_kws={"n_nets": 3, "n_samp_cells": 100},
    backend="serial",
)

print(result.head())
```

Returned columns:

| Column | Meaning |
|---|---|
| ``Gene`` | Gene symbol shared between the two samples. |
| ``Distance`` | Euclidean distance between aligned manifold embeddings of the X and Y sides of the gene. |
| ``boxcox-transformed distance`` | Distance after Box-Cox transform. |
| ``Z`` | Z-score of the transformed distance. |
| ``FC`` | Squared distance divided by the expected squared distance. |
| ``p-value`` | Chi-squared tail probability against ``FC``. |
| ``adjusted p-value`` | Benjamini-Hochberg-adjusted ``p-value``. |

## 2. Virtual Knockout: ``virtual_knockout``

```python
from scTenifold import virtual_knockout
from scTenifold.data import get_test_df

data = get_test_df(n_cells=300, n_genes=300, random_state=0)

result = virtual_knockout(
    data,
    ko_genes=["NG-1"],
    qc_kws={"min_lib_size": 1, "min_exp_avg": 0, "min_exp_sum": 0},
    network_kws={"n_nets": 3, "n_samp_cells": 100},
    ko_method="default",
)

print(result.sort_values("p-value").head())
```

For propagation-style knockouts, pass ``ko_method="propagation"`` and
``ko_kws={"degree": N}``. This rebuilds PC networks with the targeted
gene masked and controls the propagation depth with ``degree``.

## 3. Step-Wise: ``scTenifoldNet`` / ``scTenifoldKnk``

Use the class API when you want to inspect intermediate state, override
kwargs per step, or save/load partial pipelines.

```python
from scTenifold import scTenifoldNet

model = scTenifoldNet(
    x, y,
    x_label="ctrl", y_label="cond",
    qc_kws={"min_lib_size": 1},
    nc_kws={"n_nets": 3, "n_samp_cells": 100, "backend": "serial"},
)

model.run_step("qc")
model.run_step("nc")
model.run_step("td")
model.run_step("ma")
model.run_step("dr")

print(model.QC_dict["ctrl"].shape)         # genes x cells after QC
print(len(model.network_dict["ctrl"]))     # n_nets PC networks
print(model.tensor_dict["ctrl"].shape)     # genes x genes
print(model.manifold.shape)                # (2 * shared_genes) x d
print(model.d_regulation.head())

model.save("./run-ctrl-vs-cond")
```

The same pattern works for ``scTenifoldKnk`` with the extra ``"ko"``
step between ``"td"`` and ``"ma"``. See
[Pipeline Steps](pipeline-steps.md).

## Choosing Kwargs

Each step has a kwargs dict on the class: ``qc_kws``, ``nc_kws``,
``td_kws``, ``ma_kws``, ``dr_kws``, plus ``ko_kws`` on
``scTenifoldKnk``. You can:

- Pass them at construction time.
- Override per call via ``run_step("nc", n_nets=5)``.
- List defaults with ``scTenifoldNet.list_kws("nc_kws")``.
