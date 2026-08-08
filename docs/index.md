# scTenifoldpy

``scTenifoldpy`` is a Python port of the scTenifold family of
single-cell gene-regulatory-network analyses. It provides three workflows:

- **scTenifoldNet**: compare two single-cell expression matrices and rank
  genes by differential regulation. Use it when you have a control and a
  condition sample.
- **scTenifoldKnk**: build a wild-type network and simulate a virtual
  knockout of one or more genes. Use it when you only have one sample and
  want to predict downstream effects of perturbing a gene.
- **scTenifoldXct**: predict cell-cell interactions between a sender and a
  receiver cell type using per-cell-type GRNs and neural-network manifold
  alignment. Available via the optional ``[xct]`` extra.

Net and Knk share the same backbone: per-cell QC, many PC networks on
resampled cells, tensor decomposition, manifold alignment, and a
differential regulation test. See [Pipeline Steps](pipeline-steps.md)
for the step-by-step inputs and outputs.

## Install

```bash
uv venv
uv pip install scTenifoldpy
```

or:

```bash
pip install scTenifoldpy
```

See [Installation](installation.md) for optional extras, Docker, and
development setup.

## Hello World

```python
from scTenifold import compare_networks
from scTenifold.data import get_test_df

x = get_test_df(n_cells=200, n_genes=300, random_state=0)
y = get_test_df(n_cells=200, n_genes=300, random_state=1)

result = compare_networks(
    x, y,
    qc_kws={"min_lib_size": 1},
    network_kws={"n_nets": 3, "n_samp_cells": 100},
)
print(result.head())
```

The returned DataFrame has one row per shared gene with columns
``Gene``, ``Distance``, ``boxcox-transformed distance``, ``Z``, ``FC``,
``p-value``, and ``adjusted p-value``.

## Next

- [Quickstart](quickstart.md): end-to-end runnable examples.
- [Tutorials](source/1_data.ipynb): notebook examples for data,
  scTenifoldNet, virtual knockout, and visualization.
- [Local Web UI](ui.md): run Net/Knk/GRN-only from a browser (optional extra).
- [scTenifoldXct](xct.md): cell-cell interaction analysis (optional extra).
- [Parallel Backends](parallel-backends.md): scale across cores or a Ray
  cluster.
- [AnnData Input](anndata.md): pass AnnData objects instead of
  DataFrames.
- [Workflow Output](workflow-output.md): what ``save()`` writes and how
  to ``load()`` it back.
- [API Reference](api-reference.md): every exported symbol.

## Version

This documentation reflects ``0.3.x``. See the
[Changelog](changelog.md) for what changed since ``0.1``.
