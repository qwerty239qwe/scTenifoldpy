# scTenifoldXct

`scTenifoldXct` is the cell-cell interaction method of the scTenifold
suite. It builds per-cell-type gene regulatory networks, aligns them
with a neural-network manifold alignment, and scores ligand-receptor
interactions between a sender and a receiver cell type.

It is **maintained and released separately**
([cailab-tamu/scTenifoldXct](https://github.com/cailab-tamu/scTenifoldXct),
PyPI: `scTenifoldXct`). `scTenifoldpy` does not copy its code — it
declares it as an optional dependency and re-exports it, so
`from scTenifold import scTenifoldXct` returns the exact same class and
results as the standalone package. Bug reports and feature requests for
the method itself belong in the scTenifoldXct repo.

## Installation

```bash
pip install "scTenifoldpy[xct]"
# or
uv add "scTenifoldpy[xct]"
```

This pulls `scTenifoldXct` and its dependencies (PyTorch, scanpy,
anndata, ray, ...). The extra requires **Python >= 3.10**. The base
`scTenifoldpy` install is unaffected and stays torch-free; the Net/Knk
workflows never import any `xct` dependency.

## Python API

```python
import scanpy as sc
from scTenifold import scTenifoldXct

adata = sc.read_h5ad("log_normalised.h5ad")  # log1p-normalised counts

xct = scTenifoldXct(
    data=adata,
    source_celltype="cell_A",
    target_celltype="cell_B",
    obs_label="ident",          # .obs column holding cell-type labels
    rebuild_GRN=True,
    GRN_file_dir="./xct_results",
    verbose=True,
)

xct.get_embeds(train=True)      # manifold alignment
enriched = xct.chi2_test()      # significant ligand-receptor pairs
```

Differential interaction analysis across two conditions:

```python
from scTenifold import merge_scTenifoldXct

merged = merge_scTenifoldXct(xct_wt, xct_ko)
emb = merged.get_embeds(train=True)
merged.nn_aligned_diff(emb)
diff = merged.chi2_diff_test()
```

`from scTenifold import scTenifoldXct` is identical to
`from scTenifoldXct import scTenifoldXct`; use whichever import you
prefer. The full API reference lives in the
[scTenifoldXct documentation](https://github.com/cailab-tamu/scTenifoldXct).

## CLI

The Xct entry points are exposed through the shared `scTenifold` CLI as
thin passthroughs (the standalone `sctenifoldxct` console script,
installed with the extra, works too):

```bash
# single-sample interaction analysis
scTenifold xct data.h5ad -s cell_A -r cell_B -l ident -w ./xct_results

# two-sample differential interaction analysis
scTenifold xct-merge data.h5ad condition WT KO -s cell_A -r cell_B
```

## Citation

Yang, Y., Osorio, D., Long, W., et al. (2023). scTenifoldXct: a
semi-supervised method for predicting cell-cell interactions and
mapping cellular communication graphs. *Cell Systems*, 14(4), 302-311.
https://doi.org/10.1016/j.cels.2023.01.004
