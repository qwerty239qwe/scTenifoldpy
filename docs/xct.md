# scTenifoldXct

`scTenifoldXct` is the cell-cell interaction method of the scTenifold
suite. It builds per-cell-type gene regulatory networks, aligns them
with a neural-network manifold alignment, and scores ligand-receptor
interactions between a sender and a receiver cell type.

It carries heavier dependencies than the Net/Knk workflows (PyTorch,
scanpy, anndata, ray, statsmodels), so it is shipped behind an optional
extra and only imported on first use.

## Installation

```bash
pip install "scTenifoldpy[xct]"
# or
uv add "scTenifoldpy[xct]"
```

The `xct` extra requires **Python >= 3.10** (PyTorch / scanpy). The
base `scTenifoldpy` install is unaffected and stays torch-free; the
Net/Knk workflows do not import any `xct` dependency.

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

## CLI

The Xct entry points are integrated into the shared `scTenifold` CLI:

```bash
# single-sample interaction analysis
scTenifold xct data.h5ad -s cell_A -r cell_B -l ident -w ./xct_results

# two-sample differential interaction analysis
scTenifold xct-merge data.h5ad condition WT KO -s cell_A -r cell_B
```

## Public API

| Name | Purpose |
|---|---|
| `scTenifoldXct` | Single-sample cell-cell interaction analysis |
| `merge_scTenifoldXct` | Two-sample differential interaction analysis |
| `set_seed` | Seed torch/numpy for reproducible training |
| `get_Xct_pairs` | Extract candidate ligand-receptor pairs |
| `plot_XNet` | Visualise the interaction network |

All are importable from the top-level package, e.g.
`from scTenifold import scTenifoldXct`.

## Citation

Yang, Y., Osorio, D., Long, W., et al. (2023). scTenifoldXct: a
semi-supervised method for predicting cell-cell interactions and
mapping cellular communication graphs. *Cell Systems*, 14(4), 302-311.
https://doi.org/10.1016/j.cels.2023.01.004
