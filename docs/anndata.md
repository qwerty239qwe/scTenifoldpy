# AnnData Input

The canonical internal representation is a ``pandas.DataFrame`` with
**genes as rows** and **cells as columns**. Anywhere the high-level API
accepts ``data``, you can also pass an AnnData-like object — anything
with ``X``, ``var_names``, and ``obs_names`` attributes.

## What gets read

- ``adata.X`` is used by default.
- Pass ``layer="counts"`` (or any other layer name) to read
  ``adata.layers[layer]`` instead.
- Sparse ``X`` / layer values are densified via ``.toarray()``.
- The result is transposed to genes-by-cells, with ``var_names`` as the
  row index and ``obs_names`` as the column index.

## Examples

```python
from scTenifold import compare_networks, virtual_knockout

# compare_networks with two AnnData objects
result = compare_networks(adata_x, adata_y, layer="counts")

# virtual_knockout reads adata.X if layer is omitted
result = virtual_knockout(adata, ko_genes=["GeneA"])
```

The same applies to the class API — ``scTenifoldNet(x_data=..., y_data=...)``
and ``scTenifoldKnk(data=...)`` accept AnnData inputs and convert them
internally via :func:`scTenifold.core._networks.anndata_to_dataframe`.

## Round-trip

```python
import anndata as ad
from scTenifold.core._networks import anndata_to_dataframe

adata = ad.read_h5ad("sample.h5ad")
df = anndata_to_dataframe(adata, layer="counts")
assert df.index.equals(adata.var_names)
assert df.columns.equals(adata.obs_names)
```

## Caveats

- ``scTenifoldpy`` does not write results back into the AnnData object;
  outputs are pandas DataFrames keyed by gene name.
- Scanpy is an *optional* extra — install via
  ``pip install "scTenifoldpy[scanpy]"`` if you also want the Scanpy
  utilities.
