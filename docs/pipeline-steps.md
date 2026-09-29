# Pipeline Steps

Both workflows are chains of named steps. Each step reads attributes
populated by previous steps and writes its own output. ``run_step(name)``
runs exactly one step; ``build()`` runs the full pipeline.

## scTenifoldNet

```text
data_dict -> qc -> QC_dict -> nc -> network_dict -> td -> tensor_dict -> ma -> manifold -> dr -> d_regulation
```

| Step | Reads | Writes | Underlying call |
|---|---|---|---|
| ``qc`` | ``data_dict[label]`` | ``QC_dict[label]`` (CPM-normalized) | :func:`sc_QC` + :func:`cpm_norm` |
| ``nc`` | ``QC_dict[label]`` on ``shared_gene_names`` | ``network_dict[label]`` (list of sparse PC networks), ``shared_gene_names`` | :func:`make_networks` |
| ``td`` | ``network_dict[label]`` | ``tensor_dict[label]`` (genes x genes) | :func:`tensor_decomp` |
| ``ma`` | ``tensor_dict[x_label]``, ``tensor_dict[y_label]`` (symmetrized) | ``manifold`` (``2 * G`` x ``d``) | :func:`manifold_alignment` |
| ``dr`` | ``manifold`` | ``d_regulation`` (genes x stats) | :func:`d_regulation` |

## scTenifoldKnk

```text
data_dict -> qc -> QC_dict -> nc -> network_dict["WT"] -> td -> tensor_dict["WT"] -> ko -> tensor_dict["KO"] -> ma -> manifold -> dr -> d_regulation
```

Differences from ``scTenifoldNet``:

- The defaults follow the R package: ``nc_kws`` uses ``q=0.9``,
  ``td_kws`` uses ``n_decimal=3``, and ``ma_kws`` uses ``d=2``.
- ``td`` post-processes the WT tensor with :func:`strict_direction`,
  controlled by ``strict_lambda``.
- The extra ``ko`` step builds ``tensor_dict["KO"]``:
  - ``ko_method="default"`` zeros out the WT tensor rows for the
    knocked-out genes.
  - ``ko_method="propagation"`` rebuilds PC networks with the targeted
    columns masked using :func:`reconstruct_pcnets`, then re-decomposes.
    Use ``ko_kws={"degree": N}`` to set propagation depth.

## Agreement with the R Packages

With the default settings, ``scTenifoldNet`` and ``scTenifoldKnk``
(``ko_method="default"``) give the same results as ``scTenifoldNet()``
and ``scTenifoldKnk()`` in R with their defaults and ``seed = 1``:

- Cells are subsampled and the tensor decomposition is initialized with
  a port of R's random number generator (:class:`RRandom`), so
  ``random_state`` has the meaning of ``seed`` in R.
- PC networks are computed exactly (no randomized SVD), as ``pcNet``.
- The tensor decomposition (``method="cp_als"``) is a port of the R
  CP-ALS; ``method`` can still name a tensorly decomposition.
- ``d_regulation`` uses the Box-Cox power selected by
  ``MASS::boxcox``, and genes whose distance is at the level of
  floating-point noise get a p-value of 1.

Networks, tensors and manifold coordinates agree to floating-point
precision (the manifold up to the sign of each column). Edges whose
weight ties the quantile threshold, which happens with duplicated
genes, can be kept by one implementation and dropped by the other.

When some genes have distances at the level of floating-point noise,
their Box-Cox transformed distances depend on the last bits of the
arithmetic, and so do the mean and standard deviation used for every
Z-score: Z can then differ slightly between implementations, or between
runs, for all genes. Distances, fold changes, p-values and the ranking do
not depend on it.

## ``run_step`` Overrides

```python
model.run_step("nc", n_nets=5, backend="joblib-loky", n_jobs=4)
```

When ``**kwargs`` are passed, they replace the corresponding ``*_kws``
dict for that call only (over the pipeline defaults); they are not
merged with the stored dict. To merge, update the
stored dict instead:

```python
model.nc_kws["n_jobs"] = 4
model.run_step("nc")
```

## Inspecting State

After each step the relevant attribute is populated and can be
inspected, plotted, or pickled separately.

```python
model.run_step("qc")
print(model.QC_dict["ctrl"].shape)

model.run_step("nc")
print([net.shape for net in model.network_dict["ctrl"]])
```

See [Workflow Output](workflow-output.md) for the on-disk layout when
you call ``save()``.
