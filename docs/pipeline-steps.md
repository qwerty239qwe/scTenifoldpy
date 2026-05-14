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
| ``td`` | ``network_dict[label]`` | ``tensor_dict[label]`` (genes x genes, symmetrized) | :func:`tensor_decomp` |
| ``ma`` | ``tensor_dict[x_label]``, ``tensor_dict[y_label]`` | ``manifold`` (``2 * G`` x ``d``) | :func:`manifold_alignment` |
| ``dr`` | ``manifold`` | ``d_regulation`` (genes x stats) | :func:`d_regulation` |

## scTenifoldKnk

```text
data_dict -> qc -> QC_dict -> nc -> network_dict["WT"] -> td -> tensor_dict["WT"] -> ko -> tensor_dict["KO"] -> ma -> manifold -> dr -> d_regulation
```

Differences from ``scTenifoldNet``:

- ``qc`` injects KO-friendly defaults for ``min_exp_avg`` (0.05) and
  ``min_exp_sum`` (25) when missing, and does not apply CPM
  normalization.
- ``td`` post-processes the WT tensor with :func:`strict_direction`,
  controlled by ``strict_lambda``.
- The extra ``ko`` step builds ``tensor_dict["KO"]``:
  - ``ko_method="default"`` zeros out the WT tensor rows for the
    knocked-out genes.
  - ``ko_method="propagation"`` rebuilds PC networks with the targeted
    columns masked using :func:`reconstruct_pcnets`, then re-decomposes.
    Use ``ko_kws={"degree": N}`` to set propagation depth.

## ``run_step`` Overrides

```python
model.run_step("nc", n_nets=5, backend="joblib-loky", n_jobs=4)
```

When ``**kwargs`` are passed, they replace the corresponding ``*_kws``
dict for that call only; they are not merged. To merge, update the
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
