# API Reference

The API reference is split by workflow layer so the sidebar stays short
and predictable.

## Public Entry Points

- [High-Level API](api/high-level.md): one-call workflows,
  ``compare_networks`` and ``virtual_knockout``.
- [Workflow Classes](api/workflows.md): step-wise
  ``scTenifoldNet`` and ``scTenifoldKnk`` classes.

## scTenifoldXct Re-exports

When the ``[xct]`` extra is installed, the following names are importable
directly from ``scTenifold``:

| Name | Description |
|---|---|
| ``scTenifoldXct`` | Main class for single-sample cell-cell interaction analysis |
| ``merge_scTenifoldXct`` | Class for two-sample differential interaction analysis |
| ``set_seed`` | Set random seeds for reproducibility |
| ``get_Xct_pairs`` | Retrieve significant ligand-receptor pairs |
| ``plot_XNet`` | Plot the interaction network |

```python
from scTenifold import scTenifoldXct, merge_scTenifoldXct
```

These are lazy re-exports of the standalone ``scTenifoldXct`` package —
they return the exact same objects. Accessing them without the extra raises
an actionable ``ImportError``. See [scTenifoldXct](xct.md) for usage and
the full API reference in the
[scTenifoldXct repository](https://github.com/cailab-tamu/scTenifoldXct).

## Lower-Level Functions

- [Network Functions](api/networks.md): PC network construction,
  manifold alignment, differential regulation, AnnData conversion, and
  edge-direction helpers.
- [Pipeline Functions](api/pipeline.md): QC, normalization, tensor
  decomposition, and knockout propagation internals.
- [Data Helpers](api/data.md): loading and test-data generation.
- [Plotting](api/plotting.md): network, histogram, embedding, and QQ
  plots.
- [Utilities](api/utilities.md): small shared helpers.

For most users, start with [High-Level API](api/high-level.md) or
[Workflow Classes](api/workflows.md).
