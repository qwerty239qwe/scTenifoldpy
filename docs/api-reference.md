# API Reference

The API reference is split by workflow layer so the sidebar stays short
and predictable.

## Public Entry Points

- [High-Level API](api/high-level.md): one-call workflows,
  ``compare_networks`` and ``virtual_knockout``.
- [Workflow Classes](api/workflows.md): step-wise
  ``scTenifoldNet`` and ``scTenifoldKnk`` classes.

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
