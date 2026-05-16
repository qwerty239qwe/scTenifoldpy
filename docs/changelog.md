# Changelog

## 0.3.0

### API

- **scTenifoldXct** (cell-cell interaction prediction) is now available
  through `scTenifoldpy`. It is **not vendored**: the separately
  maintained PyPI package `scTenifoldXct` is declared as the optional
  `[xct]` extra and re-exported lazily, so
  `from scTenifold import scTenifoldXct` returns the exact same class
  and results as the standalone package.
- New lazily re-exported names: `scTenifoldXct`, `merge_scTenifoldXct`,
  `set_seed`, `get_Xct_pairs`, `plot_XNet`. Accessing them without the
  extra raises an actionable `ImportError`.

### Packaging

- New optional extra `xct = ["scTenifoldXct>=0.2"]` (pulls torch,
  scanpy, anndata, ray, ... transitively). The base install stays
  lightweight and torch-free; the extra requires Python >= 3.10.

### CLI

- New `scTenifold xct` / `scTenifold xct-merge` subcommands — thin
  passthroughs to the external package.

### Tests

- New `test_xct` integration-contract suite; auto-skips without the
  extra. A dedicated CI job runs it on Python 3.10-3.12.

## 0.2.0

### Packaging

- Migrated packaging metadata from ``setup.py`` to ``pyproject.toml``.
- Declared support for Python 3.9 to 3.14.
- Dropped Python 3.7 and 3.8 support.
- Made Scanpy (``scanpy``) and Ray (``parallel-ray``) optional extras.
- Added a ``uv.lock`` for reproducible development environments.
- Added Docker runtime image support.

### API

- New high-level entry points in ``scTenifold.core._api``:
  - ``compare_networks(x_data, y_data, ...)``: full scTenifoldNet
    workflow, returns the differential regulation DataFrame.
  - ``virtual_knockout(data, ko_genes, ...)``: full scTenifoldKnk
    workflow with ``ko_method="default"`` or ``"propagation"``.
- Accept AnnData-like inputs anywhere high-level APIs accept expression
  data; use ``layer="counts"`` to pick a non-default AnnData layer.
- Added strict annotations for AnnData-like expression inputs via a
  structural ``AnnDataLike`` protocol.
- Added ``Literal``-typed ``step_name`` annotations on ``run_step`` for
  both classes.

### Parallel Computing

- Selectable backends for ``make_networks``: ``"serial"``,
  ``"joblib-loky"``, ``"joblib-threading"``, ``"ray"``.
- New ``n_jobs`` argument; ``n_cpus`` retained as a deprecated alias
  during ``0.2.x``.
- Ray is no longer required by default.

### Determinism

- Preserved deterministic shared-gene ordering between samples.
- ``randomized_svd`` called with ``flip_sign=True`` for stable signs.

### Bug Fixes

- Fixed z-score calculation in ``d_regulation`` (computed on the
  Box-Cox-transformed distances).
- Allow multiple values for ``ko_genes`` without error.
- Handle ``d < 30`` requests in ``manifold_alignment`` without crashing.
- Workaround for a Ray dependency import failure on some platforms.

### Tests

- New suites: ``test_io``, ``test_modernization``, ``test_plotting``,
  ``test_cell_cycle``.

### Docs

- Added an mkdocs-material documentation site with pages for
  installation, quickstart, AnnData input, parallel backends, pipeline
  steps, workflow output, CLI, API reference, citation, and changelog.
- Added uv installation examples.
