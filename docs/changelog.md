# Changelog

## 0.4.0

### Web UI

- **New optional local web UI** for running the scTenifold suite without
  writing code. `sctenifold-ui` starts a FastAPI server and opens a
  single-page app in the browser; everything runs locally and no data
  leaves the machine. Ships as the new `[ui]` extra.
- Three workflows selectable in the UI:
  - **Compare networks** (scTenifoldNet) — rank genes by differential
    regulation between two conditions.
  - **Virtual knockout** (scTenifoldKnk) — rank genes by the predicted
    impact of knocking out one or more genes.
  - **Build GRN only** — infer a single gene regulatory network from one
    sample (`sc_QC` → `cal_pcNet`), with no comparison or knockout. No
    resampling, so the parallel-backend options don't apply and are
    hidden.
- Provide data three ways: a small **synthetic example**, the real **10x
  PBMC3k** dataset (the Seurat/Scanpy tutorial data, QC-filtered,
  downsampled, and cached on first use), or **upload your own** as a
  genes-by-cells CSV or an AnnData `.h5ad` file.
- Results render as a ranked gene table (Net/Knk) or edge list (GRN),
  capped to the top rows on screen and downloadable in full as CSV.
- Jobs run one at a time on a background worker; a run keeps going and is
  repainted correctly when you switch workflow tabs mid-run.

### Packaging

- New optional extra `ui = ["fastapi>=0.110", "uvicorn>=0.29",
  "python-multipart>=0.0.9", "anndata>=0.10"]`. The base install stays
  lightweight and web-framework-free.
- New `sctenifold-ui` console entry point.
- Regenerated `uv.lock` for the `ui` extra and the `httpx` test
  dependency.

### CLI

- New `sctenifold-ui` command (also runnable as
  `python -m scTenifold.webapp`) with `--host`, `--port`, and
  `--no-browser`. Defaults to `127.0.0.1:8001` to avoid the common 8000
  clash.

### Bug Fixes

- Report a clear, actionable error when a knockout gene was removed by QC
  before the knockout step, distinguishing genes dropped by QC from genes
  never present in the input — previously surfaced as an opaque pandas
  `KeyError`.
- Reject `n_jobs=0` (rejected by joblib) up front, validated both
  client-side and in the request schema.

### Robustness

- Uploads are streamed to disk a chunk at a time with a size cap, instead
  of buffering whole files in memory; oversized requests are rejected on
  their declared `Content-Length` before the body is read.
- Dataset fetches use `GITHUB_TOKEN`/`GH_TOKEN` when available and cache
  the repository tree per process, avoiding GitHub's 60-requests/hour
  unauthenticated API rate limit (notably in CI).

### Tests

- New suites: `test_webapp_api`, `test_webapp_jobs`, `test_webapp_cli`,
  `test_webapp_pbmc3k`, and `test_knk_ko_gene_validation`. The web UI
  suites auto-skip without the `[ui]` extra, so the default matrix stays
  green.
- New CI `test-ui` job runs the web UI suites on Python 3.9 and 3.14.

### Docs

- New **Local Web UI** page covering install, launch, the three
  workflows, dataset options, and results; added a UI screenshot and
  promoted the section in the README.

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
