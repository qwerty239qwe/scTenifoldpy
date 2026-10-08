# Changelog

## 2.0.0

Version 2.0.0 aligns with the R package scTenifoldKnk 2.0.0 and gives the
same results (`tests/test_heat.py`).

### New

- **Direction of the response.** `knockout_direction` predicts whether each
  gene goes up or down after a knockout, from the heat kernel of the WT
  gene-gene correlation matrix of log1p(CPM) expression. `scTenifoldKnk`
  adds it by default (`dr_direction=True`, `dr_direction_t=5`) as the
  `direction` and `direction score` columns of the differential regulation
  table. `dr_direction_regress_lib_size=True` (off by default) regresses
  log library size out of each gene first, for data sets in which nearly
  all genes are predicted down.
- **Heat manifold alignment.** `hk_manifold_alignment` computes the heat
  kernel of the WT network once (`heat_kernel`) and reads every knockout
  from it. Selected with `ma_method="heat"` (`ma_heat_t=10`).
- **Transcriptome-wide knockouts.** `scTenifoldKnk.transcriptome_wide()`
  knocks out each gene of the WT network in turn (heat manifold alignment
  by default) and returns the distance and direction matrices.
- `virtual_knockout` forwards the new options; `d_regulation` accepts a
  `direction` Series. The qc step of `scTenifoldKnk` keeps the library
  sizes of the QC'd cells (saved with the instance).

### Changes in behaviour

- The differential regulation table of `scTenifoldKnk` has two more
  columns by default. `dr_direction=False` gives the 0.5.1 table; the
  other columns and the gene ranking are unchanged.

## 0.5.1

- **scTenifoldKnk leaves the knocked-out genes out of the expectation** of
  the differential regulation test, as the R package scTenifoldKnk does up
  to 1.0.3 and again from 1.1.5; earlier versions of scTenifoldpy never
  did. The fold-change of each gene is its squared distance relative to
  the mean squared distance of the genes that were not knocked out. With
  the knocked-out genes in that mean, their large distances hid the other
  genes, and often only the knocked-out genes were significant
  (cailab-tamu/scTenifoldKnk#45). Distances, Z-scores and the ranking of
  the genes do not change; the FC, p-values and adjusted p-values do.
- `d_regulation` accepts `ko_genes`, a gene name or list of names left out
  of the expectation. `scTenifoldKnk` passes the genes of its knockout
  step; setting `ko_genes` or `n_ko_genes` in `dr_kws` overrides them.
- The QC step of `scTenifoldNet` and `scTenifoldKnk` no longer plots its
  histogram by default. The plot called `plt.show()`, which with an
  interactive matplotlib backend stopped the pipeline until the window was
  closed. Pass `qc_kws={"plot": True}` to get it.
- The tensor decomposition holds about one copy of the networks instead of
  up to three: `tensor_decomp` and `cp_decomposition` also accept a list of
  the networks, which the pipelines now pass, and the norm of the tensor is
  computed slice by slice. Each copy takes about 8 GB for 10 networks of
  10,000 genes, so a knockout of that size peaks at about 8 GB instead of
  about 22 GB. The results are the same.
- `ko_method="propagation"` no longer prints the gene index.

## 0.5.0

> **Breaking change: results differ from 0.4.x.** This release changes the
> random number generator, the PC networks, the tensor decomposition, QC
> and the Box-Cox step, so networks and gene rankings are different from
> those of 0.4.x for the same data. Results from 0.4.x **cannot be
> reproduced with this version, even by passing the old parameters** (such
> as `random_state=42`, `method="parafac"` or `tol=1e-6`). To reproduce
> earlier results, pin the version you used, e.g.
> `pip install "scTenifoldpy<0.5"`.

### Same results as the R packages

`scTenifoldNet` and `scTenifoldKnk` now give the same results as the R
packages scTenifoldNet 1.4.3 and scTenifoldKnk 1.1.4 with their default
settings and `seed = 1`.

- **Random numbers**: cells are subsampled and the tensor decomposition is
  initialized with a port of R's generator (`RRandom`), so `random_state`
  means the same as `seed` in R. The default is now `1` (was `42`).
- **PC networks** are exact, computed as `pcNet` in R from one
  eigendecomposition per network, instead of a randomized SVD per gene.
  Genes that are constant in the sampled cells get no edges; `n_comp = 2`
  is allowed. `make_networks` gains `prior_network`.
- **Tensor decomposition** defaults to `method="cp_als"`, a port of the R
  CP-ALS; tensorly methods are still available through `method`, and
  `tensorly` is now an optional extra (`pip install "scTenifoldpy[tensorly]"`). The
  default `tol` is `1e-5` (was `1e-6`).
- **Differential regulation** selects the Box-Cox power as
  `MASS::boxcox`, standardizes with the sample standard deviation, and
  sets the p-value of genes whose distance is at the level of
  floating-point noise to 1 (warning if that applies to every gene).
  Results are sorted stably.
- **QC** detects outlier cells with the hinges of `boxplot.stats` and keeps
  the input gene order.
- **scTenifoldNet** uses `K = 3` in the tensor decomposition, and keeps
  the self-loops in `tensor_dict` (they are symmetrized for the alignment
  only, as in R).
- **scTenifoldKnk** normalizes to CPM after QC and no longer adds
  `min_exp_avg`/`min_exp_sum` filters; it uses `q = 0.9`, `K = 3` and
  `n_decimal = 3`, and warns when the knocked-out genes have no outgoing
  edges.
- The shipped `config/net_config.yml` and `config/knk_config.yml` use these
  defaults.

### Other changes

- `pc_net_calc` warns (`DeprecationWarning`) when `random_state` is passed:
  the networks are exact, so it has no effect, and it will be removed.
- `backend="joblib-threading"` limits BLAS to one thread per worker (via
  `threadpoolctl`, now a direct dependency), since concurrent calls into a
  multithreaded OpenBLAS could crash or hang.
- The CP-ALS decomposition works on contiguous copies of the tensor
  slices, which is several times faster on large networks with the same
  results.
- `scTenifold.__version__` is read from the installed package metadata, so
  it always matches `pyproject.toml` (it still said `0.3.0` in 0.4.0).
- The `n_cpus` deprecation message no longer names a past release for its
  removal.

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
