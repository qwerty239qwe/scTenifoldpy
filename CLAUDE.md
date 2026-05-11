# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
# or for development
pip install -e .

# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/test_base.py

# Run a specific test
python -m pytest tests/test_base.py::test_scTenifoldNet

# Lint (matches CI config)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# CLI usage
python -m scTenifold config -t 1 -p ./net_config.yml   # generate config template
python -m scTenifold net -c ./net_config.yml -o ./output_folder
python -m scTenifold knk -c ./knk_config.yml -o ./output_folder
```

## Architecture

scTenifoldpy implements two published pipelines for gene regulatory network (GRN) inference from scRNA-seq data:

- **scTenifoldNet** — compares GRNs between two conditions (e.g., treated vs. control)
- **scTenifoldKnk** — simulates virtual gene knockout and scores differentially regulated genes

Both pipelines share a base class (`scBase`) and follow the same five-step sequence: **qc → nc → td → ma → dr**.

### Pipeline steps (in order)

| Step | Method | Output |
|------|--------|--------|
| `qc` | `sc_QC` + `cpm_norm` | Filtered/normalized count matrix |
| `nc` | `make_networks` (PCNet via randomized SVD, parallelized with Ray) | List of sparse PCNet adjacency matrices |
| `td` | `tensor_decomp` (CP decomposition via tensorly) | Single aggregated gene×gene DataFrame |
| `ko` | *(scTenifoldKnk only)* zero-out or propagation-based KO | KO tensor dict entry |
| `ma` | `manifold_alignment` (NLMA via sparse eigendecomposition) | `(n_genes×2, d)` DataFrame |
| `dr` | `d_regulation` (Box-Cox + chi-squared test + FDR) | Ranked gene DataFrame |

### Key design points

- **scBase** holds `data_dict`, `QC_dict`, `network_dict`, `tensor_dict`, `manifold`, `d_regulation` as state. Each step populates the next dict. Call `run_step(name)` to execute one step or `build()` to run the full pipeline.
- **Ray parallelism**: `make_networks` spins up a Ray cluster for PCNet construction (`n_cpus=-1` = auto). Pass `n_cpus=1` in tests to avoid Ray overhead.
- **scTenifoldKnk** uses labels `"WT"` and `"KO"` internally. The default KO method zeroes out the gene row in the tensor; `"propagation"` reconstructs PCNets after KO.
- **save/load**: intermediates are written under subdirectories `qc/`, `nc/`, `td/`, `ma/`, `dr/` as CSV or `.npz` files. Config is stored in `kws.json` at the root of the save directory.
- Input DataFrames are **rows=genes, cols=cells**. Gene names must be strings (not integer indices).

### Module layout

```
scTenifold/
  core/
    _base.py        # scBase, scTenifoldNet, scTenifoldKnk
    _networks.py    # PCNet construction, manifold_alignment, d_regulation
    _decomposition.py  # tensor_decomp (tensorly CP)
    _QC.py          # sc_QC filtering
    _norm.py        # cpm_norm
    _ko.py          # reconstruct_pcnets (propagation KO)
    _utils.py       # cal_fdr, timer decorator
  data/
    _get.py         # fetch_data, list_data (downloads from GitHub dataset repo)
    _io.py          # read_mtx, read_folder
    _sim.py         # get_test_df (synthetic data for tests)
  plotting/
    _plotting.py    # plot_hist
    _dim_reduction.py
  cell_cycle/
    UCell.py, scoring.py  # cell-cycle scoring utilities
  __main__.py       # Typer CLI (config/net/knk subcommands)
```

### Test fixtures

`conftest.py` provides `morphine_datasets` and `aging_datasets` session-scoped fixtures that download real data from GitHub. Tests using these will make network requests. Tests in `test_base.py` use `get_test_df()` with small synthetic data (`n_cells=100, n_genes=100`) and require `n_cpus=1` to avoid Ray initialization.
