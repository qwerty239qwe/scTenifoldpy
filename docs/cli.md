# CLI

``scTenifoldpy`` installs a ``scTenifold`` command powered by
[Typer](https://typer.tiangolo.com). It wraps both workflows around a
YAML config file.

## Generate a Config

```bash
scTenifold config --type 1 --path net_config.yml   # scTenifoldNet
scTenifold config --type 2 --path knk_config.yml   # scTenifoldKnk
```

The generated file is a ready-to-edit YAML dump produced by
``scTenifoldNet.get_empty_config()`` or
``scTenifoldKnk.get_empty_config()``.

## Config Schema

```yaml
# net_config.yml (scTenifoldNet)
x_data_path: ./data/ctrl/        # 10x folder OR a .csv/.tsv file
y_data_path: ./data/cond/
x_label: ctrl
y_label: cond

qc_kws:
  min_lib_size: 1000
  remove_outlier_cells: true
  min_percent: 0.05
  max_mito_ratio: 0.1
  min_exp_avg: 0
  min_exp_sum: 0

nc_kws:
  n_nets: 10
  n_samp_cells: 500
  n_comp: 3
  scale_scores: true
  symmetric: false
  q: 0.95
  random_state: 42
  backend: serial
  n_jobs: 1

td_kws:
  method: parafac
  n_decimal: 1
  K: 5
  tol: 1.0e-06
  max_iter: 1000
  random_state: 42

ma_kws:
  d: 30
  tol: 1.0e-08

dr_kws:
  sorted_by: p-value
  ascending: true
```

The ``scTenifoldKnk`` config has the same shape plus ``data_path`` (not
``x_data_path`` / ``y_data_path``), ``strict_lambda``, ``ko_method``,
``ko_genes``, and ``ko_kws``.

## Run a Workflow

```bash
scTenifold net --config net_config.yml --output ./saved_net
scTenifold knk --config knk_config.yml --output ./saved_knk
```

This runs every step end-to-end and writes results into ``--output``;
see [Workflow Output](workflow-output.md) for the directory layout.

## Reload from Disk

```python
from scTenifold import scTenifoldNet

model = scTenifoldNet.load("./saved_net")
print(model.d_regulation.head())
```

## scTenifoldXct Commands

These subcommands require the ``[xct]`` extra
(``pip install "scTenifoldpy[xct]"``). They are thin passthroughs to the
standalone `scTenifoldXct` package; the same commands are also available
as `sctenifoldxct` if that package is installed directly.

### Single-sample interaction analysis

```bash
scTenifold xct data.h5ad \
    --sender cell_A \
    --receiver cell_B \
    --label ident \
    --workdir ./xct_results \
    --output xct_enriched
```

| Option | Short | Default | Description |
|---|---|---|---|
| ``--sender`` | ``-s`` | ``cell_A`` | Sender cell type label |
| ``--receiver`` | ``-r`` | ``cell_B`` | Receiver cell type label |
| ``--label`` | ``-l`` | ``ident`` | ``.obs`` column holding cell-type labels |
| ``--workdir`` | ``-w`` | ``xct_results`` | Output directory for GRN files |
| ``--output`` | ``-o`` | ``xct_enriched`` | Output file stem |
| ``--n_cpus`` | | ``-1`` | CPUs for GRN construction (``-1`` = all) |
| ``--rebuild/--no-rebuild`` | | rebuild | Rebuild gene regulatory networks |
| ``--verbose`` | ``-v`` | false | Verbose output |

### Two-sample differential interaction analysis

```bash
scTenifold xct-merge data.h5ad condition WT KO \
    --sender cell_A \
    --receiver cell_B \
    --label ident
```

Positional arguments: `file`, `cond_label` (the `.obs` column distinguishing
conditions), `cond_wt` (reference label), `cond_ko` (comparison label). All
``--sender`` / ``--receiver`` / ``--label`` / ``--workdir`` / ``--output`` /
``--n_cpus`` / ``--rebuild`` flags are the same as `xct` above (output
defaults to ``xct_enriched_diff``).

## Data Path Conventions

- A directory at ``x_data_path`` is treated as a 10x folder
  (``matrix.mtx`` + ``genes.tsv`` + ``barcodes.tsv``), loaded via
  :func:`scTenifold.data.read_folder`.
- A file with ``.csv`` / ``.tsv`` suffix is loaded via
  ``pandas.read_csv`` with the appropriate separator.
