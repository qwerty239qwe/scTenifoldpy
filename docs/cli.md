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

## Data Path Conventions

- A directory at ``x_data_path`` is treated as a 10x folder
  (``matrix.mtx`` + ``genes.tsv`` + ``barcodes.tsv``), loaded via
  :func:`scTenifold.data.read_folder`.
- A file with ``.csv`` / ``.tsv`` suffix is loaded via
  ``pandas.read_csv`` with the appropriate separator.
