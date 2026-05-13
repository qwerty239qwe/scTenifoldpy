# Workflow Output

``scTenifoldNet.save(dir)`` and ``scTenifoldKnk.save(dir)`` persist every
populated step plus a ``kws.json`` config. ``load(dir)`` reconstructs
the object, including intermediate state, without re-running any step.

## Directory Layout

```text
saved_net/
|-- kws.json                 # all *_kws dicts + labels + shared_gene_names
|-- qc/
|   |-- ctrl.csv             # one CSV per label, genes x cells (post-QC)
|   `-- cond.csv
|-- nc/
|   |-- ctrl/
|   |   |-- network_0.npz    # one .npz per PC network (sparse)
|   |   |-- network_1.npz
|   |   `-- ...
|   `-- cond/
|       `-- ...
|-- td/
|   |-- ctrl.npz             # genes x genes tensor (sparse)
|   `-- cond.npz
|-- ma/
|   `-- manifold_alignment.csv
`-- dr/
    `-- d_regulation.csv
```

For ``scTenifoldKnk`` the labels are ``WT`` and ``KO`` where
applicable, and the same step folders apply.

## Selective Save

By default ``save()`` writes every step that has run. Pass ``comps`` to
restrict:

```python
model.save("./saved_net", comps=["qc", "dr"])
```

Valid component names: ``"qc"``, ``"nc"``, ``"td"``, ``"ma"``,
``"dr"``.

## Reload

```python
from scTenifold import scTenifoldNet, scTenifoldKnk

net = scTenifoldNet.load("./saved_net")
knk = scTenifoldKnk.load("./saved_knk")
```

``load()`` reads ``kws.json``, instantiates the class with the same
labels and per-step kwargs, then refills ``QC_dict``, ``network_dict``,
``tensor_dict``, ``manifold``, and ``d_regulation`` from the
corresponding subfolders. Missing folders are skipped; you can load a
half-finished run and continue with ``run_step()``.

## Caveats

- The raw input matrices (``x_data`` / ``y_data`` / ``data``) are not
  written to disk; ``load()`` reinitializes them to empty DataFrames.
  Keep your input files alongside the run directory if you need them
  later.
- ``ko_genes``, ``ko_method``, and ``strict_lambda`` are stored in
  ``kws.json`` for ``scTenifoldKnk``.
