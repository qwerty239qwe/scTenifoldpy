# Parallel Backends

Network construction (``make_networks``) builds ``n_nets`` PC networks
on independent cell subsamples. Each network is independent, so the
heavy step accepts a ``backend`` / ``n_jobs`` pair.

## Backends

| Backend | Parallelism | Extra dependency | Best for |
|---|---|---|---|
| ``"serial"`` (default) | none | none | Small data, reproducibility checks, debugging. |
| ``"joblib-loky"`` | processes | none | CPU-bound runs on a single machine; bypasses the GIL. |
| ``"joblib-threading"`` | threads | none | NumPy/BLAS-heavy runs that release the GIL; lower memory. |
| ``"ray"`` | processes / cluster | ``ray>=2`` (extra) | Larger fan-out or running across multiple machines. |

## Usage

```python
from scTenifold import make_networks

nets = make_networks(data, n_nets=10, backend="joblib-loky", n_jobs=4)
```

The same applies via ``nc_kws`` in the class API or ``network_kws`` /
``backend`` / ``n_jobs`` in ``compare_networks`` and
``virtual_knockout``:

```python
result = compare_networks(
    x, y,
    backend="joblib-loky",
    n_jobs=8,
    network_kws={"n_nets": 10},   # backend/n_jobs propagated automatically
)
```

## ``n_jobs`` Semantics

- ``1`` (default): a single worker.
- ``N > 1``: ``N`` workers.
- ``-1``: all available cores. Internally mapped to ``None`` for joblib
  and to ``num_cpus=None`` for Ray.

## Reproducibility

All backends share the same ``random_state`` propagated through
``randomized_svd``. The cell-subsample RNG is also seeded from
``random_state``, so identical seeds produce identical networks
regardless of backend, modulo BLAS-level non-determinism in the SVD.

## Deprecated Alias

``n_cpus`` is kept as an alias for ``n_jobs`` during the ``0.2.x``
series and emits a ``DeprecationWarning``. It will be removed in
``0.3``.
