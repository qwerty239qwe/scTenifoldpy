# Installation

## Requirements

- Python **3.9 to 3.14** (declared via ``requires-python`` in
  ``pyproject.toml``).
- A scientific stack (``numpy``, ``scipy``, ``pandas``, ``scikit-learn``,
  ``tensorly``, ``networkx``, ``seaborn``), pulled in automatically.
- **No R runtime dependency.** This Python port does not call R; its
  principal-component network step uses
  ``sklearn.utils.extmath.randomized_svd``.

## Core Install

With uv:

```bash
uv venv
uv pip install scTenifoldpy
```

With pip:

```bash
pip install scTenifoldpy
```

## Optional Extras

| Extra | What it adds | Install with uv |
|---|---|---|
| ``scanpy`` | ``scanpy>=1.9`` for AnnData interop | ``uv pip install "scTenifoldpy[scanpy]"`` |
| ``parallel-ray`` | ``ray>=2`` for the ``"ray"`` network-construction backend | ``uv pip install "scTenifoldpy[parallel-ray]"`` |
| ``xct`` | the separately maintained ``scTenifoldXct`` package (pulls ``torch``, ``scanpy``, ``anndata``, ``ray``, ... transitively) for the cell-cell interaction workflow | ``uv pip install "scTenifoldpy[xct]"`` |
| ``docs`` | ``mkdocs-material`` and ``mkdocstrings`` for building these docs | ``uv pip install "scTenifoldpy[docs]"`` |

The equivalent pip commands are ``pip install "scTenifoldpy[scanpy]"``,
``pip install "scTenifoldpy[parallel-ray]"``,
``pip install "scTenifoldpy[xct]"``, and
``pip install "scTenifoldpy[docs]"``.

The ``xct`` extra requires **Python >= 3.10** (PyTorch / scanpy) and is
a re-export of the standalone ``scTenifoldXct`` package; the core
install keeps its 3.9-3.14 support and is never forced to import torch.

Joblib backends (``joblib-loky`` and ``joblib-threading``) need no extra
dependency; ``joblib`` is a transitive dependency of ``scikit-learn``.

## Development Install

The repository ships a ``uv.lock`` for reproducible environments:

```bash
git clone https://github.com/qwerty239qwe/scTenifoldpy
cd scTenifoldpy
uv sync --group test --group dev --group docs
```

Equivalent with pip:

```bash
pip install -e ".[scanpy,parallel-ray,docs]"
pip install pytest pytest-cov ruff build
```

Run the test suite:

```bash
uv run pytest
```

Build the docs:

```bash
uv run mkdocs build --strict
```

## Docker

Build the default runtime image:

```bash
docker build -t sctenifoldpy .
```

Run the CLI from the container, mounting the current directory as the
working directory:

```bash
docker run --rm -v "$PWD:/workspace" sctenifoldpy scTenifold --help
```

PowerShell:

```powershell
docker run --rm -v "${PWD}:/workspace" sctenifoldpy scTenifold --help
```

Optional extras can be included at build time:

```bash
docker build --build-arg EXTRAS=scanpy -t sctenifoldpy:scanpy .
docker build --build-arg EXTRAS=parallel-ray -t sctenifoldpy:ray .
```

## Troubleshooting

- **``ImportError: ray``**: install the ``parallel-ray`` extra, or pick
  a different backend (see [Parallel Backends](parallel-backends.md)).
- **Ray on Windows**: Ray's Windows support is best-effort. Use
  ``backend="joblib-loky"`` for process-based parallelism on Windows.
- **``ImportError`` from ``scipy.sparse.csr``**: upgrade scipy
  (``uv pip install -U scipy`` or ``pip install -U scipy``). The private
  submodule path was removed in recent scipy versions.
- **``umap`` not found** during plotting: install ``umap-learn``
  separately; it is not a hard dependency.
