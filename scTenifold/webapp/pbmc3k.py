"""Fetch, cache, and subset the classic 10x Genomics PBMC3k dataset.

This is the dataset used by both the Seurat (``pbmc3k``) and Scanpy
(``scanpy.datasets.pbmc3k``) getting-started tutorials: ~2,700 peripheral
blood mononuclear cells, ~32,700 genes, raw UMI counts from 10x Genomics.
It's downloaded directly from 10x Genomics and parsed with scipy (no
scanpy/anndata dependency needed), QC-filtered the same way as Seurat's
``CreateSeuratObject(min.cells=3, min.features=200)``, then subsetted to a
size that keeps the local UI demo fast (building PC networks from the full
~13,700 x 2,700 QC'd matrix would take minutes, not seconds).
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.io import mmread

PBMC3K_URL = "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
CACHE_DIR = Path.home() / ".cache" / "scTenifoldpy" / "pbmc3k"
MATRIX_DIR = CACHE_DIR / "filtered_gene_bc_matrices" / "hg19"

# Demo-friendly subset sizes; see module docstring.
N_GENES = 300
N_CELLS = 260
MIN_CELLS = 3  # Seurat's min.cells
MIN_FEATURES = 200  # Seurat's min.features


def _download_and_extract() -> Path:
    if MATRIX_DIR.is_dir():
        return MATRIX_DIR

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(PBMC3K_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                tmp.write(chunk)
            tmp.flush()
            with tarfile.open(tmp.name, "r:gz") as tf:
                tf.extractall(CACHE_DIR)  # noqa: S202 - trusted, pinned 10x Genomics URL
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not download the PBMC3k dataset: {exc}") from exc

    if not MATRIX_DIR.is_dir():
        raise ValueError("downloaded archive did not contain the expected matrix folder")
    return MATRIX_DIR


def _make_unique(names: pd.Series) -> pd.Index:
    """Disambiguate duplicate gene symbols, the way scanpy's var_names_make_unique does."""
    seen: dict = {}
    result = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            result.append(name)
        else:
            seen[name] += 1
            result.append(f"{name}-{seen[name]}")
    return pd.Index(result)


def _read_raw(matrix_dir: Path):
    genes = pd.read_csv(matrix_dir / "genes.tsv", sep="\t", header=None)
    gene_names = _make_unique(genes.iloc[:, 1])  # human-readable symbol, not the Ensembl id
    matrix = mmread(matrix_dir / "matrix.mtx").tocsr()  # genes x cells, matching 10x's convention
    return matrix, gene_names


def load_pbmc3k(n_genes: int = N_GENES, n_cells: int = N_CELLS, random_state: int = 0) -> pd.DataFrame:
    """Return a QC-filtered, downsampled PBMC3k genes-by-cells count matrix."""
    matrix_dir = _download_and_extract()
    matrix, gene_names = _read_raw(matrix_dir)

    gene_idx = np.flatnonzero(np.asarray(matrix.getnnz(axis=1)).ravel() >= MIN_CELLS)
    matrix, gene_names = matrix[gene_idx], gene_names[gene_idx]
    cell_idx = np.flatnonzero(np.asarray(matrix.getnnz(axis=0)).ravel() >= MIN_FEATURES)
    matrix = matrix[:, cell_idx]

    # Keep the most highly expressed genes and a random cell subsample so
    # the demo builds PC networks in seconds rather than minutes.
    total_counts = np.asarray(matrix.sum(axis=1)).ravel()
    top_gene_idx = np.argsort(total_counts)[::-1][:n_genes]
    matrix, gene_names = matrix[top_gene_idx], gene_names[top_gene_idx]

    if matrix.shape[1] > n_cells:
        rng = np.random.default_rng(random_state)
        cell_idx = np.sort(rng.choice(matrix.shape[1], size=n_cells, replace=False))
        matrix = matrix[:, cell_idx]
        cell_names = [f"pbmc3k-cell-{i}" for i in cell_idx]
    else:
        cell_names = [f"pbmc3k-cell-{i}" for i in range(matrix.shape[1])]

    return pd.DataFrame(matrix.toarray(), index=gene_names, columns=cell_names)
