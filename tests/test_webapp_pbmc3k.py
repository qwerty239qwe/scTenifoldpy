"""Unit tests for scTenifold.webapp.pbmc3k (no network access).

Skipped automatically when the optional ``[ui]`` extra is absent, so the
default CI matrix stays green without fastapi/uvicorn. ``_download_and_extract``
is always monkeypatched/mocked here so these tests never hit the network.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

pytest.importorskip("fastapi")

from scTenifold.webapp import pbmc3k


def _write_fake_10x(tmp_path, n_genes=40, n_cells=60, random_state=0):
    """Write a tiny synthetic matrix/genes/barcodes triple in 10x's layout."""
    rng = np.random.default_rng(random_state)
    dense = rng.poisson(3, size=(n_genes, n_cells))
    dense[: n_genes // 4, :] = 0  # some genes never expressed -> fail min_cells
    dense[:, : n_cells // 4] = 0  # some cells with nothing -> fail min_features
    mmwrite(tmp_path / "matrix.mtx", sparse.csr_matrix(dense))
    with open(tmp_path / "genes.tsv", "w") as fh:
        for i in range(n_genes):
            fh.write(f"ENSG{i}\tGENE{i % (n_genes // 2)}\n")  # forces duplicate symbols
    with open(tmp_path / "barcodes.tsv", "w") as fh:
        for i in range(n_cells):
            fh.write(f"BARCODE-{i}\n")
    return tmp_path


def test_make_unique_disambiguates_duplicates():
    names = pd.Series(["A", "B", "A", "A", "C"])
    result = pbmc3k._make_unique(names)
    assert list(result) == ["A", "B", "A-1", "A-2", "C"]


def test_load_pbmc3k_filters_and_subsets(tmp_path, monkeypatch):
    # Real Seurat-style thresholds (>=200 genes/cell) need hundreds of genes
    # to be satisfiable; lower them so the tiny fixture can exercise the
    # same filtering code path without a huge synthetic matrix.
    monkeypatch.setattr(pbmc3k, "MIN_CELLS", 1)
    monkeypatch.setattr(pbmc3k, "MIN_FEATURES", 5)
    fake_dir = _write_fake_10x(tmp_path, n_genes=40, n_cells=60)
    monkeypatch.setattr(pbmc3k, "_download_and_extract", lambda: fake_dir)

    df = pbmc3k.load_pbmc3k(n_genes=10, n_cells=20, random_state=0)

    assert df.shape == (10, 20)
    assert df.index.is_unique
    assert (df.to_numpy() >= 0).all()


def test_load_pbmc3k_keeps_all_cells_when_fewer_than_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(pbmc3k, "MIN_CELLS", 1)
    monkeypatch.setattr(pbmc3k, "MIN_FEATURES", 5)
    fake_dir = _write_fake_10x(tmp_path, n_genes=40, n_cells=20)
    monkeypatch.setattr(pbmc3k, "_download_and_extract", lambda: fake_dir)

    df = pbmc3k.load_pbmc3k(n_genes=10, n_cells=1000, random_state=0)

    assert df.shape[0] == 10
    assert df.shape[1] <= 20  # never invents cells beyond what QC left


def test_download_and_extract_wraps_request_errors(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise OSError("network is down")

    monkeypatch.setattr(pbmc3k.requests, "get", _boom)
    monkeypatch.setattr(pbmc3k, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pbmc3k, "MATRIX_DIR", tmp_path / "cache" / "does-not-exist")

    with pytest.raises(ValueError, match="could not download"):
        pbmc3k._download_and_extract()
