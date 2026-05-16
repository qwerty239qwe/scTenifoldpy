"""Tests for the vendored scTenifoldXct subpackage.

Skipped automatically when the optional ``[xct]`` dependencies are not
installed, so the default CI matrix stays green without torch/scanpy.
"""
import csv
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("scanpy")
pytest.importorskip("anndata")

import anndata
import scanpy as sc

from scTenifold import scTenifoldXct, merge_scTenifoldXct


def _lr_genes(n_pairs=15):
    """Pull real ligand/receptor symbols from the bundled DB so the
    candidate set is non-empty for a synthetic dataset."""
    db = files("scTenifold.xct") / "database"
    with (db / "LR.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    genes = []
    for r in rows[:n_pairs]:
        genes += [r["ligand"], r["receptor"]]
    # de-dupe, keep order
    seen, uniq = set(), []
    for g in genes:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq


@pytest.fixture(scope="module")
def adata():
    rng = np.random.default_rng(0)
    genes = _lr_genes() + [f"FILLER{i}" for i in range(10)]
    n_a, n_b = 40, 40
    counts = rng.poisson(2.0, size=(n_a + n_b, len(genes))).astype(float)
    ad = anndata.AnnData(counts)
    ad.var_names = genes
    ad.obs["ident"] = ["cell_A"] * n_a + ["cell_B"] * n_b
    sc.pp.log1p(ad)
    ad.layers["log1p"] = ad.X.copy()
    return ad


@pytest.fixture(scope="module")
def xct_obj(adata, tmp_path_factory):
    return scTenifoldXct(
        data=adata,
        source_celltype="cell_A",
        target_celltype="cell_B",
        obs_label="ident",
        rebuild_GRN=True,
        GRN_file_dir=str(tmp_path_factory.mktemp("grn")),
        verbose=False,
    )


def test_construction_builds_grns(xct_obj):
    from scTenifold.xct.core import GRN

    assert isinstance(xct_obj._net_A, GRN)
    assert isinstance(xct_obj._net_B, GRN)
    assert xct_obj._net_A.shape == xct_obj._net_B.shape


def test_embeds_and_stats(xct_obj):
    emb = xct_obj.get_embeds(train=True, n_steps=20)
    assert emb is not None
    assert isinstance(xct_obj.chi2_test(), pd.DataFrame)
    assert isinstance(xct_obj.null_test(), pd.DataFrame)


def test_merge_diff_test(adata, tmp_path):
    a = scTenifoldXct(data=adata, source_celltype="cell_A", target_celltype="cell_B",
                      obs_label="ident", rebuild_GRN=True,
                      GRN_file_dir=str(tmp_path / "a"), verbose=False)
    b = scTenifoldXct(data=adata, source_celltype="cell_B", target_celltype="cell_A",
                      obs_label="ident", rebuild_GRN=True,
                      GRN_file_dir=str(tmp_path / "b"), verbose=False)
    merged = merge_scTenifoldXct(a, b)
    emb = merged.get_embeds(train=True, n_steps=20)
    merged.nn_aligned_diff(emb)
    assert isinstance(merged.chi2_diff_test(), pd.DataFrame)


def test_cli_commands_registered():
    from scTenifold.__main__ import app

    names = {c.name for c in app.registered_commands}
    assert {"xct", "xct-merge"} <= names
