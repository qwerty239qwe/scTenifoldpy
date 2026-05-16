"""Integration-contract tests for the optional scTenifoldXct re-export.

scTenifoldXct is maintained and tested in its own repo
(cailab-tamu/scTenifoldXct). Here we only assert that scTenifoldpy
re-exports it correctly and wires the CLI — not its algorithm.

Skipped automatically when the optional ``[xct]`` extra is absent, so
the default CI matrix stays green without torch/scanpy.
"""
import csv
from importlib.resources import files

import numpy as np
import pytest

pytest.importorskip("scTenifoldXct")
pytest.importorskip("torch")
pytest.importorskip("scanpy")

import anndata
import scanpy as sc
import scTenifoldXct as ext

from scTenifold import (scTenifoldXct, merge_scTenifoldXct,
                        set_seed, get_Xct_pairs, plot_XNet)


def test_reexport_is_identity():
    """scTenifoldpy must re-export the external package, not a copy."""
    assert scTenifoldXct is ext.scTenifoldXct
    assert merge_scTenifoldXct is ext.merge_scTenifoldXct
    assert set_seed is ext.set_seed
    assert get_Xct_pairs is ext.get_Xct_pairs
    assert plot_XNet is ext.plot_XNet


def test_cli_commands_registered():
    from scTenifold.__main__ import app

    names = {c.name for c in app.registered_commands}
    assert {"xct", "xct-merge"} <= names


def _lr_genes(n_pairs=15):
    db = files("scTenifoldXct") / "database"
    with (db / "LR.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    genes = []
    for r in rows[:n_pairs]:
        genes += [r["ligand"], r["receptor"]]
    seen, uniq = set(), []
    for g in genes:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq


def test_construction_smoke(tmp_path):
    """Minimal end-to-end construction to catch gross integration breaks."""
    rng = np.random.default_rng(0)
    genes = _lr_genes() + [f"FILLER{i}" for i in range(10)]
    counts = rng.poisson(2.0, size=(80, len(genes))).astype(float)
    ad = anndata.AnnData(counts)
    ad.var_names = genes
    ad.obs["ident"] = ["cell_A"] * 40 + ["cell_B"] * 40
    sc.pp.log1p(ad)
    ad.layers["log1p"] = ad.X.copy()

    obj = scTenifoldXct(
        data=ad,
        source_celltype="cell_A",
        target_celltype="cell_B",
        obs_label="ident",
        rebuild_GRN=True,
        GRN_file_dir=str(tmp_path),
        verbose=False,
    )
    from scTenifoldXct.core import GRN

    assert isinstance(obj._net_A, GRN)
    assert isinstance(obj._net_B, GRN)
    assert obj._net_A.shape == obj._net_B.shape
