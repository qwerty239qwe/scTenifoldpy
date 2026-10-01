"""tensor_decomp and cp_decomposition accept the networks as a list of slices,
which the pipeline uses to avoid holding the stacked tensor as well."""
import numpy as np
import pandas as pd
import pytest

from scTenifold import scTenifoldKnk
from scTenifold.core._decomposition import cp_decomposition, tensor_decomp


def _networks(n_genes=40, n_nets=4):
    rng = np.random.default_rng(1)
    nets = []
    for _ in range(n_nets):
        A = rng.normal(size=(n_genes, n_genes))
        A[np.abs(A) < 1] = 0
        nets.append(A)
    return nets


def test_list_of_slices_gives_the_same_decomposition():
    nets = _networks()
    genes = [f"g{i}" for i in range(nets[0].shape[0])]
    stacked = tensor_decomp(np.stack(nets, axis=-1), genes, K=3, n_decimal=3)
    listed = tensor_decomp(nets, genes, K=3, n_decimal=3)
    pd.testing.assert_frame_equal(stacked, listed)
    a = cp_decomposition(np.stack(nets, axis=-1), K=3)
    b = cp_decomposition(nets, K=3)
    np.testing.assert_array_equal(a["slice_sum"], b["slice_sum"])
    assert a["all_resids"] == b["all_resids"]


def test_tensorly_method_accepts_a_list_of_slices():
    pytest.importorskip("tensorly")
    nets = _networks()
    genes = [f"g{i}" for i in range(nets[0].shape[0])]
    pd.testing.assert_frame_equal(tensor_decomp(np.stack(nets, axis=-1), genes, method="parafac", K=3),
                                  tensor_decomp(nets, genes, method="parafac", K=3))


def test_slices_of_different_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        cp_decomposition([np.zeros((3, 3)), np.zeros((3, 4))], K=2)
    with pytest.raises(ValueError, match="3-mode"):
        cp_decomposition(np.zeros((3, 3)), K=2)


def test_qc_does_not_plot_by_default(monkeypatch):
    import scTenifold.core._base as base

    def fail(*args, **kwargs):
        raise AssertionError("plot_hist was called")

    monkeypatch.setattr(base, "plot_hist", fail)
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.poisson(5, size=(30, 200)), index=[f"g{i}" for i in range(30)],
                      columns=[f"c{j}" for j in range(200)])
    sc = scTenifoldKnk(df, ko_genes=["g1"], qc_kws={"min_lib_size": 0})
    sc.run_step("qc")
    sc = scTenifoldKnk(df, ko_genes=["g1"], qc_kws={"min_lib_size": 0, "plot": True})
    with pytest.raises(AssertionError, match="plot_hist was called"):
        sc.run_step("qc")
