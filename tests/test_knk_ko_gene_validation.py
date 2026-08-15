"""scTenifoldKnk should report a clear error when a requested knockout gene
isn't in the post-QC gene set, instead of leaking pandas'/numpy's raw
"None of [Index([...])] are in the [index]" KeyError.

A knockout gene can be a real, correctly-spelled column in the input and
still vanish before the knockout step runs: QC (min_lib_size / min_percent /
min_exp_avg / min_exp_sum) filters genes as well as cells. Covers both
ko_method branches ("default" and "propagation"), since each looks the gene
up a different way.
"""
import numpy as np
import pandas as pd
import pytest

from scTenifold import virtual_knockout
from scTenifold.data import get_test_df

QC_KWS = {"min_lib_size": 0, "min_percent": 0.001, "plot": False}


def _df_with_a_qc_dropped_gene(dropped_gene="Ccl22", n_genes=50, n_cells=200, random_state=0):
    """A dataset where `dropped_gene` is present but expressed in zero
    cells, so QC's min_percent filter removes it while every other gene
    survives."""
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(
        rng.poisson(5, size=(n_genes, n_cells)),
        index=[f"Gene{i}" for i in range(n_genes)],
        columns=[f"cell{j}" for j in range(n_cells)],
    )
    df.loc[dropped_gene] = 0
    return df


@pytest.mark.parametrize("ko_method", ["default", "propagation"])
def test_qc_dropped_ko_gene_reports_clear_error(ko_method):
    df = _df_with_a_qc_dropped_gene()
    assert "Ccl22" in df.index  # sanity: it really is in the raw input

    with pytest.raises(ValueError, match=r"removed by QC filtering before the knockout step: \['Ccl22'\]"):
        virtual_knockout(df, ko_genes=["Ccl22"], ko_method=ko_method, qc_kws=QC_KWS)


def test_ko_gene_never_in_input_is_reported_separately_from_qc_dropped():
    df = _df_with_a_qc_dropped_gene()

    with pytest.raises(ValueError, match=r"not found in the input data: \['NotAGene'\]"):
        virtual_knockout(df, ko_genes=["NotAGene"], qc_kws=QC_KWS)


def test_mixed_qc_dropped_and_unknown_genes_reports_both_reasons():
    df = _df_with_a_qc_dropped_gene()

    with pytest.raises(ValueError) as excinfo:
        virtual_knockout(df, ko_genes=["Ccl22", "NotAGene"], qc_kws=QC_KWS)
    message = str(excinfo.value)
    assert "removed by QC filtering before the knockout step: ['Ccl22']" in message
    assert "not found in the input data: ['NotAGene']" in message


def test_ko_gene_that_survives_qc_still_runs_normally():
    df = get_test_df(n_cells=100, n_genes=100, random_state=42)
    # get_test_df's synthetic "MT-*" genes can be low-expressed enough for
    # QC's min_exp_avg/min_exp_sum filter to drop them (as seen with MT-1
    # here); use a non-mitochondrial gene so this test is about the ko-gene
    # check rather than QC fixture quirks.
    survivor = next(g for g in df.index if not g.upper().startswith("MT-"))
    result = virtual_knockout(df, ko_genes=[survivor], qc_kws={"min_lib_size": 1, "plot": False})
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
