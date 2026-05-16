from __future__ import annotations

import logging

import anndata

logger = logging.getLogger(__name__)


def get_cko_data(data: anndata.AnnData, cko_gene_names: list[str] | str) -> anndata.AnnData:
    """
    Return a copy of ``data`` with the conditional-knockout genes zeroed out.

    Gene names are upper-cased to match the convention used throughout the
    package (scTenifoldXct and the LR/TF databases). The input AnnData is not
    modified in place.
    """
    if isinstance(cko_gene_names, str):
        cko_gene_names = [cko_gene_names.upper()]
    else:
        cko_gene_names = [gene.upper() for gene in cko_gene_names]

    data = data.copy()
    data.var_names = data.var_names.str.upper()
    gene_index = [data.var_names.get_loc(gene_name) for gene_name in cko_gene_names]
    data.X[:, gene_index] = 0.
    logger.info(f"CKO genes {cko_gene_names} are set")

    return data
