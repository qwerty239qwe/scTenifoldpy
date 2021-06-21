import numpy as np
import pandas as pd
from warnings import warn


def scQC(X: pd.DataFrame,
         min_lib_size: float = 1000,
         remove_outlier_cells: bool = True,
         min_PCT: float = 0.05,
         max_MT_ratio: float = 0.1):

    """Emulate scTenifoldNet's func"""

    outlier_coef = 1.5
    X[X < 0] = 0
    lib_size = X.sum(axis=0)
    X = X.loc[:, lib_size > min_lib_size]
    if remove_outlier_cells:
        medians = lib_size.to_frame().quantile(0.5, axis=0).values[0]
        interquartile_ranges = (lib_size.to_frame().quantile(0.75, axis=0) -
                               lib_size.to_frame().quantile(0.25, axis=0)).values[0]
        X = X.loc[:, (lib_size >= medians - interquartile_ranges * outlier_coef) &
                     (lib_size <= medians + interquartile_ranges * outlier_coef)]
    mt_genes = X.index.str.upper().str.match("^MT-")
    if any(mt_genes):
        mt_rates = X[mt_genes].sum(axis=0) / X.sum(axis=0)
        X = X.loc[:, mt_rates < max_MT_ratio]
    else:
        warn("Mitochondrial genes were not found. Be aware that apoptotic cells may be present in your sample.")
    X = X[X[X != 0].mean(axis=1) > min_PCT]
    return X