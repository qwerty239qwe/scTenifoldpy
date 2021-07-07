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
    before_s = X.shape[1]
    X = X.loc[:, lib_size > min_lib_size]
    print(f"Removed {before_s - X.shape[1]} samples with lib size < {min_lib_size}")
    if remove_outlier_cells:
        lib_size = X.sum(axis=0)
        before_s = X.shape[1]
        Q3 = lib_size.to_frame().quantile(0.75, axis=0).values[0]
        Q1 = lib_size.to_frame().quantile(0.25, axis=0).values[0]
        interquartile_range = Q3 - Q1
        X = X.loc[:, (lib_size >= Q1 - interquartile_range * outlier_coef) &
                     (lib_size <= Q3 + interquartile_range * outlier_coef)]
        print(f"Removed {before_s - X.shape[1]} outlier samples from original data")
    mt_genes = X.index.str.upper().str.match("^MT-")
    if any(mt_genes):
        print(f"Found mitochondrial genes: {X[mt_genes].index.to_list()}")
        before_s = X.shape[1]
        mt_rates = X[mt_genes].sum(axis=0) / X.sum(axis=0)
        print(mt_rates)

        X = X.loc[:, mt_rates < max_MT_ratio]
        print(f"Removed {before_s - X.shape[1]} samples from original data (mt genes ratio > {max_MT_ratio})")
    else:
        warn("Mitochondrial genes were not found. Be aware that apoptotic cells may be present in your sample.")
    before_g = X.shape[0]
    X = X[(X != 0).mean(axis=1) > min_PCT]
    print(f"Removed {before_g - X.shape[0]} genes with average expression value < {min_PCT}")
    return X