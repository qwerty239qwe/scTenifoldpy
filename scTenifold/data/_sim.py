import pandas as pd
import numpy as np


def get_test_df(n_cells: int = 100,
                n_genes: int = 1000):
    data = np.random.default_rng().negative_binomial(20, 0.98,
                                                     n_cells * n_genes).reshape(n_genes, n_cells)
    pseudo_gene_names = ["MT-{}".format(i) for i in range(1, 11)] + ["NG-{}".format(i) for i in range(1, n_genes - 9)]
    pseudo_cell_names = ["Cell-{}".format(i) for i in range(1, n_cells + 1)]
    return pd.DataFrame(data, index=pseudo_gene_names, columns=pseudo_cell_names)