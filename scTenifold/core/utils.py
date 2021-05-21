import numpy as np
import pandas as pd


def get_test_df(n_cells = 100, n_genes = 1000):
    data = np.random.default_rng().negative_binomial(20, 0.98,
                                                     n_cells * n_genes).reshape(n_genes, n_cells)
    pseudo_gene_names = ["MT-{}".format(i) for i in range(1, 11)] + ["NG-{}".format(i) for i in range(1, n_genes - 9)]
    pseudo_cell_names = ["Cell-{}".format(i) for i in range(1, n_cells + 1)]
    return pd.DataFrame(data, index=pseudo_gene_names, columns=pseudo_cell_names)


def cal_fdr(p_vals):
    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr