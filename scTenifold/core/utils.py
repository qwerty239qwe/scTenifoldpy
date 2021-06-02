import numpy as np
import pandas as pd
from functools import wraps, partial
import time


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


def timer(func=None):
    if func is None:
        return partial(timer)

    @wraps(func)
    def _counter(*args, **kwargs):
        if not kwargs.get("verbosity") is None:
            verbosity = kwargs.pop("verbosity")
        else:
            verbosity = 1
        if verbosity >= 1:
            start = time.perf_counter()
        sol = func(*args, **kwargs)
        if verbosity >= 1:
            end = time.perf_counter()
            print(func.__name__, " processing time: ", str(end - start))
        return sol
    return _counter
