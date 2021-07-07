import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from functools import wraps, partial
import time
import re

__all__ = ("get_test_df", "cal_fdr", "read_mtx", "timer")


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


def _get_mtx_body(rows, print_header=True):
    find_header_btn, row_ptr = False, 0
    while not find_header_btn:
        m = re.match(r"\d*\s\d*\s\d*", rows[row_ptr].strip())
        if m is not None:
            find_header_btn = True
        row_ptr += 1
    header, body = rows[:row_ptr], rows[row_ptr:]
    if print_header:
        print(header)
    return body


def _build_matrix_from_sparse(sparse_data, shape):
    row, col, data = [], [], []
    for data_row in sparse_data:
        r, c, d = data_row.strip().split(" ")
        row.append(int(r) - 1)
        col.append(int(c) - 1)
        data.append(int(d))
    return csr_matrix((data, (row, col)), shape=shape).toarray()


def read_mtx(mtx_file_name, gene_file_name, barcode_file_name):
    genes = pd.read_csv(gene_file_name, sep='\t', header=None).iloc[:, 0]
    barcodes = pd.read_csv(barcode_file_name, sep='\t', header=None).iloc[:, 0]
    with open(mtx_file_name) as f:
        rows = f.readlines()
        body = _get_mtx_body(rows)
    print(f"creating a {(len(genes), len(barcodes))} matrix")
    data = _build_matrix_from_sparse(body, shape=(len(genes), len(barcodes)))
    df = pd.DataFrame(index=genes, columns=barcodes, data=data)
    return df