import re

from scipy.sparse.csr import csr_matrix
import pandas as pd


__all__ = ["read_mtx"]


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


def read_mtx(mtx_file_name,
             gene_file_name,
             barcode_file_name) -> pd.DataFrame:
    """
    Read mtx data

    Parameters
    ----------
    mtx_file_name: str
        File name of mtx data
    gene_file_name
        File name of gene vector
    barcode_file_name
        File name of barcode vector

    Returns
    -------
    df: pd.DataFrame
        A dataframe with genes as rows and cells as columns
    """
    genes = pd.read_csv(gene_file_name, sep='\t', header=None).iloc[:, 0]
    barcodes = pd.read_csv(barcode_file_name, sep='\t', header=None).iloc[:, 0]
    with open(mtx_file_name) as f:
        rows = f.readlines()
        body = _get_mtx_body(rows)
    print(f"creating a {(len(genes), len(barcodes))} matrix")
    data = _build_matrix_from_sparse(body, shape=(len(genes), len(barcodes)))
    df = pd.DataFrame(index=genes, columns=barcodes, data=data)
    return df
