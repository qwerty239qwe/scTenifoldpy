import re
from pathlib import Path
import zipfile

from scipy.sparse.csr import csr_matrix
import pandas as pd


__all__ = ["read_mtx"]


def _get_mtx_body(rows, decode=None, print_header=True):
    find_header_btn, row_ptr = False, 0
    while not find_header_btn:
        m = re.match(r"\d*\s\d*\s\d*", rows[row_ptr].strip())
        if m is not None:
            find_header_btn = True
        row_ptr += 1
    if decode is None:
        header, body = rows[:row_ptr], rows[row_ptr:]
    else:
        header, body = rows[:row_ptr].decode(decode), rows[row_ptr:].decode(decode)
    if print_header:
        print(header)
    return body


def _build_matrix_from_sparse(sparse_data, shape):
    row, col, data = [], [], []
    for data_row in sparse_data:
        r, c, d = data_row.strip().split(" ")
        row.append(int(r) - 1)
        col.append(int(c) - 1)
        data.append(float(d))
    return csr_matrix((data, (row, col)), shape=shape).toarray()


def _parse_mtx(mtx_file_name):
    suffix = Path(mtx_file_name).suffix
    if suffix == ".txt":
        with open(mtx_file_name) as f:
            rows = f.readlines()
            body = _get_mtx_body(rows)
        is_dense = False
    elif suffix == ".tsv":
        body = pd.read_csv(mtx_file_name, sep='\t', header=None, index=False).values
        is_dense = True
    elif suffix == ".csv":
        body = pd.read_csv(mtx_file_name, header=None, index=False).values
        is_dense = True
    elif suffix == ".zip":
        archive = zipfile.ZipFile(mtx_file_name, 'r')
        with archive.open(archive.namelist()[0]) as fn:
            rows = fn.readlines()
            body = _get_mtx_body(rows, decode="utf-8")
        is_dense = False
    else:
        raise ValueError("The suffix of this file is not valid")
    return body, is_dense


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
    body, is_dense = _parse_mtx(mtx_file_name)
    print(f"creating a {(len(genes), len(barcodes))} matrix")
    if not is_dense:
        data = _build_matrix_from_sparse(body, shape=(len(genes), len(barcodes)))
    else:
        data = body
    df = pd.DataFrame(index=genes, columns=barcodes, data=data)
    return df
