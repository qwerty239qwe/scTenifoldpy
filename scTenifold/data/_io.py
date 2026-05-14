import re
from pathlib import Path
import zipfile
from typing import Optional, Union
from warnings import warn

from scipy.sparse import csr_matrix
import pandas as pd


__all__ = ["read_mtx", "read_folder"]


def _get_mtx_body(rows, decode=None, print_header=True):
    find_header_btn, row_ptr = False, 0
    while not find_header_btn:
        m = re.match(r"\d*\s\d*\s\d*", rows[row_ptr].strip()
            if decode is None else rows[row_ptr].decode(decode).strip())
        if m is not None:
            find_header_btn = True
        row_ptr += 1
    if decode is None:
        header, body = rows[:row_ptr], rows[row_ptr:]
    else:
        header, body = [r.decode(decode) for r in rows[:row_ptr]], [r.decode(decode) for r in rows[row_ptr:]]
    if print_header:
        print(header)
    return body, header[-1].strip().split(" ")


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
            body, header = _get_mtx_body(rows)
            n_rows, n_cols = int(header[0]), int(header[1])
        is_dense = False
    elif suffix == ".tsv":
        body = pd.read_csv(mtx_file_name, sep='\t', header=None, index_col=False).values
        n_rows, n_cols = body.shape
        is_dense = True
    elif suffix == ".csv":
        body = pd.read_csv(mtx_file_name, header=None, index_col=False).values
        n_rows, n_cols = body.shape
        is_dense = True
    elif suffix == ".zip":
        archive = zipfile.ZipFile(mtx_file_name, 'r')
        with archive.open(archive.namelist()[0]) as fn:
            sf = Path(archive.namelist()[0]).suffix
            if sf not in [".csv", ".tsv"]:
                rows = fn.readlines()
                body, header = _get_mtx_body(rows, decode="utf-8")
                n_rows, n_cols = int(header[0]), int(header[1])
                is_dense = False
            else:
                body = pd.DataFrame([f.decode("utf-8").strip().split("," if sf == ".csv" else "\t")
                                     for f in fn.readlines()]).iloc[1:, 1:].values
                n_rows, n_cols = body.shape
                is_dense = True
    else:
        raise ValueError("The suffix of this file is not valid")
    return body, is_dense, n_rows, n_cols


def read_mtx(mtx_file_name: Union[str, Path],
             gene_file_name: Union[str, Path],
             barcode_file_name: Optional[Union[str, Path]]) -> pd.DataFrame:
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
    if mtx_file_name is None:
        raise ValueError("matrix file is required")
    if gene_file_name is None:
        raise ValueError("gene file is required")
    genes = pd.read_csv(gene_file_name, sep='\t', header=None).iloc[:, 0]
    barcodes = pd.read_csv(barcode_file_name, sep='\t', header=None).iloc[:, 0] \
        if barcode_file_name is not None else None
    if barcodes is None:
        warn("Barcode file is not existed. Added fake barcode name in the dataset")
    body, is_dense, n_rows, n_cols = _parse_mtx(mtx_file_name)
    barcodes = barcodes if barcodes is not None else [f"barcode_{i}" for i in range(n_cols)]
    print(f"creating a {(len(genes), len(barcodes))} matrix")
    if not is_dense:
        data = _build_matrix_from_sparse(body, shape=(len(genes), len(barcodes)))
    else:
        data = body
    df = pd.DataFrame(index=genes, columns=barcodes, data=data)
    return df


def read_folder(file_dir: Union[str, Path],
                matrix_fn: str = "matrix",
                gene_fn: str = "genes",
                barcodes_fn: str = "barcodes") -> pd.DataFrame:
    """Read mtx + genes + barcodes from a directory by filename substring.

    Parameters
    ----------
    file_dir
        Path to a directory containing matrix, gene, and barcode files.
    matrix_fn
        Substring identifying the matrix file (e.g. ``"matrix"``).
    gene_fn
        Substring identifying the gene file.
    barcodes_fn
        Substring identifying the barcode file.

    Returns
    -------
    Genes-by-cells DataFrame.
    """
    dir_path = Path(file_dir)
    fn_dic = {fn: [] for fn in [matrix_fn, gene_fn, barcodes_fn]}
    if not dir_path.is_dir():
        raise ValueError("Path is not exist or is not a folder path")
    for fn in dir_path.iterdir():
        for k in fn_dic:
            if k in fn.name:
                fn_dic[k].append(fn)

    resolved = {}
    for key, matches in fn_dic.items():
        if len(matches) > 1:
            raise ValueError(f"Multiple files match {key!r}: {[match.name for match in matches]}")
        resolved[key] = matches[0] if matches else None

    matrix_path = resolved[matrix_fn]
    gene_path = resolved[gene_fn]
    barcode_path = resolved[barcodes_fn]
    if matrix_path is None:
        raise ValueError("matrix file is required")
    if gene_path is None:
        raise ValueError("gene file is required")

    return read_mtx(mtx_file_name=str(matrix_path),
                    gene_file_name=str(gene_path),
                    barcode_file_name=str(barcode_path) if barcode_path else None)
