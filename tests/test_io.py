import zipfile
import pandas as pd
import pytest

from scTenifold.data._io import read_mtx, read_folder, _parse_mtx, _get_mtx_body


GENES = ["GeneA", "GeneB", "GeneC"]
BARCODES = ["c1", "c2"]
MTX_HEADER = "%%MatrixMarket matrix coordinate real general\n"
MTX_BODY_LINES = ["3 2 3\n", "1 1 5\n", "2 2 3\n", "3 1 7\n"]


def _write_genes_barcodes(d):
    (d / "genes.tsv").write_text("\n".join(GENES))
    (d / "barcodes.tsv").write_text("\n".join(BARCODES))


def test_get_mtx_body_no_decode():
    rows = ["%comment\n", "3 2 3\n", "1 1 5\n"]
    body, header = _get_mtx_body(rows, decode=None, print_header=False)
    assert header == ["3", "2", "3"]
    assert body == ["1 1 5\n"]


def test_parse_mtx_txt(tmp_path):
    p = tmp_path / "matrix.txt"
    p.write_text(MTX_HEADER + "".join(MTX_BODY_LINES))
    body, is_dense, n_rows, n_cols = _parse_mtx(str(p))
    assert is_dense is False
    assert (str(n_rows), str(n_cols)) == ("3", "2")


def test_parse_mtx_csv(tmp_path):
    p = tmp_path / "matrix.csv"
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    df.to_csv(p, header=False, index=False)
    body, is_dense, n_rows, n_cols = _parse_mtx(str(p))
    assert is_dense is True
    assert n_rows == 3 and n_cols == 2


def test_parse_mtx_tsv(tmp_path):
    p = tmp_path / "matrix.tsv"
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    df.to_csv(p, sep="\t", header=False, index=False)
    body, is_dense, n_rows, n_cols = _parse_mtx(str(p))
    assert is_dense is True
    assert n_rows == 3 and n_cols == 2


def test_parse_mtx_invalid_suffix(tmp_path):
    p = tmp_path / "matrix.bogus"
    p.write_text("x")
    with pytest.raises(ValueError, match="suffix"):
        _parse_mtx(str(p))


def test_parse_mtx_zip_with_mtx(tmp_path):
    inner = tmp_path / "inner.txt"
    inner.write_text(MTX_HEADER + "".join(MTX_BODY_LINES))
    zp = tmp_path / "matrix.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(inner, arcname="inner.txt")
    body, is_dense, n_rows, n_cols = _parse_mtx(str(zp))
    assert is_dense is False


def test_parse_mtx_zip_with_csv(tmp_path):
    inner = tmp_path / "inner.csv"
    pd.DataFrame([["", "c1", "c2"], ["g1", 1, 2], ["g2", 3, 4]]).to_csv(
        inner, header=False, index=False
    )
    zp = tmp_path / "matrix.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(inner, arcname="inner.csv")
    body, is_dense, n_rows, n_cols = _parse_mtx(str(zp))
    assert is_dense is True


def test_parse_mtx_zip_with_tsv(tmp_path):
    inner = tmp_path / "inner.tsv"
    pd.DataFrame([["", "c1", "c2"], ["g1", 1, 2], ["g2", 3, 4]]).to_csv(
        inner, header=False, index=False, sep="\t"
    )
    zp = tmp_path / "matrix.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(inner, arcname="inner.tsv")
    body, is_dense, n_rows, n_cols = _parse_mtx(str(zp))
    assert is_dense is True


def test_read_mtx_txt(tmp_path):
    _write_genes_barcodes(tmp_path)
    mp = tmp_path / "matrix.txt"
    mp.write_text(MTX_HEADER + "".join(MTX_BODY_LINES))
    df = read_mtx(str(mp), str(tmp_path / "genes.tsv"), str(tmp_path / "barcodes.tsv"))
    assert df.shape == (3, 2)
    assert df.loc["GeneA", "c1"] == 5


def test_read_mtx_csv_dense(tmp_path):
    _write_genes_barcodes(tmp_path)
    mp = tmp_path / "matrix.csv"
    pd.DataFrame([[1, 2], [3, 4], [5, 6]]).to_csv(mp, header=False, index=False)
    df = read_mtx(str(mp), str(tmp_path / "genes.tsv"), str(tmp_path / "barcodes.tsv"))
    assert df.shape == (3, 2)


def test_read_mtx_missing_barcodes_warns_csv(tmp_path):
    (tmp_path / "genes.tsv").write_text("\n".join(GENES))
    mp = tmp_path / "matrix.csv"
    pd.DataFrame([[1, 2], [3, 4], [5, 6]]).to_csv(mp, header=False, index=False)
    with pytest.warns(UserWarning, match="Barcode file"):
        df = read_mtx(str(mp), str(tmp_path / "genes.tsv"), None)
    assert list(df.columns) == ["barcode_0", "barcode_1"]


def test_read_mtx_missing_barcodes_warns_txt(tmp_path):
    (tmp_path / "genes.tsv").write_text("\n".join(GENES))
    mp = tmp_path / "matrix.txt"
    mp.write_text(MTX_HEADER + "".join(MTX_BODY_LINES))
    with pytest.warns(UserWarning, match="Barcode file"):
        df = read_mtx(str(mp), str(tmp_path / "genes.tsv"), None)
    assert list(df.columns) == ["barcode_0", "barcode_1"]


def test_read_folder(tmp_path):
    _write_genes_barcodes(tmp_path)
    mp = tmp_path / "matrix.txt"
    mp.write_text(MTX_HEADER + "".join(MTX_BODY_LINES))
    df = read_folder(str(tmp_path))
    assert df.shape == (3, 2)


def test_read_folder_from_other_cwd(tmp_path, monkeypatch):
    _write_genes_barcodes(tmp_path)
    mp = tmp_path / "matrix.txt"
    mp.write_text(MTX_HEADER + "".join(MTX_BODY_LINES))
    other = tmp_path.parent
    monkeypatch.chdir(other)
    df = read_folder(str(tmp_path))
    assert df.shape == (3, 2)


def test_read_folder_not_a_dir(tmp_path):
    p = tmp_path / "nope.txt"
    p.write_text("x")
    with pytest.raises(ValueError, match="not exist"):
        read_folder(str(p))
