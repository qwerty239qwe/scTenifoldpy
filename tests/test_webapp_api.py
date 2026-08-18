"""HTTP-level tests for the scTenifold local web UI's FastAPI app.

Skipped automatically when the optional ``[ui]`` extra is absent, so the
default CI matrix stays green without fastapi/uvicorn.
"""
import io
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from scTenifold.webapp.main import create_app


@pytest.fixture()
def client():
    return TestClient(create_app())


def _run_job_to_completion(client, payload, timeout: float = 240.0):
    resp = client.post("/api/jobs", json=payload)
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    deadline = time.monotonic() + timeout
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "done", status
    return job_id


def test_serves_static_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "scTenifoldpy" in resp.text


def test_example_datasets_endpoint(client):
    resp = client.get("/api/datasets/example")
    assert resp.status_code == 200
    datasets = resp.json()
    assert len(datasets) == 2
    assert all(d["n_genes"] > 0 and d["n_cells"] > 0 for d in datasets)


def test_upload_dataset_rejects_unsupported_extension(client):
    resp = client.post(
        "/api/datasets", files={"file": ("data.txt", io.BytesIO(b"not a csv"), "text/plain")}
    )
    assert resp.status_code == 400
    assert "only .csv or .h5ad" in resp.json()["detail"]


def test_upload_dataset_rejects_corrupt_h5ad(client):
    resp = client.post(
        "/api/datasets", files={"file": ("data.h5ad", io.BytesIO(b"not an h5ad file"), "application/octet-stream")}
    )
    assert resp.status_code == 400
    assert "could not parse .h5ad" in resp.json()["detail"]


def test_upload_dataset_rejects_non_numeric_columns(client):
    csv_bytes = b"gene,cell1,cell2\nG1,1,label\nG2,2,3\n"
    resp = client.post("/api/datasets", files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")})
    assert resp.status_code == 400
    assert "non-numeric" in resp.json()["detail"]

def test_upload_dataset_accepts_valid_csv(client):
    csv_bytes = b"gene,cell1,cell2,cell3\nG1,1,2,3\nG2,4,5,6\n"
    resp = client.post("/api/datasets", files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")})
    assert resp.status_code == 200
    info = resp.json()
    assert info["n_genes"] == 2
    assert info["n_cells"] == 3
    assert info["gene_names"] == ["G1", "G2"]


def test_upload_dataset_accepts_valid_h5ad(client, tmp_path):
    anndata = pytest.importorskip("anndata")
    import numpy as np
    import pandas as pd

    # AnnData convention: X is cells (obs) x genes (var). Force classic
    # object-dtype string indices: some pandas builds default
    # 'future.infer_string' to True, which produces a nullable StringArray
    # index that this anndata version can't write (independent of the
    # actual product code path being tested here).
    with pd.option_context("future.infer_string", False):
        adata = anndata.AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        adata.obs_names = ["cellA", "cellB", "cellC"]
        adata.var_names = ["G1", "G2"]
        h5ad_path = tmp_path / "data.h5ad"
        adata.write_h5ad(h5ad_path)

    with open(h5ad_path, "rb") as fh:
        resp = client.post("/api/datasets", files={"file": ("data.h5ad", fh, "application/octet-stream")})
    assert resp.status_code == 200, resp.text
    info = resp.json()
    # anndata_to_dataframe transposes: genes (var_names) become rows, cells (obs_names) become columns.
    assert info["n_genes"] == 2
    assert info["n_cells"] == 3
    assert info["gene_names"] == ["G1", "G2"]


def test_pbmc3k_endpoint_uses_loader_and_splits_in_half(client, monkeypatch):
    import pandas as pd

    fake_df = pd.DataFrame(
        {f"cell{i}": [1, 2, 3] for i in range(6)}, index=["MS4A1", "CD3D", "LYZ"]
    )
    monkeypatch.setattr("scTenifold.webapp.main.load_pbmc3k", lambda: fake_df)

    resp = client.get("/api/datasets/pbmc3k")
    assert resp.status_code == 200
    datasets = resp.json()
    assert len(datasets) == 2
    assert datasets[0]["n_cells"] == 3
    assert datasets[1]["n_cells"] == 3
    assert datasets[0]["gene_names"] == ["MS4A1", "CD3D", "LYZ"]


def test_pbmc3k_endpoint_surfaces_download_failures(client, monkeypatch):
    def _boom():
        raise ValueError("could not download the PBMC3k dataset: network is down")

    monkeypatch.setattr("scTenifold.webapp.main.load_pbmc3k", _boom)

    resp = client.get("/api/datasets/pbmc3k")
    assert resp.status_code == 502
    assert "could not download" in resp.json()["detail"]


def test_create_job_unknown_dataset_returns_404(client):
    resp = client.post("/api/jobs", json={"workflow": "knk", "dataset_id": "missing", "ko_genes": ["G1"]})
    assert resp.status_code == 404


def test_create_net_job_requires_dataset_id_y(client):
    datasets = client.get("/api/datasets/example").json()
    resp = client.post("/api/jobs", json={"workflow": "net", "dataset_id": datasets[0]["dataset_id"]})
    assert resp.status_code == 400
    assert "dataset_id_y" in resp.json()["detail"]


def test_create_knk_job_requires_ko_genes(client):
    datasets = client.get("/api/datasets/example").json()
    resp = client.post("/api/jobs", json={"workflow": "knk", "dataset_id": datasets[0]["dataset_id"]})
    assert resp.status_code == 400
    assert "ko_genes" in resp.json()["detail"]


def test_create_knk_job_rejects_unknown_gene(client):
    datasets = client.get("/api/datasets/example").json()
    resp = client.post(
        "/api/jobs",
        json={"workflow": "knk", "dataset_id": datasets[0]["dataset_id"], "ko_genes": ["NOT-A-GENE"]},
    )
    assert resp.status_code == 400
    assert "NOT-A-GENE" in resp.json()["detail"]


def test_result_endpoints_reject_unfinished_or_unknown_job(client):
    resp = client.get("/api/jobs/does-not-exist/result")
    assert resp.status_code == 404

    datasets = client.get("/api/datasets/example").json()
    resp = client.post(
        "/api/jobs",
        json={
            "workflow": "net",
            "dataset_id": datasets[0]["dataset_id"],
            "dataset_id_y": datasets[1]["dataset_id"],
        },
    )
    job_id = resp.json()["job_id"]
    resp = client.get(f"/api/jobs/{job_id}/result")
    assert resp.status_code == 409


def test_net_job_full_round_trip(client):
    datasets = client.get("/api/datasets/example").json()
    job_id = _run_job_to_completion(
        client,
        {
            "workflow": "net",
            "dataset_id": datasets[0]["dataset_id"],
            "dataset_id_y": datasets[1]["dataset_id"],
            "min_lib_size": 10,
        },
    )

    resp = client.get(f"/api/jobs/{job_id}/result")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) > 0
    assert {"gene", "distance", "z", "fc", "p_value", "adjusted_p_value"} <= rows[0].keys()

    resp = client.get(f"/api/jobs/{job_id}/result.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert resp.text.splitlines()[0].startswith("Gene,")


def test_knk_job_full_round_trip(client):
    datasets = client.get("/api/datasets/example").json()
    job_id = _run_job_to_completion(
        client,
        {
            "workflow": "knk",
            "dataset_id": datasets[0]["dataset_id"],
            "ko_genes": ["NG-1"],
            "min_lib_size": 10,
        },
    )

    resp = client.get(f"/api/jobs/{job_id}/result")
    assert resp.status_code == 200
    assert len(resp.json()["rows"]) > 0


def test_grn_job_full_round_trip(client):
    datasets = client.get("/api/datasets/example").json()
    job_id = _run_job_to_completion(
        client,
        {
            "workflow": "grn",
            "dataset_id": datasets[0]["dataset_id"],
            "min_lib_size": 10,
        },
    )

    resp = client.get(f"/api/jobs/{job_id}/result")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) > 0
    assert {"source", "target", "weight"} <= rows[0].keys()

    resp = client.get(f"/api/jobs/{job_id}/result.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert resp.text.splitlines()[0] == "Source,Target,Weight"


def test_grn_job_does_not_require_dataset_id_y_or_ko_genes(client):
    datasets = client.get("/api/datasets/example").json()
    resp = client.post(
        "/api/jobs", json={"workflow": "grn", "dataset_id": datasets[0]["dataset_id"]}
    )
    assert resp.status_code == 200, resp.text
