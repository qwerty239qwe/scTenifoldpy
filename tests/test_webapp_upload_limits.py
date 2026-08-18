"""Upload size-limit tests for the scTenifold local web UI.

Covers both layers of the cap: the Content-Length middleware (rejects before
the body is read) and the chunked spooling in the upload endpoint (rejects a
body that outgrows the cap regardless of what Content-Length claimed).

Skipped automatically when the optional ``[ui]`` extra is absent, so the
default CI matrix stays green without fastapi/uvicorn.
"""
import asyncio
import io
import os
import tempfile

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from scTenifold.webapp import main
from scTenifold.webapp.main import create_app


@pytest.fixture()
def client():
    return TestClient(create_app())


def _csv_bytes(n_genes: int = 5, n_cells: int = 4) -> bytes:
    header = "gene," + ",".join(f"cell{j}" for j in range(n_cells))
    rows = [f"g{i}," + ",".join(str(i + j) for j in range(n_cells)) for i in range(n_genes)]
    return ("\n".join([header, *rows]) + "\n").encode()


def _temp_files() -> set:
    return set(os.listdir(tempfile.gettempdir()))


def _run(coro):
    """Drive one coroutine to completion (no pytest-asyncio in the test deps)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_oversized_upload_rejected_by_content_length(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 128)
    payload = _csv_bytes(n_genes=200)
    assert len(payload) > 128

    async def _fail(*args, **kwargs):
        raise AssertionError("middleware should reject before the endpoint runs")

    monkeypatch.setattr(main, "_spool_upload", _fail)

    resp = client.post("/api/datasets", files={"file": ("big.csv", payload, "text/csv")})
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"]


def test_spool_upload_stops_once_the_cap_is_passed(monkeypatch):
    # Exercises the endpoint-side guard directly: Content-Length can be absent
    # or wrong, so the byte counter is what actually bounds the write.
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr(main, "UPLOAD_CHUNK_BYTES", 16)
    upload = UploadFile(file=io.BytesIO(b"x" * 500), filename="big.csv")

    before = _temp_files()
    with pytest.raises(HTTPException) as excinfo:
        _run(main._spool_upload(upload, ".csv"))
    assert excinfo.value.status_code == 413
    # the partially written temp file must not survive the abort
    assert _temp_files() == before


def test_upload_spanning_multiple_chunks_round_trips(client, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_CHUNK_BYTES", 8)
    payload = _csv_bytes(n_genes=12, n_cells=6)

    resp = client.post("/api/datasets", files={"file": ("small.csv", payload, "text/csv")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_genes"] == 12
    assert resp.json()["n_cells"] == 6


def test_temp_file_removed_when_parsing_fails(client):
    before = _temp_files()
    resp = client.post(
        "/api/datasets", files={"file": ("bad.csv", b"not,a\nvalid,matrix\n", "text/csv")}
    )
    assert resp.status_code == 400
    assert _temp_files() == before
