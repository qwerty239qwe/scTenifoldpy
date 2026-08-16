"""FastAPI app for the scTenifoldpy local web UI.

Route layout:
    GET  /api/datasets/example       generate a small synthetic X/Y dataset pair
    GET  /api/datasets/pbmc3k        real 10x PBMC3k data, QC-filtered + downsampled
    POST /api/datasets               upload a genes-by-cells expression CSV or .h5ad
    POST /api/jobs                   start a scTenifoldNet, scTenifoldKnk, or GRN-only run
    GET  /api/jobs/{id}              poll job status/stage
    GET  /api/jobs/{id}/result       ranked genes as JSON
    GET  /api/jobs/{id}/result.csv   ranked genes as a CSV download
    GET  /                           static single-page app
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from scTenifold.core._networks import anndata_to_dataframe
from scTenifold.data import get_test_df

from .jobs import DatasetNotFoundError, JobManager, JobNotFoundError
from .pbmc3k import load_pbmc3k
from .schemas import (
    DatasetInfo,
    EdgeResultRow,
    GeneResultRow,
    JobCreate,
    JobCreated,
    JobResult,
    JobStatus,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB (raised from 200 MB to fit .h5ad uploads)
UPLOAD_CHUNK_BYTES = 1024 * 1024
_TOO_LARGE = f"file too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"


def _dataset_info(dataset_id: str, name: str, df: pd.DataFrame) -> DatasetInfo:
    return DatasetInfo(
        dataset_id=dataset_id,
        name=name,
        n_genes=df.shape[0],
        n_cells=df.shape[1],
        gene_names=list(df.index.astype(str)),
    )


def _validate_expression_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("dataset is empty")
    if not df.index.is_unique:
        raise ValueError("gene names must be unique")
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"non-numeric column(s) found: {non_numeric}; expected a genes-by-cells count matrix"
        )
    return df


def _read_expression_csv(path: str) -> pd.DataFrame:
    """Parse an uploaded genes-by-cells CSV (gene names in the first column)."""
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse CSV: {exc}") from exc
    return _validate_expression_df(df)


def _read_h5ad(path: str) -> pd.DataFrame:
    """Parse an uploaded AnnData .h5ad file (genes in .var, cells in .obs)."""
    try:
        import anndata
    except ImportError as exc:
        raise ValueError(
            "reading .h5ad files needs the 'anndata' package: pip install \"scTenifoldpy[ui]\""
        ) from exc

    try:
        adata = anndata.read_h5ad(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse .h5ad file: {exc}") from exc

    df = anndata_to_dataframe(adata)
    df.index = df.index.astype(str)
    return _validate_expression_df(df)


async def _spool_upload(file: UploadFile, suffix: str) -> str:
    """Copy an upload to a temp file a chunk at a time, bailing out as soon as
    the running total passes the cap — so an oversized file is never held in
    memory whole (``await file.read()`` with no argument would do exactly that).
    Returns the temp file's path; the caller is responsible for removing it."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    total = 0
    try:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, _TOO_LARGE)
            tmp.write(chunk)
        tmp.close()
    except BaseException:
        tmp.close()
        os.unlink(tmp.name)
        raise
    return tmp.name


def create_app() -> FastAPI:
    app = FastAPI(title="scTenifoldpy", description="Local UI for the scTenifold suite")
    manager = JobManager()

    @app.middleware("http")
    async def reject_oversized_bodies(request: Request, call_next):
        """Reject on the declared Content-Length, before the multipart parser
        pulls the body off the socket. The endpoint can't do this: FastAPI has
        already read (and spooled) the whole upload by the time it runs, so a
        check there stops the memory blow-up but not the transfer."""
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            return JSONResponse({"detail": _TOO_LARGE}, status_code=413)
        return await call_next(request)

    @app.get("/api/datasets/example", response_model=list[DatasetInfo])
    def load_example_datasets():
        """Generate a small reproducible synthetic X/Y pair (see scTenifold.data.get_test_df)."""
        x_df = get_test_df(n_cells=200, n_genes=300, random_state=1)
        y_df = get_test_df(n_cells=200, n_genes=300, random_state=2)
        x_id = manager.add_dataset(x_df)
        y_id = manager.add_dataset(y_df)
        return [
            _dataset_info(x_id, "synthetic-X", x_df),
            _dataset_info(y_id, "synthetic-Y", y_df),
        ]

    @app.get("/api/datasets/pbmc3k", response_model=list[DatasetInfo])
    def load_pbmc3k_example():
        """Real 10x PBMC3k data (the Seurat/Scanpy tutorial dataset), QC-filtered
        and downsampled for a fast local demo, split into two random halves so
        it can also be used with the 'net' workflow."""
        try:
            df = load_pbmc3k()
        except ValueError as exc:
            raise HTTPException(502, str(exc)) from exc

        half = df.shape[1] // 2
        x_df, y_df = df.iloc[:, :half], df.iloc[:, half:]
        x_id = manager.add_dataset(x_df)
        y_id = manager.add_dataset(y_df)
        return [
            _dataset_info(x_id, "pbmc3k-A", x_df),
            _dataset_info(y_id, "pbmc3k-B", y_df),
        ]

    @app.post("/api/datasets", response_model=DatasetInfo)
    async def upload_dataset(file: UploadFile):
        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".csv", ".h5ad"):
            raise HTTPException(400, "only .csv or .h5ad files are supported")
        path = await _spool_upload(file, suffix)
        try:
            df = _read_h5ad(path) if suffix == ".h5ad" else _read_expression_csv(path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            os.unlink(path)

        dataset_id = manager.add_dataset(df)
        return _dataset_info(dataset_id, file.filename, df)

    @app.post("/api/jobs", response_model=JobCreated)
    def create_job(params: JobCreate):
        try:
            x_df = manager.get_dataset(params.dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(404, f"unknown dataset_id {params.dataset_id!r}") from exc

        if params.workflow == "net":
            if not params.dataset_id_y:
                raise HTTPException(400, "dataset_id_y is required for the 'net' workflow")
            try:
                manager.get_dataset(params.dataset_id_y)
            except DatasetNotFoundError as exc:
                raise HTTPException(404, f"unknown dataset_id_y {params.dataset_id_y!r}") from exc
        elif params.workflow == "knk":
            if not params.ko_genes:
                raise HTTPException(400, "ko_genes is required for the 'knk' workflow")
            gene_names = set(x_df.index.astype(str))
            missing = [g for g in params.ko_genes if g not in gene_names]
            if missing:
                raise HTTPException(400, f"knockout gene(s) not found in dataset: {missing}")
        # 'grn' only needs dataset_id, already validated above.

        job_id = manager.submit(params)
        return JobCreated(job_id=job_id)

    @app.get("/api/jobs/{job_id}", response_model=JobStatus)
    def get_job_status(job_id: str):
        try:
            job = manager.get_job(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(404, f"unknown job_id {job_id!r}") from exc
        return JobStatus(**job.to_status_dict())

    @app.get("/api/jobs/{job_id}/result", response_model=JobResult)
    def get_job_result(job_id: str):
        job = _require_finished_job(manager, job_id)
        df = job.result
        if job.params.workflow == "grn":
            rows = [
                EdgeResultRow(source=str(row["Source"]), target=str(row["Target"]), weight=float(row["Weight"]))
                for _, row in df.iterrows()
            ]
        else:
            rows = [
                GeneResultRow(
                    gene=str(row["Gene"]),
                    distance=float(row["Distance"]),
                    boxcox_distance=float(row["boxcox-transformed distance"]),
                    z=float(row["Z"]),
                    fc=float(row["FC"]),
                    p_value=float(row["p-value"]),
                    adjusted_p_value=float(row["adjusted p-value"]),
                )
                for _, row in df.iterrows()
            ]
        return JobResult(job_id=job_id, workflow=job.params.workflow, rows=rows)

    @app.get("/api/jobs/{job_id}/result.csv")
    def get_job_result_csv(job_id: str):
        job = _require_finished_job(manager, job_id)
        buf = io.StringIO()
        job.result.to_csv(buf, index=False)
        buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="sctenifold_{job_id}.csv"'}
        return StreamingResponse(buf, media_type="text/csv", headers=headers)

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


def _require_finished_job(manager: JobManager, job_id: str):
    try:
        job = manager.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, f"unknown job_id {job_id!r}") from exc
    if job.status == "error":
        raise HTTPException(500, f"job failed: {job.error}")
    if job.status != "done":
        raise HTTPException(409, f"job not finished yet (status={job.status}, stage={job.stage})")
    return job


app = create_app()
