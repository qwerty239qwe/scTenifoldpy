"""FastAPI app for the scTenifoldpy local web UI.

Route layout:
    GET  /api/datasets/example       generate a small synthetic X/Y dataset pair
    POST /api/datasets               upload a genes-by-cells expression CSV
    POST /api/jobs                   start a scTenifoldNet or scTenifoldKnk run
    GET  /api/jobs/{id}              poll job status/stage
    GET  /api/jobs/{id}/result       ranked genes as JSON
    GET  /api/jobs/{id}/result.csv   ranked genes as a CSV download
    GET  /                           static single-page app
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from scTenifold.data import get_test_df

from .jobs import DatasetNotFoundError, JobManager, JobNotFoundError
from .schemas import (
    DatasetInfo,
    GeneResultRow,
    JobCreate,
    JobCreated,
    JobResult,
    JobStatus,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


def _dataset_info(dataset_id: str, name: str, df: pd.DataFrame) -> DatasetInfo:
    return DatasetInfo(
        dataset_id=dataset_id,
        name=name,
        n_genes=df.shape[0],
        n_cells=df.shape[1],
        gene_names=list(df.index.astype(str)),
    )


def _read_expression_csv(raw: bytes) -> pd.DataFrame:
    """Parse an uploaded genes-by-cells CSV (gene names in the first column)."""
    try:
        df = pd.read_csv(io.BytesIO(raw), index_col=0)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse CSV: {exc}") from exc
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("dataset is empty")
    if not df.index.is_unique:
        raise ValueError("gene names (first column) must be unique")
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"non-numeric column(s) found: {non_numeric}; expected a genes-by-cells count matrix"
        )
    return df


def create_app() -> FastAPI:
    app = FastAPI(title="scTenifoldpy", description="Local UI for the scTenifold suite")
    manager = JobManager()

    # -- datasets -----------------------------------------------------
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

    @app.post("/api/datasets", response_model=DatasetInfo)
    async def upload_dataset(file: UploadFile):
        if not file.filename.endswith(".csv"):
            raise HTTPException(400, "only .csv files are supported (genes as rows, cells as columns)")
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large (limit 200 MB)")
        try:
            df = _read_expression_csv(raw)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        dataset_id = manager.add_dataset(df)
        return _dataset_info(dataset_id, file.filename, df)

    # -- jobs -----------------------------------------------------------
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
        else:
            if not params.ko_genes:
                raise HTTPException(400, "ko_genes is required for the 'knk' workflow")
            gene_names = set(x_df.index.astype(str))
            missing = [g for g in params.ko_genes if g not in gene_names]
            if missing:
                raise HTTPException(400, f"knockout gene(s) not found in dataset: {missing}")

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

    # -- static frontend --------------------------------------------
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
