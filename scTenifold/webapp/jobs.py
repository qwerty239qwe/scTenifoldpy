"""In-process job manager for the scTenifold web UI.

scTenifoldNet/scTenifoldKnk network construction, tensor decomposition and
manifold alignment are CPU-bound and (with the default serial backend)
single-threaded. This is a local, single-user tool, so runs are executed on
a single-worker background thread pool and tracked by job id rather than run
concurrently. Everything lives in memory — no persistence across process
restarts.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from scTenifold import compare_networks, virtual_knockout

from .schemas import JobCreate

logger = logging.getLogger(__name__)

# Stages surfaced to the UI as a simple progress indicator. compare_networks
# / virtual_knockout run QC, network construction, decomposition and
# alignment as one call, so there is no finer-grained progress to report.
STAGE_QUEUED = "queued"
STAGE_RUNNING = "building networks, decomposing tensors, aligning manifolds"
STAGE_DONE = "done"


@dataclass
class Job:
    id: str
    params: JobCreate
    status: str = "queued"  # queued | running | done | error
    stage: str = STAGE_QUEUED
    error: Optional[str] = None
    result: Optional[pd.DataFrame] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self.stage = stage

    def to_status_dict(self) -> dict:
        with self._lock:
            return {
                "job_id": self.id,
                "status": self.status,
                "stage": self.stage,
                "error": self.error,
            }


class DatasetNotFoundError(KeyError):
    pass


class JobNotFoundError(KeyError):
    pass


class JobManager:
    """Holds uploaded datasets and runs scTenifold jobs on a single worker thread."""

    def __init__(self) -> None:
        self._datasets: dict[str, pd.DataFrame] = {}
        self._jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sctenifold-job")

    # -- datasets ---------------------------------------------------
    def add_dataset(self, df: pd.DataFrame) -> str:
        dataset_id = uuid.uuid4().hex[:12]
        self._datasets[dataset_id] = df
        return dataset_id

    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            raise DatasetNotFoundError(dataset_id) from exc

    # -- jobs ---------------------------------------------------------
    def submit(self, params: JobCreate) -> str:
        x_df = self.get_dataset(params.dataset_id)  # fail fast if unknown dataset
        y_df = self.get_dataset(params.dataset_id_y) if params.dataset_id_y else None
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, params=params)
        self._jobs[job_id] = job
        self._executor.submit(self._run, job, x_df, y_df)
        return job_id

    def get_job(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    def _run(self, job: Job, x_df: pd.DataFrame, y_df: Optional[pd.DataFrame]) -> None:
        job.status = "running"
        job.set_stage(STAGE_RUNNING)
        params = job.params
        # plot=False: QC's histogram plotting uses an interactive matplotlib
        # backend, which cannot create GUI windows off the main thread.
        qc_kws = {"min_lib_size": params.min_lib_size, "min_percent": params.min_percent, "plot": False}
        network_kws = {
            "backend": params.backend,
            "n_jobs": params.n_jobs,
            "random_state": params.random_state,
        }
        try:
            if params.workflow == "net":
                df = compare_networks(
                    x_df,
                    y_df,
                    x_label=params.x_label,
                    y_label=params.y_label,
                    qc_kws=qc_kws,
                    network_kws=network_kws,
                )
            else:
                df = virtual_knockout(
                    x_df,
                    ko_genes=params.ko_genes,
                    ko_method=params.ko_method,
                    strict_lambda=params.strict_lambda,
                    qc_kws=qc_kws,
                    network_kws=network_kws,
                )
            job.result = df
            job.set_stage(STAGE_DONE)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            logger.exception("job %s failed", job.id)
            job.error = str(exc)
            job.status = "error"
