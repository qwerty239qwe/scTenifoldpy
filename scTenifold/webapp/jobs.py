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

from scTenifold import cal_pcNet, compare_networks, sc_QC, virtual_knockout

from .schemas import JobCreate

logger = logging.getLogger(__name__)

# Stages surfaced to the UI as a simple progress indicator. Each workflow
# runs QC through its final step as one call, so there is no finer-grained
# progress to report.
STAGE_QUEUED = "queued"
STAGE_RUNNING = {
    "net": "building networks, decomposing tensors, aligning manifolds",
    "knk": "building networks, decomposing tensors, aligning manifolds",
    "grn": "running QC and building the gene regulatory network",
}
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
        params = job.params
        job.set_stage(STAGE_RUNNING[params.workflow])
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
            elif params.workflow == "knk":
                df = virtual_knockout(
                    x_df,
                    ko_genes=params.ko_genes,
                    ko_method=params.ko_method,
                    strict_lambda=params.strict_lambda,
                    qc_kws=qc_kws,
                    network_kws=network_kws,
                )
            else:
                df = self._build_grn(x_df, params, qc_kws)
            job.result = df
            job.set_stage(STAGE_DONE)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            logger.exception("job %s failed", job.id)
            job.error = str(exc)
            job.status = "error"

    @staticmethod
    def _build_grn(x_df: pd.DataFrame, params: JobCreate, qc_kws: dict) -> pd.DataFrame:
        """Run QC then build a single consensus gene regulatory network.

        Unlike 'net'/'knk', there's only one network to build (no
        resampling across many networks), so 'backend'/'n_jobs' don't
        apply here.
        """
        # sc_QC (unlike the scTenifoldNet/Knk classes' _QC wrapper) has no
        # 'plot' kwarg of its own.
        qc_df = sc_QC(x_df, min_lib_size=qc_kws["min_lib_size"], min_percent=qc_kws["min_percent"])
        network = cal_pcNet(qc_df, random_state=params.random_state).tocoo()
        gene_names = qc_df.index.to_numpy()
        edges = pd.DataFrame({
            "Source": gene_names[network.row],
            "Target": gene_names[network.col],
            "Weight": network.data,
        })
        return edges.sort_values("Weight", key=abs, ascending=False).reset_index(drop=True)
