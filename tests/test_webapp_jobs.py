"""Unit tests for scTenifold.webapp.jobs.JobManager.

Skipped automatically when the optional ``[ui]`` extra is absent, so the
default CI matrix stays green without fastapi/uvicorn.
"""
import time

import pytest

pytest.importorskip("fastapi")

from scTenifold.data import get_test_df
from scTenifold.webapp.jobs import DatasetNotFoundError, JobManager, JobNotFoundError
from scTenifold.webapp.schemas import JobCreate


def _wait_for(manager: JobManager, job_id: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get_job(job_id)
        if job.status in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_dataset_round_trip():
    manager = JobManager()
    df = get_test_df(n_cells=20, n_genes=30, random_state=1)
    dataset_id = manager.add_dataset(df)
    assert manager.get_dataset(dataset_id) is df


def test_get_dataset_unknown_id_raises():
    manager = JobManager()
    with pytest.raises(DatasetNotFoundError):
        manager.get_dataset("does-not-exist")


def test_get_job_unknown_id_raises():
    manager = JobManager()
    with pytest.raises(JobNotFoundError):
        manager.get_job("does-not-exist")


def test_net_job_runs_to_completion():
    manager = JobManager()
    x_id = manager.add_dataset(get_test_df(n_cells=150, n_genes=150, random_state=1))
    y_id = manager.add_dataset(get_test_df(n_cells=150, n_genes=150, random_state=2))
    params = JobCreate(workflow="net", dataset_id=x_id, dataset_id_y=y_id, min_lib_size=10)

    job_id = manager.submit(params)
    job = _wait_for(manager, job_id)

    assert job.status == "done"
    assert job.error is None
    assert list(job.result.columns) == [
        "Gene", "Distance", "boxcox-transformed distance", "Z", "FC", "p-value", "adjusted p-value",
    ]


def test_knk_job_runs_to_completion():
    manager = JobManager()
    df = get_test_df(n_cells=150, n_genes=150, random_state=1)
    x_id = manager.add_dataset(df)
    params = JobCreate(workflow="knk", dataset_id=x_id, ko_genes=["NG-1"], min_lib_size=10)

    job_id = manager.submit(params)
    job = _wait_for(manager, job_id)

    assert job.status == "done"
    assert job.error is None
    assert set(job.result["Gene"]) <= set(df.index.astype(str))


def test_submit_unknown_dataset_raises_before_running():
    manager = JobManager()
    params = JobCreate(workflow="knk", dataset_id="missing", ko_genes=["NG-1"])
    with pytest.raises(DatasetNotFoundError):
        manager.submit(params)


def test_grn_job_runs_to_completion():
    manager = JobManager()
    df = get_test_df(n_cells=150, n_genes=150, random_state=1)
    x_id = manager.add_dataset(df)
    params = JobCreate(workflow="grn", dataset_id=x_id, min_lib_size=10)

    job_id = manager.submit(params)
    job = _wait_for(manager, job_id)

    assert job.status == "done"
    assert job.error is None
    assert list(job.result.columns) == ["Source", "Target", "Weight"]
    assert len(job.result) > 0
    gene_names = set(df.index.astype(str))
    assert set(job.result["Source"]) <= gene_names
    assert set(job.result["Target"]) <= gene_names
    # sorted by |weight| descending
    weights = job.result["Weight"].abs().to_numpy()
    assert (weights[:-1] >= weights[1:]).all()
