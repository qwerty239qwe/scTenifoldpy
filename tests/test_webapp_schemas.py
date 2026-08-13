"""Validation tests for scTenifold.webapp.schemas request models.

Skipped automatically when the optional ``[ui]`` extra is absent, so the
default CI matrix stays green without fastapi/uvicorn.
"""
import pytest

pytest.importorskip("fastapi")

from pydantic import ValidationError

from scTenifold.webapp.schemas import JobCreate


def _job(**overrides):
    params = {"workflow": "grn", "dataset_id": "abc"}
    params.update(overrides)
    return JobCreate(**params)


def test_n_jobs_defaults_to_one():
    assert _job().n_jobs == 1


@pytest.mark.parametrize("n_jobs", [-1, 1, 2, 16])
def test_n_jobs_accepts_all_cores_and_positive_counts(n_jobs):
    assert _job(n_jobs=n_jobs).n_jobs == n_jobs


def test_n_jobs_rejects_zero():
    # joblib raises "n_jobs == 0 in Parallel has no meaning"; the API should
    # reject it before a job is ever scheduled.
    with pytest.raises(ValidationError, match="must be -1"):
        _job(n_jobs=0)


@pytest.mark.parametrize("n_jobs", [-2, -10])
def test_n_jobs_rejects_below_negative_one(n_jobs):
    with pytest.raises(ValidationError):
        _job(n_jobs=n_jobs)
