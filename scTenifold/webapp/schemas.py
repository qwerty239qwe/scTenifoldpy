"""Pydantic request/response models for the scTenifold web API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    dataset_id: str
    name: str
    n_genes: int
    n_cells: int
    gene_names: list[str]


class JobCreate(BaseModel):
    workflow: Literal["net", "knk"]
    dataset_id: str
    dataset_id_y: Optional[str] = Field(None, description="required for the 'net' workflow")
    x_label: str = "X"
    y_label: str = "Y"
    ko_genes: Optional[list[str]] = Field(None, description="required for the 'knk' workflow")
    ko_method: Literal["default", "propagation"] = "default"
    strict_lambda: float = Field(0, ge=0)
    backend: Literal["serial", "joblib-loky", "joblib-threading"] = "serial"
    n_jobs: int = Field(1, ge=-1)
    random_state: int = 42
    min_lib_size: float = Field(10, ge=0)
    min_percent: float = Field(0.001, ge=0, le=1)


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    stage: str
    error: Optional[str] = None


class GeneResultRow(BaseModel):
    gene: str
    distance: float
    boxcox_distance: float
    z: float
    fc: float
    p_value: float
    adjusted_p_value: float


class JobResult(BaseModel):
    job_id: str
    workflow: str
    rows: list[GeneResultRow]
