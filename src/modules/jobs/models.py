"""
SQLModel table definitions for the jobs module.

SQLModel unifies the SQLAlchemy ORM model and the Pydantic schema in one class.
Tables defined here are the source of truth for DB schema — Alembic reads from them.
"""
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from src.core.domain import JobPriority, JobStatus, JobType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    type: JobType = Field(index=True)
    status: JobStatus = Field(default=JobStatus.QUEUED, index=True)
    priority: JobPriority = Field(default=JobPriority.NORMAL, index=True)

    # Routing — which engine handles this job
    provider: str = Field(index=True)       # e.g. "diffusers", "local_tts", "stub"
    model: str | None = Field(default=None) # e.g. "stable-diffusion-xl"

    # Payload — flexible JSON, validated at API layer by Pydantic schemas
    input_payload: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    result_summary: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = None
    error_detail: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

    # Client metadata
    client_id: str | None = Field(default=None, index=True)
    idempotency_key: str | None = Field(default=None, index=True, unique=True)
    tags: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))

    # Lifecycle timestamps
    created_at: datetime = Field(default_factory=_utcnow, index=True)
    queued_at: datetime | None = None
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Execution control
    worker_id: str | None = None
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    timeout_seconds: int | None = None

    # Progress tracking (updated by workers in real-time)
    progress_percent: float | None = None
    current_step: str | None = None

    # Pipeline support — subjobs link to their parent
    parent_job_id: UUID | None = Field(default=None, foreign_key="jobs.id", index=True)
    pipeline_step_index: int | None = None


class JobEvent(SQLModel, table=True):
    """
    Persisted log of every SSE event emitted for a job.
    Used to replay missed events when a client reconnects (Last-Event-ID).
    """
    __tablename__ = "job_events"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    job_id: UUID = Field(foreign_key="jobs.id", index=True)
    event_type: str = Field(index=True)
    occurred_at: datetime = Field(default_factory=_utcnow, index=True)
    payload: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))


class Artifact(SQLModel, table=True):
    """An output file or data blob produced by a job, stored in MinIO."""
    __tablename__ = "artifacts"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    job_id: UUID = Field(foreign_key="jobs.id", index=True)
    artifact_type: str                   # ArtifactType enum value
    filename: str
    storage_key: str                     # MinIO object key: jobs/{job_id}/outputs/{filename}
    public_url: str | None = None        # presigned URL (refreshed on demand)
    mime_type: str | None = None
    size_bytes: int | None = None
    extra_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class RegisteredModel(SQLModel, table=True):
    """Catalog of AI models available on this machine, per provider."""
    __tablename__ = "registered_models"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    provider: str = Field(index=True)          # matches BaseProvider.provider_id
    model_id: str = Field(index=True)          # e.g. "stable-diffusion-xl"
    modality: str                              # Modality enum value
    display_name: str | None = None
    description: str | None = None
    is_available: bool = Field(default=True, index=True)
    # What the model supports: {"supports_negative_prompt": true, "max_steps": 150, ...}
    capabilities: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    # Resource needs: {"vram_mb": 8192, "ram_mb": 4096}
    resource_requirements: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    registered_at: datetime = Field(default_factory=_utcnow)


class WorkerRuntime(SQLModel, table=True):
    """Live heartbeat record for each active worker process."""
    __tablename__ = "worker_runtimes"

    id: str = Field(primary_key=True)          # stable worker_id string
    modality: str                              # which modality this worker handles
    status: str = Field(default="idle")        # "idle" | "busy" | "offline"
    current_job_id: UUID | None = None
    jobs_processed: int = Field(default=0)
    jobs_failed: int = Field(default=0)
    last_heartbeat: datetime | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    extra_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
