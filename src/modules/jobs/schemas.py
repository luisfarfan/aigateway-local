"""
API contracts for the jobs module.

These Pydantic models define what the HTTP API accepts and returns.
They are separate from SQLModel table models — they represent the external interface,
not the DB schema.
"""
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from src.core.domain import JobPriority, JobStatus, JobType


# ─── Input payload schemas per job type ───────────────────────────────────────
# These validate the `input` field of CreateJobRequest depending on job type.
# The API layer validates the full payload; workers receive it as raw dict.

class TextGenerationInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    system_prompt: str | None = None
    max_tokens: int = Field(default=2048, ge=1, le=32_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop_sequences: list[str] = Field(default_factory=list)


class TextEmbeddingInput(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=512)
    normalize: bool = Field(default=True)


class TextToSpeechInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5_000)
    voice: str = Field(..., description="Voice ID supported by the provider")
    language: str = Field(default="en")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="wav")  # wav | mp3 | ogg


class SpeechToTextInput(BaseModel):
    audio_key: str = Field(..., description="MinIO object key of the source audio file")
    language: str | None = Field(default=None, description="None = auto-detect")
    task: str = Field(default="transcribe")  # transcribe | translate


class ImageGenerationInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2_000)
    negative_prompt: str | None = None
    width: int = Field(default=1024, ge=64, le=2048, multiple_of=8)
    height: int = Field(default=1024, ge=64, le=2048, multiple_of=8)
    steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
    seed: int | None = None
    num_images: int = Field(default=1, ge=1, le=4)
    scheduler: str | None = None  # DDIM | Euler | DPM++ etc.


class ImageEditInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2_000)
    image_key: str = Field(..., description="MinIO object key of the source image")
    mask_key: str | None = Field(default=None, description="MinIO key of the mask image")
    negative_prompt: str | None = None
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
    seed: int | None = None


class VideoGenerationInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2_000)
    negative_prompt: str | None = None
    width: int = Field(default=576, ge=64, le=1280, multiple_of=8)
    height: int = Field(default=320, ge=64, le=720, multiple_of=8)
    num_frames: int = Field(default=16, ge=1, le=128)
    fps: int = Field(default=8, ge=1, le=60)
    seed: int | None = None


class PipelineStepSpec(BaseModel):
    step_type: str = Field(
        ..., description="e.g. 'script_generation', 'tts', 'image_generation', 'video_assembly'"
    )
    provider: str | None = None
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class MultimodalPipelineInput(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2_000)
    steps: list[PipelineStepSpec] = Field(..., min_length=1, max_length=20)
    context: dict[str, Any] = Field(default_factory=dict)


# ─── Request / Response ────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    """
    The request body for POST /jobs.
    The `input` field is a raw dict here — validated per-type by the job service
    using the specific Input schemas above.
    """
    type: JobType
    priority: JobPriority = JobPriority.NORMAL
    provider: str = Field(..., min_length=1, description="Provider ID, e.g. 'diffusers'")
    model: str | None = Field(default=None, description="Model ID within the provider")
    input: dict[str, Any] = Field(..., description="Job-type-specific input payload")
    idempotency_key: str | None = Field(
        default=None,
        max_length=255,
        description="If provided, duplicate requests return the existing job",
    )
    tags: dict[str, str] = Field(default_factory=dict, max_length=10)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int | None = Field(
        default=None,
        ge=10,
        le=7200,
        description="Hard timeout for execution. None = provider default.",
    )


class ArtifactResponse(BaseModel):
    id: UUID
    job_id: UUID
    artifact_type: str
    filename: str
    public_url: str | None
    mime_type: str | None
    size_bytes: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class JobResponse(BaseModel):
    id: UUID
    type: JobType
    status: JobStatus
    priority: JobPriority
    provider: str
    model: str | None
    progress_percent: float | None
    current_step: str | None
    created_at: datetime
    queued_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    result_summary: dict[str, Any] | None
    artifacts: list[ArtifactResponse] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    retry_count: int
    parent_job_id: UUID | None

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class CancelJobResponse(BaseModel):
    job_id: UUID
    cancelled: bool
    message: str


class JobListFilters(BaseModel):
    """Query parameters for GET /jobs."""
    status: JobStatus | None = None
    type: JobType | None = None
    priority: JobPriority | None = None
    provider: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
