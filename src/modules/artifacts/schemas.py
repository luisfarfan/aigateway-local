"""API schemas for the artifacts module."""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


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


class UploadInputFileRequest(BaseModel):
    """
    Request to pre-upload a source file before creating a job.
    Used for image_edit (source image), speech_to_text (audio), etc.
    Returns a storage_key that is then referenced in the job's input payload.
    """
    filename: str = Field(..., min_length=1, max_length=255)
    mime_type: str = Field(..., description="e.g. 'image/png', 'audio/wav'")
    size_bytes: int = Field(..., ge=1)


class UploadInputFileResponse(BaseModel):
    storage_key: str = Field(..., description="MinIO object key — use in job input payload")
    upload_url: str = Field(..., description="Presigned PUT URL to upload the file directly")
    expires_in_seconds: int
