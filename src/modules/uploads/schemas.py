"""Schemas for the uploads module."""
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Returned after a direct multipart file upload."""
    storage_key: str = Field(..., description="Use this key in your job's input payload")
    filename: str
    mime_type: str
    size_bytes: int


class PresignedUploadRequest(BaseModel):
    """Request body for obtaining a presigned PUT URL."""
    filename: str = Field(..., min_length=1, max_length=255)
    mime_type: str = Field(..., description="e.g. 'image/png', 'audio/wav', 'video/mp4'")
    size_bytes: int = Field(..., ge=1, le=5_368_709_120)  # max 5 GB


class PresignedUploadResponse(BaseModel):
    """Client uses upload_url to PUT the file directly to MinIO, then uses storage_key in jobs."""
    storage_key: str = Field(..., description="Use this key in your job's input payload")
    upload_url: str = Field(..., description="Presigned PUT URL — upload directly to MinIO")
    expires_in_seconds: int = 3600
