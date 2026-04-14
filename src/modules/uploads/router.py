"""
Uploads router — allows clients to upload source files before creating jobs.

Required for:
  - image_edit   → upload the source image, get storage_key → use as `image_key` in job input
  - image_edit   → upload the mask image,   get storage_key → use as `mask_key`
  - speech_to_text → upload the audio file, get storage_key → use as `audio_key`

Two upload methods:

1. Direct upload (POST /uploads):
   Client sends the file as multipart/form-data.
   Gateway reads and uploads to MinIO.
   Best for small-medium files (< 100 MB).

2. Presigned upload (POST /uploads/presigned):
   Gateway returns a presigned PUT URL.
   Client uploads directly to MinIO (no traffic through the gateway).
   Best for large files (images, long audio, video).
"""
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.core.exceptions import StorageError
from src.core.storage import storage
from src.modules.auth.middleware import get_current_client_id
from src.modules.uploads.schemas import (
    PresignedUploadRequest,
    PresignedUploadResponse,
    UploadResponse,
)

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/uploads", tags=["Uploads"])

# 500 MB hard limit for direct uploads
_MAX_DIRECT_UPLOAD_BYTES = 500 * 1024 * 1024

# Allowed MIME types for source files
_ALLOWED_MIME_TYPES = {
    # Images
    "image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/bmp",
    # Audio
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/flac", "audio/aac", "audio/m4a",
    # Video (for future STT on video files)
    "video/mp4", "video/mpeg", "video/webm",
    # Generic — allow if client knows what they're doing
    "application/octet-stream",
}


def _upload_key(client_id: str, filename: str) -> str:
    """Generates a unique, namespaced MinIO key for an uploaded source file."""
    file_id = uuid4().hex[:12]
    # Sanitize filename — remove path traversal chars
    safe_name = filename.replace("/", "_").replace("..", "_")
    return f"uploads/{client_id}/{file_id}/{safe_name}"


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a source file (multipart)",
)
async def upload_file(
    file: UploadFile = File(..., description="Source file to upload (image, audio, video)"),
    client_id: str = Depends(get_current_client_id),
) -> UploadResponse:
    """
    Upload a source file that will be referenced in a job's input payload.

    Returns a `storage_key` — pass it as `image_key`, `mask_key`, or `audio_key`
    in the corresponding job's `input` field.

    Example flow:
    ```
    # 1. Upload image
    POST /api/v1/uploads  (multipart, file=photo.png)
    → { "storage_key": "uploads/abc/def/photo.png", ... }

    # 2. Create job referencing the uploaded file
    POST /api/v1/jobs
    { "type": "image_edit", "provider": "diffusers",
      "input": { "prompt": "...", "image_key": "uploads/abc/def/photo.png" } }
    ```
    """
    mime_type = file.content_type or "application/octet-stream"
    if mime_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{mime_type}'. "
                   f"Allowed: images, audio, video.",
        )

    data = await file.read()

    if len(data) > _MAX_DIRECT_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large for direct upload ({len(data) / 1e6:.1f} MB). "
                   f"Use POST /uploads/presigned for files > 500 MB.",
        )

    if len(data) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="File is empty.")

    key = _upload_key(client_id, file.filename or "file")

    try:
        await storage.upload(key, data, mime_type)
    except StorageError as e:
        log.exception("upload_storage_error", key=key)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"Storage error: {e}")

    log.info(
        "file_uploaded",
        key=key,
        size_bytes=len(data),
        mime_type=mime_type,
        client_id=client_id,
    )

    return UploadResponse(
        storage_key=key,
        filename=file.filename or "file",
        mime_type=mime_type,
        size_bytes=len(data),
    )


@router.post(
    "/presigned",
    response_model=PresignedUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a presigned URL for direct upload to storage",
)
async def presigned_upload(
    body: PresignedUploadRequest,
    client_id: str = Depends(get_current_client_id),
) -> PresignedUploadResponse:
    """
    Returns a presigned PUT URL so the client can upload directly to MinIO.

    Use this for large files (> 100 MB) to avoid routing traffic through the gateway.

    Flow:
    ```
    # 1. Get presigned URL
    POST /api/v1/uploads/presigned
    { "filename": "recording.wav", "mime_type": "audio/wav", "size_bytes": 52428800 }
    → { "storage_key": "uploads/.../recording.wav", "upload_url": "http://minio/...", ... }

    # 2. Upload directly to MinIO (PUT request, no auth needed — URL is time-limited)
    PUT {upload_url}  (body = raw file bytes)

    # 3. Create job
    POST /api/v1/jobs
    { "type": "speech_to_text", "input": { "audio_key": "uploads/.../recording.wav" } }
    ```
    """
    if body.mime_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{body.mime_type}'.",
        )

    key = _upload_key(client_id, body.filename)

    try:
        url = await storage.presigned_upload_url(key, expires_in=3600)
    except StorageError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"Storage error: {e}")

    log.info(
        "presigned_upload_issued",
        key=key,
        mime_type=body.mime_type,
        size_bytes=body.size_bytes,
        client_id=client_id,
    )

    return PresignedUploadResponse(
        storage_key=key,
        upload_url=url,
        expires_in_seconds=3600,
    )
