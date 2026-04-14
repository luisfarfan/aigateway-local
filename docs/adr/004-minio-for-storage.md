# ADR-004: MinIO for Object Storage

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

AI jobs produce binary artifacts: images (PNG/JPEG), audio files (WAV/MP3), videos (MP4),
and logs. These artifacts need to be stored durably, served to clients, and potentially
moved to cloud storage in the future.

## Decision

Use **MinIO** as the local object storage backend, accessed via the S3-compatible API.

## Rationale

**Option A: Raw filesystem (`/data/jobs/{job_id}/outputs/`)**
- Simple to implement initially.
- No abstraction: switching to S3 later requires rewriting all storage code.
- No built-in presigned URLs, multipart upload, or lifecycle policies.
- Serving files requires a separate static file server or FastAPI `FileResponse`.

**Option B: MinIO (S3-compatible)**
- Runs locally in Docker — zero cloud dependency.
- Same API as AWS S3, Cloudflare R2, Backblaze B2. Migration = change 3 env vars.
- Built-in presigned URL generation for time-limited client access.
- Multipart upload for large video files.
- Lifecycle policies for automatic artifact expiry.
- Client is `aiobotocore` or `boto3` — same code for local and cloud.

The abstraction cost is minimal (one storage service class). The future-proofing is high.

## Storage Layout

```
bucket: local-ai-gateway
  jobs/{job_id}/outputs/{filename}     ← generated artifacts
  jobs/{job_id}/inputs/{filename}      ← uploaded source files (for image_edit, STT)
  jobs/{job_id}/logs/{filename}        ← execution logs
```

## Consequences

- `aiobotocore` added as dependency for async S3 operations.
- All artifact paths stored in DB are MinIO object keys, not filesystem paths.
- Public artifact access via presigned URLs (configurable TTL, default 24h).
- MinIO runs as a Docker service in docker-compose; production = swap endpoint to S3.
- Storage abstracted behind `StoragePort` interface — MinIO is one adapter.
