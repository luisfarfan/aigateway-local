"""
Object storage service — MinIO / S3-compatible adapter.

Abstracts all storage operations behind a single class.
Swapping MinIO for real S3: change 3 env vars (endpoint, access_key, secret_key).
The rest of the codebase doesn't change.

Key layout in the bucket:
  jobs/{job_id}/outputs/{filename}   ← generated artifacts (images, audio, video)
  jobs/{job_id}/inputs/{filename}    ← source files uploaded by the client
  jobs/{job_id}/logs/{filename}      ← execution logs
"""
import structlog
from aiobotocore.session import get_session
from botocore.exceptions import ClientError

from src.core.config import get_settings
from src.core.exceptions import ArtifactNotFoundError, StorageError

log = structlog.get_logger(__name__)

_settings = get_settings()


class StorageService:
    """
    Async MinIO/S3 client.

    All methods are async. Use a single instance (singleton) — it creates
    a new aiobotocore client per call (stateless, thread-safe).
    """

    def __init__(self) -> None:
        self._session = get_session()
        self._bucket = _settings.minio_bucket

    def _client_ctx(self):
        """Returns an async context manager that yields a boto3-compatible S3 client."""
        return self._session.create_client(
            "s3",
            endpoint_url=_settings.minio_endpoint,
            aws_access_key_id=_settings.minio_access_key,
            aws_secret_access_key=_settings.minio_secret_key,
            region_name=_settings.minio_region,
        )

    async def ensure_bucket(self) -> None:
        """
        Creates the storage bucket if it doesn't exist.
        Called once at app startup (lifespan).
        """
        async with self._client_ctx() as client:
            try:
                await client.head_bucket(Bucket=self._bucket)
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                    await client.create_bucket(Bucket=self._bucket)
                    log.info("storage_bucket_created", bucket=self._bucket)
                else:
                    raise StorageError(f"Failed to check bucket: {e}") from e

    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload raw bytes to the given key."""
        try:
            async with self._client_ctx() as client:
                await client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=data,
                    ContentType=content_type,
                )
        except ClientError as e:
            raise StorageError(f"Upload failed for key '{key}': {e}") from e

    async def upload_file(self, key: str, file_path: str, content_type: str) -> None:
        """Upload a file from the local filesystem."""
        with open(file_path, "rb") as f:
            data = f.read()
        await self.upload(key, data, content_type)

    async def download(self, key: str) -> bytes:
        """Download an object and return its raw bytes."""
        try:
            async with self._client_ctx() as client:
                response = await client.get_object(Bucket=self._bucket, Key=key)
                return await response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                raise ArtifactNotFoundError(key) from e
            raise StorageError(f"Download failed for key '{key}': {e}") from e

    async def delete(self, key: str) -> None:
        """Delete an object. Silently succeeds if the key doesn't exist."""
        try:
            async with self._client_ctx() as client:
                await client.delete_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            raise StorageError(f"Delete failed for key '{key}': {e}") from e

    async def presigned_download_url(
        self,
        key: str,
        expires_in: int | None = None,
    ) -> str:
        """Generate a presigned GET URL valid for `expires_in` seconds."""
        ttl = expires_in or _settings.minio_presigned_expiry
        try:
            async with self._client_ctx() as client:
                return await client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self._bucket, "Key": key},
                    ExpiresIn=ttl,
                )
        except ClientError as e:
            raise StorageError(f"Presigned URL generation failed for '{key}': {e}") from e

    async def presigned_upload_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned PUT URL so clients can upload directly to MinIO."""
        try:
            async with self._client_ctx() as client:
                return await client.generate_presigned_url(
                    "put_object",
                    Params={"Bucket": self._bucket, "Key": key},
                    ExpiresIn=expires_in,
                )
        except ClientError as e:
            raise StorageError(f"Presigned upload URL failed for '{key}': {e}") from e

    async def get_size(self, key: str) -> int:
        """Returns the size in bytes of an object."""
        try:
            async with self._client_ctx() as client:
                response = await client.head_object(Bucket=self._bucket, Key=key)
                return response["ContentLength"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise ArtifactNotFoundError(key) from e
            raise StorageError(f"head_object failed for '{key}': {e}") from e

    # ─── Key helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def output_key(job_id: str, filename: str) -> str:
        return f"jobs/{job_id}/outputs/{filename}"

    @staticmethod
    def input_key(job_id: str, filename: str) -> str:
        return f"jobs/{job_id}/inputs/{filename}"

    @staticmethod
    def log_key(job_id: str, filename: str) -> str:
        return f"jobs/{job_id}/logs/{filename}"


# Module-level singleton — import and use directly.
storage = StorageService()
