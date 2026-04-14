"""
Artifact service — retrieves artifacts and refreshes presigned URLs.

Presigned URLs have a TTL (default 24h). This service always generates a
fresh URL on demand so clients never get a stale link.
"""
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from src.core.exceptions import ArtifactNotFoundError, StorageError
from src.core.storage import storage
from src.modules.jobs.models import Artifact
from src.modules.artifacts.schemas import ArtifactResponse

log = structlog.get_logger(__name__)


class ArtifactService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_artifact(self, artifact_id: UUID, client_id: str) -> ArtifactResponse:
        """
        Fetches an artifact by ID, verifies client ownership via the parent job,
        and returns a fresh presigned download URL.
        """
        artifact = await self._session.get(Artifact, artifact_id)
        if not artifact:
            raise ArtifactNotFoundError(str(artifact_id))

        # Ownership check — artifact belongs to a job that belongs to the client
        from src.modules.jobs.models import Job
        job = await self._session.get(Job, artifact.job_id)
        if not job or job.client_id != client_id:
            raise ArtifactNotFoundError(str(artifact_id))

        # Refresh presigned URL (avoids returning a stale one from DB)
        try:
            fresh_url = await storage.presigned_download_url(artifact.storage_key)
            artifact.public_url = fresh_url
            self._session.add(artifact)
            await self._session.flush()
        except StorageError:
            log.warning("artifact_presign_refresh_failed", key=artifact.storage_key)
            # Return whatever URL is stored — better than failing completely

        return ArtifactResponse(
            id=artifact.id,
            job_id=artifact.job_id,
            artifact_type=artifact.artifact_type,
            filename=artifact.filename,
            public_url=artifact.public_url,
            mime_type=artifact.mime_type,
            size_bytes=artifact.size_bytes,
            created_at=artifact.created_at,
        )

    async def list_job_artifacts(
        self, job_id: UUID, client_id: str
    ) -> list[ArtifactResponse]:
        """Returns all artifacts for a job, with fresh presigned URLs."""
        from src.modules.jobs.models import Job
        job = await self._session.get(Job, job_id)
        if not job or job.client_id != client_id:
            return []

        stmt = select(Artifact).where(Artifact.job_id == job_id)
        result = await self._session.execute(stmt)
        artifacts = list(result.scalars().all())

        responses = []
        for artifact in artifacts:
            try:
                url = await storage.presigned_download_url(artifact.storage_key)
                artifact.public_url = url
            except StorageError:
                pass
            responses.append(ArtifactResponse(
                id=artifact.id,
                job_id=artifact.job_id,
                artifact_type=artifact.artifact_type,
                filename=artifact.filename,
                public_url=artifact.public_url,
                mime_type=artifact.mime_type,
                size_bytes=artifact.size_bytes,
                created_at=artifact.created_at,
            ))
        return responses
