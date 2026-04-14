"""
Artifacts router — download and inspect job outputs.

Endpoints:
  GET /artifacts/{id}           — artifact metadata + fresh presigned URL
  GET /artifacts/{id}/download  — 302 redirect to presigned URL (browser-friendly)
  GET /jobs/{id}/artifacts      — all artifacts for a job
"""
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_session
from src.core.exceptions import ArtifactNotFoundError
from src.modules.artifacts.schemas import ArtifactResponse
from src.modules.artifacts.service import ArtifactService
from src.modules.auth.middleware import get_current_client_id

log = structlog.get_logger(__name__)
router = APIRouter(tags=["Artifacts"])


def _build_service(session: AsyncSession = Depends(get_session)) -> ArtifactService:
    return ArtifactService(session)


@router.get(
    "/artifacts/{artifact_id}",
    response_model=ArtifactResponse,
    summary="Get artifact metadata and a fresh download URL",
)
async def get_artifact(
    artifact_id: UUID,
    client_id: str = Depends(get_current_client_id),
    service: ArtifactService = Depends(_build_service),
) -> ArtifactResponse:
    """
    Returns artifact metadata including a **fresh** presigned download URL.
    The URL is valid for 24h (configurable via MINIO_PRESIGNED_EXPIRY).
    """
    try:
        return await service.get_artifact(artifact_id, client_id)
    except ArtifactNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found.",
        )


@router.get(
    "/artifacts/{artifact_id}/download",
    summary="Redirect to artifact download URL",
    response_class=RedirectResponse,
)
async def download_artifact(
    artifact_id: UUID,
    client_id: str = Depends(get_current_client_id),
    service: ArtifactService = Depends(_build_service),
) -> RedirectResponse:
    """
    Issues a **302 redirect** to the presigned MinIO download URL.
    Useful for browser-based downloads or `curl -L`.

    ```bash
    curl -L -H "Authorization: Bearer <key>" \\
      http://gateway/api/v1/artifacts/{id}/download \\
      -o result.png
    ```
    """
    try:
        artifact = await service.get_artifact(artifact_id, client_id)
    except ArtifactNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Artifact {artifact_id} not found.")

    if not artifact.public_url:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact URL not available — storage may be unreachable.",
        )

    return RedirectResponse(url=artifact.public_url, status_code=status.HTTP_302_FOUND)


@router.get(
    "/jobs/{job_id}/artifacts",
    response_model=list[ArtifactResponse],
    summary="List all artifacts for a job",
)
async def list_job_artifacts(
    job_id: UUID,
    client_id: str = Depends(get_current_client_id),
    service: ArtifactService = Depends(_build_service),
) -> list[ArtifactResponse]:
    """
    Returns all output artifacts for a job with fresh presigned URLs.
    Equivalent to the `artifacts` field on `GET /jobs/{id}` but with refreshed URLs.
    """
    return await service.list_job_artifacts(job_id, client_id)
