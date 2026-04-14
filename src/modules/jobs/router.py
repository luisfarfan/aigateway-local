"""
Jobs HTTP router — CRUD endpoints for job management.

All routes require authentication (get_current_client_id).
The router composes dependencies and delegates to JobService.
"""
from uuid import UUID

import structlog
from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_session
from src.core.exceptions import (
    JobAlreadyExistsError,
    JobCancellationError,
    JobNotFoundError,
    ProviderNotFoundError,
    ProviderNotSupportedError,
)
from src.core.redis import get_redis
from src.modules.auth.middleware import get_current_client_id
from src.modules.events.publisher import EventPublisher
from src.modules.jobs.repository import JobRepository
from src.modules.jobs.schemas import (
    CancelJobResponse,
    CreateJobRequest,
    JobListFilters,
    JobListResponse,
    JobPriority,
    JobResponse,
    JobStatus,
    JobType,
)
from src.modules.jobs.service import JobService
from src.modules.providers.registry import ProviderRegistry

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["Jobs"])


# ─── Dependencies ─────────────────────────────────────────────────────────────

def _get_registry(request: Request) -> ProviderRegistry:
    return request.app.state.provider_registry


def _get_arq(request: Request) -> ArqRedis:
    return request.app.state.arq_pool


def _build_service(
    session: AsyncSession = Depends(get_session),
    registry: ProviderRegistry = Depends(_get_registry),
    arq: ArqRedis = Depends(_get_arq),
) -> JobService:
    redis = get_redis()
    repo = JobRepository(session)
    publisher = EventPublisher(redis, session)
    return JobService(repo=repo, publisher=publisher, registry=registry, arq=arq)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and enqueue a new job",
)
async def create_job(
    body: CreateJobRequest,
    client_id: str = Depends(get_current_client_id),
    service: JobService = Depends(_build_service),
) -> JobResponse:
    """
    Creates a job and immediately enqueues it for processing.

    - Returns **202 Accepted** with the job object (status=`queued`).
    - If `idempotency_key` is provided and a job with that key already exists,
      returns the existing job without creating a duplicate.
    - Subscribe to `GET /jobs/{id}/events` for real-time progress via SSE.
    """
    try:
        return await service.create_job(body, client_id)
    except JobAlreadyExistsError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(e))
    except ProviderNotFoundError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except ProviderNotSupportedError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.get(
    "",
    response_model=JobListResponse,
    summary="List jobs for the authenticated client",
)
async def list_jobs(
    status_filter: JobStatus | None = Query(default=None, alias="status"),
    type_filter: JobType | None = Query(default=None, alias="type"),
    priority_filter: JobPriority | None = Query(default=None, alias="priority"),
    provider: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    client_id: str = Depends(get_current_client_id),
    service: JobService = Depends(_build_service),
) -> JobListResponse:
    filters = JobListFilters(
        status=status_filter,
        type=type_filter,
        priority=priority_filter,
        provider=provider,
        page=page,
        page_size=page_size,
    )
    return await service.list_jobs(filters, client_id)


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get a single job by ID",
)
async def get_job(
    job_id: UUID,
    client_id: str = Depends(get_current_client_id),
    service: JobService = Depends(_build_service),
) -> JobResponse:
    try:
        return await service.get_job(job_id, client_id)
    except JobNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found.")


@router.delete(
    "/{job_id}",
    response_model=CancelJobResponse,
    summary="Cancel a job",
)
async def cancel_job(
    job_id: UUID,
    client_id: str = Depends(get_current_client_id),
    service: JobService = Depends(_build_service),
) -> CancelJobResponse:
    """
    Cancels a job if it hasn't started yet (queued/scheduled).
    If the job is already running, a cancellation signal is sent — the worker
    may or may not honour it depending on the provider.
    Terminal jobs (completed/failed/cancelled) cannot be cancelled.
    """
    try:
        return await service.cancel_job(job_id, client_id)
    except JobNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found.")
    except JobCancellationError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(e))
