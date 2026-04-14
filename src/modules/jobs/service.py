"""
Job service — business logic for job lifecycle management.

Orchestrates: validation → idempotency check → DB persist → event publish → enqueue.
The service is the only place that coordinates across repository, publisher, and dispatcher.
"""
from datetime import datetime, timezone
from uuid import UUID

import structlog
from arq.connections import ArqRedis
from redis.asyncio import Redis

from src.core.domain import JobStatus
from src.core.exceptions import (
    JobAlreadyExistsError,
    JobCancellationError,
    JobNotFoundError,
    ProviderNotFoundError,
    ProviderNotSupportedError,
)
from src.modules.events.publisher import EventPublisher
from src.modules.events.schemas import SSEEvents
from src.modules.jobs.models import Job
from src.modules.jobs.repository import JobRepository
from src.modules.jobs.schemas import (
    ArtifactResponse,
    CancelJobResponse,
    CreateJobRequest,
    JobListFilters,
    JobListResponse,
    JobResponse,
)
from src.core.metrics import queue_depth
from src.modules.providers.registry import ProviderRegistry
from src.modules.queue.dispatcher import enqueue_job

log = structlog.get_logger(__name__)


class JobService:
    def __init__(
        self,
        repo: JobRepository,
        publisher: EventPublisher,
        registry: ProviderRegistry,
        arq: ArqRedis,
    ) -> None:
        self._repo = repo
        self._pub = publisher
        self._registry = registry
        self._arq = arq

    # ─── Create ───────────────────────────────────────────────────────────────

    async def create_job(
        self,
        request: CreateJobRequest,
        client_id: str,
    ) -> JobResponse:
        structlog.contextvars.bind_contextvars(
            job_type=request.type,
            provider=request.provider,
            client_id=client_id,
        )

        # 1. Validate provider can handle this job type + model
        try:
            self._registry.resolve(request.provider, request.type, request.model)
        except ValueError as e:
            if "not registered" in str(e):
                raise ProviderNotFoundError(request.provider) from e
            raise ProviderNotSupportedError(
                request.provider, request.type, request.model
            ) from e

        # 2. Idempotency check — return existing job if key already used
        if request.idempotency_key:
            existing = await self._repo.get_by_idempotency_key(request.idempotency_key)
            if existing is not None:
                log.info("job_idempotency_hit", job_id=str(existing.id))
                artifacts = await self._repo.get_artifacts(existing.id)
                return _to_response(existing, artifacts)

        # 3. Persist the job
        now = datetime.now(timezone.utc)
        job = Job(
            type=request.type,
            status=JobStatus.QUEUED,
            priority=request.priority,
            provider=request.provider,
            model=request.model,
            input_payload=request.input,
            client_id=client_id,
            idempotency_key=request.idempotency_key,
            tags=request.tags,
            max_retries=request.max_retries,
            timeout_seconds=request.timeout_seconds,
            queued_at=now,
        )
        job = await self._repo.create(job)

        # 4. Publish job_created event (Redis + DB)
        await self._pub.publish(
            SSEEvents.job_created(job.id, request.provider, request.model)
        )

        # 5. Enqueue to ARQ priority queue
        await enqueue_job(self._arq, job.id, request.priority)

        # Track queue depth (decremented in executor when job starts RUNNING)
        queue_depth.labels(priority=str(request.priority.value)).inc()

        log.info("job_created", job_id=str(job.id))
        return _to_response(job, [])

    # ─── Read ─────────────────────────────────────────────────────────────────

    async def get_job(self, job_id: UUID, client_id: str) -> JobResponse:
        job = await self._repo.get_by_id(job_id)
        if not job or job.client_id != client_id:
            raise JobNotFoundError(job_id)
        artifacts = await self._repo.get_artifacts(job_id)
        return _to_response(job, artifacts)

    async def list_jobs(
        self,
        filters: JobListFilters,
        client_id: str,
    ) -> JobListResponse:
        jobs, total = await self._repo.list(filters, client_id=client_id)
        items = []
        for job in jobs:
            artifacts = await self._repo.get_artifacts(job.id)
            items.append(_to_response(job, artifacts))
        return JobListResponse(
            items=items,
            total=total,
            page=filters.page,
            page_size=filters.page_size,
            has_next=(filters.page * filters.page_size) < total,
        )

    # ─── Cancel ───────────────────────────────────────────────────────────────

    async def cancel_job(self, job_id: UUID, client_id: str) -> CancelJobResponse:
        job = await self._repo.get_by_id(job_id)
        if not job or job.client_id != client_id:
            raise JobNotFoundError(job_id)

        if job.status.is_terminal():
            raise JobCancellationError(job_id, job.status)

        if not job.status.can_transition_to(JobStatus.CANCELLED):
            raise JobCancellationError(job_id, job.status)

        await self._repo.update_status(job, JobStatus.CANCELLED)
        await self._pub.publish(SSEEvents.cancelled(job_id, reason="Cancelled by client"))

        log.info("job_cancelled", job_id=str(job_id))
        return CancelJobResponse(
            job_id=job_id,
            cancelled=True,
            message="Job cancelled successfully.",
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_response(job: Job, artifacts: list) -> JobResponse:
    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        priority=job.priority,
        provider=job.provider,
        model=job.model,
        progress_percent=job.progress_percent,
        current_step=job.current_step,
        created_at=job.created_at,
        queued_at=job.queued_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        result_summary=job.result_summary,
        retry_count=job.retry_count,
        parent_job_id=job.parent_job_id,
        tags=job.tags or {},
        artifacts=[
            ArtifactResponse(
                id=a.id,
                job_id=a.job_id,
                artifact_type=a.artifact_type,
                filename=a.filename,
                public_url=a.public_url,
                mime_type=a.mime_type,
                size_bytes=a.size_bytes,
                created_at=a.created_at,
            )
            for a in artifacts
        ],
    )
