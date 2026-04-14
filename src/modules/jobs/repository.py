"""
Job repository — all DB access for the jobs module.

No business logic here. The service layer calls the repository.
The repository only knows about SQLModel/SQLAlchemy operations.
"""
from datetime import datetime
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, func, select

from src.core.domain import JobStatus
from src.modules.jobs.models import Artifact, Job, JobEvent
from src.modules.jobs.schemas import JobListFilters

log = structlog.get_logger(__name__)


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    # ─── Jobs ─────────────────────────────────────────────────────────────────

    async def create(self, job: Job) -> Job:
        self._s.add(job)
        await self._s.flush()
        await self._s.refresh(job)
        log.info("job_persisted", job_id=str(job.id), type=job.type, provider=job.provider)
        return job

    async def get_by_id(self, job_id: UUID) -> Job | None:
        return await self._s.get(Job, job_id)

    async def get_by_idempotency_key(self, key: str) -> Job | None:
        stmt = select(Job).where(Job.idempotency_key == key)
        result = await self._s.execute(stmt)
        return result.scalar_one_or_none()

    async def list(self, filters: JobListFilters, client_id: str | None = None) -> tuple[list[Job], int]:
        stmt = select(Job)
        count_stmt = select(func.count()).select_from(Job)

        if client_id:
            stmt = stmt.where(Job.client_id == client_id)
            count_stmt = count_stmt.where(Job.client_id == client_id)
        if filters.status:
            stmt = stmt.where(Job.status == filters.status)
            count_stmt = count_stmt.where(Job.status == filters.status)
        if filters.type:
            stmt = stmt.where(Job.type == filters.type)
            count_stmt = count_stmt.where(Job.type == filters.type)
        if filters.priority:
            stmt = stmt.where(Job.priority == filters.priority)
            count_stmt = count_stmt.where(Job.priority == filters.priority)
        if filters.provider:
            stmt = stmt.where(Job.provider == filters.provider)
            count_stmt = count_stmt.where(Job.provider == filters.provider)

        total = (await self._s.execute(count_stmt)).scalar_one()

        offset = (filters.page - 1) * filters.page_size
        stmt = (
            stmt.order_by(col(Job.created_at).desc())
            .offset(offset)
            .limit(filters.page_size)
        )
        jobs = list((await self._s.execute(stmt)).scalars().all())
        return jobs, total

    async def update(self, job: Job) -> Job:
        self._s.add(job)
        await self._s.flush()
        await self._s.refresh(job)
        return job

    async def update_progress(
        self,
        job: Job,
        progress_percent: float | None = None,
        current_step: str | None = None,
    ) -> Job:
        """
        Update progress fields without a status transition.
        Called frequently by providers during execution — does NOT validate the
        state machine (progress stays within RUNNING).
        """
        if progress_percent is not None:
            job.progress_percent = progress_percent
        if current_step is not None:
            job.current_step = current_step
        return await self.update(job)

    async def increment_retry(self, job: Job) -> Job:
        """Increment retry_count in place."""
        job.retry_count += 1
        return await self.update(job)

    async def update_status(
        self,
        job: Job,
        new_status: JobStatus,
        *,
        worker_id: str | None = None,
        error_message: str | None = None,
        progress_percent: float | None = None,
        current_step: str | None = None,
        result_summary: dict | None = None,
    ) -> Job:
        """
        Transitions a job to a new status with optional field updates.
        Validates the state machine transition before writing.
        """
        if not job.status.can_transition_to(new_status):
            raise ValueError(
                f"Invalid transition: {job.status} → {new_status} for job {job.id}"
            )

        now = datetime.utcnow()
        job.status = new_status

        if new_status == JobStatus.QUEUED:
            job.queued_at = now
        elif new_status == JobStatus.SCHEDULED:
            job.scheduled_at = now
        elif new_status == JobStatus.RUNNING:
            job.started_at = now
        elif new_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            job.completed_at = now

        if worker_id is not None:
            job.worker_id = worker_id
        if error_message is not None:
            job.error_message = error_message
        if progress_percent is not None:
            job.progress_percent = progress_percent
        if current_step is not None:
            job.current_step = current_step
        if result_summary is not None:
            job.result_summary = result_summary

        return await self.update(job)

    # ─── Artifacts ────────────────────────────────────────────────────────────

    async def get_artifacts(self, job_id: UUID) -> list[Artifact]:
        stmt = select(Artifact).where(Artifact.job_id == job_id).order_by(Artifact.created_at)
        result = await self._s.execute(stmt)
        return list(result.scalars().all())

    async def create_artifact(self, artifact: Artifact) -> Artifact:
        self._s.add(artifact)
        await self._s.flush()
        await self._s.refresh(artifact)
        return artifact

    # ─── Events ───────────────────────────────────────────────────────────────

    async def create_event(self, event: JobEvent) -> JobEvent:
        self._s.add(event)
        await self._s.flush()
        return event

    async def get_events_since(
        self,
        job_id: UUID,
        since: datetime | None = None,
    ) -> list[JobEvent]:
        """Returns persisted events for a job, optionally filtered by timestamp."""
        stmt = select(JobEvent).where(JobEvent.job_id == job_id)
        if since:
            stmt = stmt.where(col(JobEvent.occurred_at) > since)
        stmt = stmt.order_by(col(JobEvent.occurred_at).asc())
        result = await self._s.execute(stmt)
        return list(result.scalars().all())
