"""
Job executor — the core execution lifecycle for every job type.

Called by execute_job() in workers/main.py.
Extracted here for clarity and testability.

Implements the 4 critical correctness properties:
  1. Artifact DB persistence  — on_artifact creates Artifact record + presigned URL
  2. Scheduler integration    — ModalityScheduler gates execution per modality
  3. Timeout enforcement      — asyncio.wait_for wraps provider.execute()
  4. Retry with backoff       — failed jobs are re-enqueued with exponential delay

Execution phases:
  QUEUED → SCHEDULED  (load + validate, short session)
  SCHEDULED → RUNNING (acquire modality slot, open execution session)
  RUNNING → COMPLETED / FAILED / retried QUEUED
"""
import asyncio
import time
from datetime import timedelta
from typing import Any
from uuid import UUID

import structlog
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.database import AsyncSessionLocal
from src.core.domain import JobStatus
from src.core.metrics import active_jobs, inference_duration_seconds, jobs_total, queue_depth
from src.core.storage import storage
from src.modules.events.publisher import RedisOnlyPublisher
from src.modules.events.schemas import SSEEvents
from src.modules.jobs.models import Artifact
from src.modules.jobs.repository import JobRepository
from src.modules.providers.base import ExecutionContext, ProviderResult
from src.modules.providers.registry import ProviderRegistry
from src.modules.queue.dispatcher import EXECUTE_JOB_TASK
from src.modules.queue.scheduler import ModalityScheduler

log = structlog.get_logger(__name__)
settings = get_settings()


async def run_job(ctx: dict[str, Any], job_id_str: str) -> dict[str, Any]:
    """
    Top-level executor called by the ARQ task.
    Coordinates the full lifecycle across DB sessions and Redis.
    """
    job_id = UUID(job_id_str)
    registry: ProviderRegistry = ctx["registry"]
    redis: Redis = ctx["redis"]
    arq_pool = ctx["arq_pool"]
    worker_id: str = ctx["worker_id"]

    publisher = RedisOnlyPublisher(redis)
    structlog.contextvars.bind_contextvars(job_id=job_id_str, worker_id=worker_id)

    # ── Phase 1: load, validate, resolve provider ──────────────────────────
    async with AsyncSessionLocal() as session:
        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)

        if not job:
            log.error("execute_job_not_found")
            return {"error": "job_not_found"}

        if job.status.is_terminal():
            log.warning("execute_job_already_terminal", status=job.status)
            return {"skipped": True, "status": job.status}

        try:
            provider = registry.resolve(job.provider, job.type, job.model)
        except ValueError as e:
            await _fail_final(repo, job, publisher, str(e), session)
            return {"error": str(e)}

        # QUEUED → SCHEDULED (short commit, release connection)
        job = await repo.update_status(job, JobStatus.SCHEDULED, worker_id=worker_id)
        await session.commit()

    await publisher.publish(SSEEvents.scheduled(job_id, worker_id))
    log.info("execute_job_scheduled", provider=job.provider, type=job.type)

    # ── Phase 2: acquire modality concurrency slot ─────────────────────────
    # Blocks here until a slot is free. No DB connection held during the wait.
    scheduler = ModalityScheduler(redis)

    try:
        async with scheduler.slot(job.type, max_wait=300.0):
            await _run_with_session(
                job_id=job_id,
                job=job,
                provider=provider,
                publisher=publisher,
                arq_pool=arq_pool,
                worker_id=worker_id,
            )
    except TimeoutError as e:
        log.warning("execute_job_scheduler_timeout", modality=str(e))
        async with AsyncSessionLocal() as session:
            repo = JobRepository(session)
            fresh = await repo.get_by_id(job_id)
            if fresh and not fresh.status.is_terminal():
                await _fail_final(
                    repo, fresh, publisher,
                    f"Scheduler timeout: no {job.type} slot available after 300s.",
                    session,
                )

    return {"done": True}


async def _run_with_session(
    *,
    job_id: UUID,
    job: Any,
    provider: Any,
    publisher: RedisOnlyPublisher,
    arq_pool: Any,
    worker_id: str,
) -> None:
    """
    Runs the actual provider execution inside a single open DB session.
    The session stays open for the duration of inference so that progress
    callbacks can commit progress updates without reopening connections.
    """
    timeout = float(job.timeout_seconds or settings.job_default_timeout)
    artifact_records: list[Artifact] = []

    async with AsyncSessionLocal() as session:
        repo = JobRepository(session)

        # Reload job (might have been cancelled while waiting for scheduler slot)
        current_job = await repo.get_by_id(job_id)
        if not current_job or current_job.status == JobStatus.CANCELLED:
            log.info("execute_job_cancelled_before_start", job_id=str(job_id))
            return

        # SCHEDULED → RUNNING
        current_job = await repo.update_status(
            current_job, JobStatus.RUNNING, worker_id=worker_id
        )
        await session.commit()
        await publisher.publish(SSEEvents.started(job_id, worker_id))

        # Job left the queue — update queue depth metric
        queue_depth.labels(priority=str(current_job.priority.value)).dec()

        provider_id = str(current_job.provider or job.provider)
        job_type_str = str(current_job.type.value)

        # Track concurrency and wall-clock inference time
        active_jobs.labels(provider=provider_id, job_type=job_type_str).inc()
        t0 = time.monotonic()

        # ── Callbacks ─────────────────────────────────────────────────────

        async def on_progress(percent: float, step: str | None) -> None:
            """Called by providers at each progress step."""
            # Check cancellation on every progress tick
            fresh = await repo.get_by_id(job_id)
            if fresh and fresh.status == JobStatus.CANCELLED:
                raise asyncio.CancelledError("Job was cancelled")
            if fresh:
                await repo.update_progress(fresh, progress_percent=percent, current_step=step)
                await session.commit()
            await publisher.publish(SSEEvents.progress(job_id, percent, step=step))

        async def on_artifact(key: str, artifact_type: str, mime_type: str) -> None:
            """Called by providers when they produce an output file."""
            artifact = await _persist_artifact(
                job_id, key, artifact_type, mime_type, session, publisher
            )
            if artifact:
                artifact_records.append(artifact)

        # ── Execute with timeout ───────────────────────────────────────────
        exec_ctx = ExecutionContext(
            job_id=job_id,
            job_type=job.type,
            provider_id=job.provider,
            model=job.model,
            input_payload=job.input_payload,
            priority=job.priority,
            timeout_seconds=int(timeout),
            worker_id=worker_id,
            on_progress=on_progress,
            on_artifact=on_artifact,
        )

        result: ProviderResult
        try:
            result = await asyncio.wait_for(provider.execute(exec_ctx), timeout=timeout)
        except asyncio.TimeoutError:
            result = ProviderResult(
                success=False,
                error_message=f"Job timed out after {int(timeout)}s.",
            )
        except asyncio.CancelledError:
            log.info("execute_job_cancelled_mid_execution", job_id=str(job_id))
            active_jobs.labels(provider=provider_id, job_type=job_type_str).dec()
            return
        except Exception as e:
            log.exception("provider_execute_unhandled_error")
            result = ProviderResult(success=False, error_message=f"Unexpected error: {e}")
        finally:
            elapsed = time.monotonic() - t0
            inference_duration_seconds.labels(
                provider=provider_id, job_type=job_type_str
            ).observe(elapsed)

        # Decrement active count now that inference finished (success or failure)
        active_jobs.labels(provider=provider_id, job_type=job_type_str).dec()

        # ── Handle result ──────────────────────────────────────────────────
        final_job = await repo.get_by_id(job_id)
        if not final_job or final_job.status == JobStatus.CANCELLED:
            log.info("execute_job_cancelled_after_execution", job_id=str(job_id))
            return

        if result.success:
            artifact_urls = [a.public_url for a in artifact_records if a.public_url]
            await repo.update_status(
                final_job,
                JobStatus.COMPLETED,
                result_summary=result.result_summary,
                progress_percent=100.0,
                worker_id=worker_id,
            )
            await session.commit()
            await publisher.publish(
                SSEEvents.completed(job_id, result.result_summary, artifact_urls)
            )
            jobs_total.labels(
                job_type=job_type_str, status="completed", provider=provider_id
            ).inc()
            log.info(
                "execute_job_completed",
                job_id=str(job_id),
                artifacts=len(artifact_records),
            )
        else:
            await _handle_failure(
                repo=repo,
                job=final_job,
                error=result.error_message or "Unknown error",
                publisher=publisher,
                arq_pool=arq_pool,
                session=session,
            )


async def _handle_failure(
    *,
    repo: JobRepository,
    job: Any,
    error: str,
    publisher: RedisOnlyPublisher,
    arq_pool: Any,
    session: AsyncSession,
) -> None:
    """
    Decides whether to retry or permanently fail a job.

    Retry strategy: exponential backoff — 2^attempt seconds (2s, 4s, 8s).
    Retry emits a progress event (not terminal) so SSE clients stay connected.
    Final failure emits the terminal failed event and closes the SSE stream.
    """
    should_retry = job.retry_count < job.max_retries
    provider_id = str(job.provider or "unknown")
    job_type_str = str(job.type.value)

    if should_retry:
        delay_s = 2 ** (job.retry_count + 1)  # 2, 4, 8 seconds

        # RUNNING → FAILED (internal — no terminal SSE yet)
        job = await repo.update_status(
            job, JobStatus.FAILED, error_message=error, worker_id=None
        )
        job = await repo.increment_retry(job)
        await session.commit()

        # Inform client: retry coming (progress event, not terminal)
        await publisher.publish(
            SSEEvents.progress(
                job.id,
                0.0,
                message=f"Retrying ({job.retry_count}/{job.max_retries}) in {delay_s}s — {error}",
            )
        )

        # FAILED → QUEUED (retry path)
        job = await repo.update_status(job, JobStatus.QUEUED)
        await session.commit()

        # Re-enqueue with delay
        await arq_pool.enqueue_job(
            EXECUTE_JOB_TASK,
            str(job.id),
            _queue_name=job.priority.to_arq_queue(),
            _defer_by=timedelta(seconds=delay_s),
        )

        jobs_total.labels(job_type=job_type_str, status="retried", provider=provider_id).inc()
        queue_depth.labels(priority=str(job.priority.value)).inc()

        log.info(
            "execute_job_retrying",
            job_id=str(job.id),
            attempt=job.retry_count,
            max=job.max_retries,
            delay_s=delay_s,
            error=error,
        )
    else:
        jobs_total.labels(job_type=job_type_str, status="failed", provider=provider_id).inc()
        await _fail_final(repo, job, publisher, error, session)


async def _fail_final(
    repo: JobRepository,
    job: Any,
    publisher: RedisOnlyPublisher,
    error: str,
    session: AsyncSession,
) -> None:
    """Permanent failure — emits terminal SSE failed event."""
    await repo.update_status(job, JobStatus.FAILED, error_message=error)
    await session.commit()
    await publisher.publish(SSEEvents.failed(job.id, error))
    log.error("execute_job_failed_final", job_id=str(job.id), error=error)


async def _persist_artifact(
    job_id: UUID,
    storage_key: str,
    artifact_type: str,
    mime_type: str,
    session: AsyncSession,
    publisher: RedisOnlyPublisher,
) -> Artifact | None:
    """
    Creates an Artifact DB record for a file already uploaded to MinIO.
    Generates a presigned download URL and emits an artifact_ready SSE event.
    """
    try:
        size_bytes = await storage.get_size(storage_key)
        presigned_url = await storage.presigned_download_url(storage_key)
        filename = storage_key.split("/")[-1]

        artifact = Artifact(
            job_id=job_id,
            artifact_type=artifact_type,
            filename=filename,
            storage_key=storage_key,
            public_url=presigned_url,
            mime_type=mime_type,
            size_bytes=size_bytes,
        )
        session.add(artifact)
        await session.flush()
        await session.refresh(artifact)

        await publisher.publish(
            SSEEvents.artifact_ready(job_id, presigned_url, artifact_type, filename)
        )
        log.info(
            "artifact_persisted",
            key=storage_key,
            type=artifact_type,
            size_bytes=size_bytes,
        )
        return artifact

    except Exception:
        log.exception("artifact_persist_failed", key=storage_key)
        return None
