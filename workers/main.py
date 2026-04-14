"""
ARQ worker entrypoint.

This process pulls jobs from the Redis priority queues and executes them.
Run with: python -m workers.main

Architecture:
  - ARQ calls `execute_job(ctx, job_id_str)` for each dequeued task.
  - `execute_job` loads the Job from DB, resolves the provider via registry,
    runs the provider, and updates DB + emits SSE events throughout.

Phase 3 will implement the full execution logic inside execute_job.
This file wires the ARQ machinery correctly.
"""
import asyncio
import uuid
from typing import Any

import structlog
from arq import Worker
from arq.connections import RedisSettings

from src.core.config import get_settings
from src.core.database import AsyncSessionLocal, create_all_tables
from src.core.logging import configure_logging
from src.core.redis import get_arq_redis_settings, get_redis
from src.core.storage import storage
from src.modules.providers.registry import ProviderRegistry
from src.modules.providers.stub.provider import StubProvider

log = structlog.get_logger(__name__)
settings = get_settings()


# ─── Startup / shutdown ───────────────────────────────────────────────────────

async def startup(ctx: dict[str, Any]) -> None:
    """Called once when the ARQ worker process starts."""
    configure_logging()

    worker_id = settings.worker_id or f"worker-{uuid.uuid4().hex[:8]}"
    log.info("worker_starting", worker_id=worker_id)

    # DB
    await create_all_tables()

    # Storage
    await storage.ensure_bucket()

    # Provider registry — same as API but for the worker process
    registry = ProviderRegistry()
    registry.register(StubProvider(step_delay_seconds=1.5))
    # registry.register(DiffusersProvider())
    # registry.register(LocalTTSProvider())

    # Initialize all providers (loads models into memory / GPU)
    for pid in registry.list_provider_ids():
        provider = registry.get(pid)
        if provider:
            await provider.initialize()
            log.info("provider_initialized", provider_id=pid)

    ctx["registry"] = registry
    ctx["redis"] = get_redis()
    ctx["worker_id"] = worker_id
    log.info("worker_ready", worker_id=worker_id, providers=registry.list_provider_ids())


async def shutdown(ctx: dict[str, Any]) -> None:
    """Called when the ARQ worker process shuts down."""
    registry: ProviderRegistry = ctx.get("registry")
    if registry:
        for pid in registry.list_provider_ids():
            provider = registry.get(pid)
            if provider:
                await provider.teardown()
                log.info("provider_teardown", provider_id=pid)
    log.info("worker_stopped", worker_id=ctx.get("worker_id"))


# ─── Tasks ────────────────────────────────────────────────────────────────────

async def execute_job(ctx: dict[str, Any], job_id_str: str) -> dict[str, Any]:
    """
    Main ARQ task — entry point for all job types.

    Receives the domain job_id as a string, loads the job from DB,
    routes it to the correct provider, and runs the full execution lifecycle.

    Full implementation lives in Phase 3 (workers/executor.py).
    This stub validates the wiring is correct end-to-end.
    """
    from uuid import UUID

    from src.modules.jobs.repository import JobRepository
    from src.modules.events.publisher import RedisOnlyPublisher
    from src.modules.events.schemas import SSEEvents
    from src.core.domain import JobStatus

    job_id = UUID(job_id_str)
    registry: ProviderRegistry = ctx["registry"]
    redis = ctx["redis"]
    worker_id: str = ctx["worker_id"]

    structlog.contextvars.bind_contextvars(job_id=job_id_str, worker_id=worker_id)
    publisher = RedisOnlyPublisher(redis)

    async with AsyncSessionLocal() as session:
        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)

        if not job:
            log.error("execute_job_not_found", job_id=job_id_str)
            return {"error": "job_not_found"}

        if job.status.is_terminal():
            log.warning("execute_job_already_terminal", status=job.status)
            return {"skipped": True, "status": job.status}

        log.info("execute_job_start", type=job.type, provider=job.provider)

        try:
            # Resolve provider
            provider = registry.resolve(job.provider, job.type, job.model)
        except ValueError as e:
            await repo.update_status(
                job, JobStatus.FAILED, error_message=str(e), worker_id=worker_id
            )
            await publisher.publish(SSEEvents.failed(job_id, str(e)))
            await session.commit()
            return {"error": str(e)}

        # Transition: QUEUED → SCHEDULED → RUNNING
        await repo.update_status(job, JobStatus.SCHEDULED, worker_id=worker_id)
        await session.commit()
        await publisher.publish(SSEEvents.scheduled(job_id, worker_id))

        await repo.update_status(job, JobStatus.RUNNING, worker_id=worker_id)
        await session.commit()
        await publisher.publish(SSEEvents.started(job_id, worker_id))

        # --- Full execution delegated to Phase 3 executor ---
        # For now: call provider directly (no artifact persistence yet)
        from src.modules.providers.base import ExecutionContext

        artifact_keys: list[str] = []

        async def on_progress(percent: float, step: str | None) -> None:
            await repo.update_status(
                job, JobStatus.RUNNING,
                progress_percent=percent, current_step=step, worker_id=worker_id
            )
            await session.commit()
            await publisher.publish(SSEEvents.progress(job_id, percent, step=step))

        async def on_artifact(key: str, artifact_type: str, mime_type: str) -> None:
            artifact_keys.append(key)
            log.info("artifact_produced", key=key, type=artifact_type)

        exec_ctx = ExecutionContext(
            job_id=job_id,
            job_type=job.type,
            provider_id=job.provider,
            model=job.model,
            input_payload=job.input_payload,
            priority=job.priority,
            timeout_seconds=job.timeout_seconds or settings.job_default_timeout,
            worker_id=worker_id,
            on_progress=on_progress,
            on_artifact=on_artifact,
        )

        result = await provider.execute(exec_ctx)

        if result.success:
            await repo.update_status(
                job, JobStatus.COMPLETED,
                result_summary=result.result_summary,
                progress_percent=100.0,
                worker_id=worker_id,
            )
            await session.commit()
            await publisher.publish(
                SSEEvents.completed(job_id, result.result_summary, artifact_keys)
            )
            log.info("execute_job_completed", job_id=job_id_str)
            return {"success": True, "summary": result.result_summary}
        else:
            await repo.update_status(
                job, JobStatus.FAILED,
                error_message=result.error_message,
                worker_id=worker_id,
            )
            await session.commit()
            await publisher.publish(SSEEvents.failed(job_id, result.error_message or ""))
            log.error("execute_job_failed", error=result.error_message)
            return {"success": False, "error": result.error_message}


# ─── ARQ WorkerSettings ───────────────────────────────────────────────────────

class WorkerSettings:
    """
    ARQ reads this class to configure the worker.
    See: https://arq-docs.helpmanual.io/
    """
    functions = [execute_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = get_arq_redis_settings()

    # Listen to all priority queues — high first, then normal, then low
    queue_read_limit = 10
    queues = ["high", "normal", "low"]

    max_jobs = settings.queue_global_concurrency
    job_timeout = settings.job_default_timeout
    keep_result = settings.arq_result_ttl
    retry_jobs = True
    max_tries = 1  # we handle retries at the domain level


if __name__ == "__main__":
    # python -m workers.main
    from arq import run_worker
    run_worker(WorkerSettings)
