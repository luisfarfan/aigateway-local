"""
ARQ worker entrypoint.

Responsibilities:
  - Define WorkerSettings for ARQ (queues, concurrency, startup/shutdown)
  - Register provider adapters on startup
  - Delegate all execution logic to workers/executor.py

Run with:
  python -m workers.main
  (or via: make worker)
"""
import uuid
from typing import Any

import structlog
from arq import Worker
from arq.connections import RedisSettings

from src.core.config import get_settings
from src.core.database import create_all_tables
from src.core.logging import configure_logging
from src.core.redis import get_arq_pool, get_arq_redis_settings, get_redis
from src.core.storage import storage
from src.modules.providers.registry import ProviderRegistry
from src.modules.providers.stub.provider import StubProvider
from workers.executor import run_job

log = structlog.get_logger(__name__)
settings = get_settings()


# ─── Startup / shutdown ───────────────────────────────────────────────────────

async def startup(ctx: dict[str, Any]) -> None:
    """Called once when the ARQ worker process starts."""
    configure_logging()

    worker_id = settings.worker_id or f"worker-{uuid.uuid4().hex[:8]}"
    structlog.contextvars.bind_contextvars(worker_id=worker_id)
    log.info("worker_starting", worker_id=worker_id, modalities=settings.enabled_modalities)

    # DB tables (dev) — prod: `alembic upgrade head`
    await create_all_tables()

    # MinIO bucket
    await storage.ensure_bucket()

    # Provider registry — same flag-based approach as API lifespan
    import os
    registry = ProviderRegistry()

    if os.environ.get("ENABLE_PROVIDER_STUB", "true").lower() == "true":
        registry.register(StubProvider(step_delay_seconds=1.5))

    if os.environ.get("ENABLE_PROVIDER_DIFFUSERS", "false").lower() == "true":
        try:
            from src.modules.providers.diffusers.provider import DiffusersProvider
            registry.register(DiffusersProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="diffusers", error=str(e))

    if os.environ.get("ENABLE_PROVIDER_LOCAL_LLM", "false").lower() == "true":
        try:
            from src.modules.providers.local_llm.provider import LocalLLMProvider
            registry.register(LocalLLMProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="local_llm", error=str(e))

    if os.environ.get("ENABLE_PROVIDER_LOCAL_TTS", "false").lower() == "true":
        try:
            from src.modules.providers.local_tts.provider import LocalTTSProvider
            registry.register(LocalTTSProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="local_tts", error=str(e))

    if os.environ.get("ENABLE_PROVIDER_LOCAL_STT", "false").lower() == "true":
        try:
            from src.modules.providers.local_stt.provider import LocalSTTProvider
            registry.register(LocalSTTProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="local_stt", error=str(e))

    # Initialize all loaded providers (loads models into GPU/memory)
    for pid in registry.list_provider_ids():
        provider = registry.get(pid)
        if provider:
            await provider.initialize()
            log.info("provider_initialized", provider_id=pid)

    # Shared state available in every task via ctx
    ctx["registry"] = registry
    ctx["redis"] = get_redis()
    ctx["arq_pool"] = await get_arq_pool()   # needed for retry re-enqueueing
    ctx["worker_id"] = worker_id

    log.info(
        "worker_ready",
        worker_id=worker_id,
        providers=registry.list_provider_ids(),
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    """Called at worker shutdown — teardown providers and release resources."""
    registry: ProviderRegistry | None = ctx.get("registry")
    if registry:
        for pid in registry.list_provider_ids():
            provider = registry.get(pid)
            if provider:
                try:
                    await provider.teardown()
                    log.info("provider_teardown", provider_id=pid)
                except Exception:
                    log.exception("provider_teardown_error", provider_id=pid)

    log.info("worker_stopped", worker_id=ctx.get("worker_id"))


# ─── ARQ task ────────────────────────────────────────────────────────────────

async def execute_job(ctx: dict[str, Any], job_id_str: str) -> dict[str, Any]:
    """
    ARQ task entry point — called by the queue for every dequeued job.
    Delegates the full execution lifecycle to workers/executor.py.
    """
    return await run_job(ctx, job_id_str)


# ─── ARQ WorkerSettings ───────────────────────────────────────────────────────

class WorkerSettings:
    """
    ARQ reads this class to configure the worker process.
    https://arq-docs.helpmanual.io/
    """
    functions = [execute_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = get_arq_redis_settings()

    # Priority queues — worker drains "high" before "normal" before "low"
    queues = ["high", "normal", "low"]
    queue_read_limit = 10

    # Global concurrency limit (overridden per modality by ModalityScheduler)
    max_jobs = settings.queue_global_concurrency

    job_timeout = settings.job_default_timeout
    keep_result = settings.arq_result_ttl

    # ARQ-level retries disabled — we handle retries in executor.py with
    # exponential backoff and proper state machine transitions
    max_tries = 1


if __name__ == "__main__":
    from arq import run_worker
    run_worker(WorkerSettings)
