"""
FastAPI lifespan — startup and shutdown hooks.

Startup order:
  1. Logging
  2. DB tables (dev) / verify connection (prod)
  3. MinIO bucket
  4. Redis + ARQ pool
  5. Register providers in ProviderRegistry
  6. Attach shared state to app

Shutdown order (reverse):
  1. ARQ pool
  2. Redis pool
  3. DB engine
"""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.core.config import get_settings
from src.core.database import create_all_tables, dispose_engine
from src.core.logging import configure_logging
from src.core.redis import close_arq_pool, close_redis, get_arq_pool
from src.core.storage import storage
from src.modules.providers.registry import ProviderRegistry
from src.modules.providers.stub.provider import StubProvider

log = structlog.get_logger(__name__)


def _build_provider_registry() -> ProviderRegistry:
    """
    Instantiates and registers all available provider adapters.

    Control which providers load via env vars (set in .env):
      ENABLE_PROVIDER_STUB=true        always on in development
      ENABLE_PROVIDER_DIFFUSERS=true   requires diffusers + torch on Ubuntu
      ENABLE_PROVIDER_LOCAL_LLM=true   requires Ollama running or transformers
      ENABLE_PROVIDER_LOCAL_TTS=true   requires TTS engine installed

    To add a new engine:
      1. Create its adapter in src/modules/providers/{name}/provider.py
      2. Import it here and add its enable flag below.
    """
    import os
    registry = ProviderRegistry()

    # Stub — always enabled in dev, optional in prod
    if os.environ.get("ENABLE_PROVIDER_STUB", "true").lower() == "true":
        registry.register(StubProvider(step_delay_seconds=1.5))

    # Diffusers — image/video generation via HuggingFace Diffusers
    if os.environ.get("ENABLE_PROVIDER_DIFFUSERS", "false").lower() == "true":
        try:
            from src.modules.providers.diffusers.provider import DiffusersProvider
            registry.register(DiffusersProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="diffusers", error=str(e))

    # Local LLM — text generation via Ollama or HuggingFace Transformers
    if os.environ.get("ENABLE_PROVIDER_LOCAL_LLM", "false").lower() == "true":
        try:
            from src.modules.providers.local_llm.provider import LocalLLMProvider
            registry.register(LocalLLMProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="local_llm", error=str(e))

    # Local TTS — text-to-speech via XTTS / Kokoro / Piper
    if os.environ.get("ENABLE_PROVIDER_LOCAL_TTS", "false").lower() == "true":
        try:
            from src.modules.providers.local_tts.provider import LocalTTSProvider
            registry.register(LocalTTSProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="local_tts", error=str(e))

    registered = registry.list_provider_ids()
    log.info("providers_registered", providers=registered, count=len(registered))
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # ── Startup ──────────────────────────────────────────────────────────────
    configure_logging()
    log.info("gateway_starting", environment=settings.environment)

    # Database
    await create_all_tables()  # dev: creates tables; prod: use `alembic upgrade head`
    log.info("database_ready")

    # Object storage
    await storage.ensure_bucket()
    log.info("storage_ready", bucket=settings.minio_bucket)

    # Redis + ARQ
    arq_pool = await get_arq_pool()
    log.info("redis_ready")

    # Provider registry
    registry = _build_provider_registry()

    # Attach shared state — accessible in routers via request.app.state.*
    app.state.provider_registry = registry
    app.state.arq_pool = arq_pool

    log.info("gateway_started", host=settings.api_host, port=settings.api_port)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("gateway_stopping")
    await close_arq_pool()
    await close_redis()
    await dispose_engine()
    log.info("gateway_stopped")
