"""
FastAPI application factory.

Creates and configures the app instance:
  - Lifespan (startup/shutdown)
  - Middleware (CORS, logging, error handlers)
  - Routers (jobs, events, registry, health)
  - Prometheus metrics
"""
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.lifespan import lifespan
from src.api.middleware import register_exception_handlers, register_middleware
from src.core.config import get_settings

settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local AI Gateway",
        description=(
            "Multimodal local AI gateway — queue jobs for text, audio, image, and video "
            "generation, and stream real-time progress via SSE."
        ),
        version="0.1.0",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # Middleware (order matters: added last = executed first)
    register_middleware(app)
    register_exception_handlers(app)

    # Prometheus metrics — exposes /metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # ── Routers ───────────────────────────────────────────────────────────────
    from src.modules.jobs.router import router as jobs_router
    from src.modules.events.router import router as events_router
    from src.modules.artifacts.router import router as artifacts_router
    from src.modules.uploads.router import router as uploads_router

    app.include_router(jobs_router, prefix=settings.api_prefix)
    app.include_router(events_router, prefix=settings.api_prefix)
    app.include_router(artifacts_router, prefix=settings.api_prefix)
    app.include_router(uploads_router, prefix=settings.api_prefix)

    # ── Health / readiness ────────────────────────────────────────────────────
    _register_health_routes(app)

    return app


def _register_health_routes(app: FastAPI) -> None:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from src.core.redis import get_redis

    @app.get("/health", tags=["Observability"], summary="Liveness probe")
    async def health():
        """Returns 200 if the API process is alive."""
        return {"status": "ok", "service": "local-ai-gateway"}

    @app.get("/ready", tags=["Observability"], summary="Readiness probe")
    async def ready(request: Request):
        """
        Returns 200 if all dependencies (DB, Redis, MinIO) are reachable.
        Returns 503 if any dependency is down.
        """
        checks: dict[str, str] = {}
        overall = True

        # Redis
        try:
            redis = get_redis()
            await redis.ping()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"error: {e}"
            overall = False

        # DB
        try:
            from src.core.database import engine
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as e:
            checks["database"] = f"error: {e}"
            overall = False

        # Provider registry
        try:
            registry = request.app.state.provider_registry
            checks["providers"] = str(registry.list_provider_ids())
        except Exception:
            checks["providers"] = "unavailable"

        status_code = 200 if overall else 503
        return JSONResponse(
            status_code=status_code,
            content={"status": "ready" if overall else "degraded", "checks": checks},
        )

    @app.get(
        f"{settings.api_prefix}/providers",
        tags=["Observability"],
        summary="List registered providers and capabilities",
    )
    async def list_providers(request: Request):
        registry = request.app.state.provider_registry
        return {
            "providers": [
                {
                    "provider_id": cap.provider_id,
                    "modality": cap.modality,
                    "supported_job_types": cap.supported_job_types,
                    "supported_models": cap.supported_models,
                    "max_concurrent_jobs": cap.max_concurrent_jobs,
                    "requires_gpu": cap.requires_gpu,
                }
                for cap in registry.list_capabilities()
            ]
        }


# Module-level app instance — used by uvicorn: `uvicorn src.api.app:app`
app = create_app()
