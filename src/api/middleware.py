"""
Global middleware and exception handlers.

Registered on the FastAPI app in app.py.
Handles: CORS, structured error responses, exception → HTTP mapping.
"""
import time

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.exceptions import (
    AuthenticationError,
    GatewayError,
    JobCancellationError,
    JobNotFoundError,
    ProviderNotFoundError,
    ProviderNotSupportedError,
    RateLimitError,
)

log = structlog.get_logger(__name__)


def register_middleware(app: FastAPI) -> None:
    """Attach all middleware to the app. Called from app.py."""

    # CORS — allow all origins in dev; restrict in production via env
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        """Logs every request with timing. Binds request context for structlog."""
        request_id = request.headers.get("X-Request-ID", "")
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        log.info(
            "http_request",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """Map domain exceptions to HTTP responses. Called from app.py."""

    @app.exception_handler(JobNotFoundError)
    async def job_not_found(request: Request, exc: JobNotFoundError):
        return _error(status.HTTP_404_NOT_FOUND, exc.message)

    @app.exception_handler(JobCancellationError)
    async def job_cancellation(request: Request, exc: JobCancellationError):
        return _error(status.HTTP_409_CONFLICT, exc.message)

    @app.exception_handler(ProviderNotFoundError)
    async def provider_not_found(request: Request, exc: ProviderNotFoundError):
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.message)

    @app.exception_handler(ProviderNotSupportedError)
    async def provider_not_supported(request: Request, exc: ProviderNotSupportedError):
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.message)

    @app.exception_handler(AuthenticationError)
    async def auth_error(request: Request, exc: AuthenticationError):
        return _error(status.HTTP_401_UNAUTHORIZED, exc.message)

    @app.exception_handler(RateLimitError)
    async def rate_limit(request: Request, exc: RateLimitError):
        return _error(status.HTTP_429_TOO_MANY_REQUESTS, exc.message)

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError):
        log.warning("unhandled_gateway_error", error=exc.message)
        return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, exc.message)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled_exception")
        return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error.")


def _error(code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"detail": detail})
