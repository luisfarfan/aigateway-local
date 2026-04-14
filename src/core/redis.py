"""
Redis connection management.

Redis serves two roles in this system:
  1. ARQ queue backend — workers pull jobs from priority queues
  2. SSE pub/sub bus — workers publish events, API subscribes and streams to clients

We maintain a single connection pool for general use + a separate pub/sub connection
because Redis pub/sub connections cannot be reused for regular commands.
"""
import structlog
from arq.connections import ArqRedis, RedisSettings, create_pool
from redis.asyncio import Redis, ConnectionPool

from src.core.config import get_settings

log = structlog.get_logger(__name__)

_settings = get_settings()

# ─── Connection pool (general commands: GET, SET, PUBLISH) ────────────────────

_pool: ConnectionPool | None = None


def _make_pool() -> ConnectionPool:
    return ConnectionPool.from_url(
        _settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
    )


def get_redis() -> Redis:
    """
    Returns a Redis client backed by the shared connection pool.
    Safe to call from anywhere — the pool is initialised lazily on first call.
    """
    global _pool
    if _pool is None:
        _pool = _make_pool()
    return Redis(connection_pool=_pool)


async def close_redis() -> None:
    """Called at app shutdown to cleanly release all connections."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
    log.info("redis_pool_closed")


# ─── ARQ settings (used by both API and workers) ──────────────────────────────

def get_arq_redis_settings() -> RedisSettings:
    """
    ARQ-specific Redis connection settings.
    Used by the API when enqueuing jobs and by workers when registering.
    """
    return RedisSettings(
        host=_settings.redis_host,
        port=_settings.redis_port,
        password=_settings.redis_password or None,
        database=_settings.redis_db,
    )


# Cached ARQ pool — created once at API startup.
_arq_pool: ArqRedis | None = None


async def get_arq_pool() -> ArqRedis:
    """
    Returns the shared ARQ connection pool.
    Call once at startup (lifespan) and reuse throughout.
    """
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(get_arq_redis_settings())
        log.info("arq_pool_created")
    return _arq_pool


async def close_arq_pool() -> None:
    global _arq_pool
    if _arq_pool is not None:
        await _arq_pool.aclose()
        _arq_pool = None
    log.info("arq_pool_closed")


# ─── Pub/Sub channel naming ───────────────────────────────────────────────────

def job_channel(job_id: str) -> str:
    """Redis pub/sub channel name for a specific job's SSE events."""
    return f"job:{job_id}"


def client_channel(client_id: str) -> str:
    """Redis pub/sub channel for all events belonging to a specific client."""
    return f"client:{client_id}"
