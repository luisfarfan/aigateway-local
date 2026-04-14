"""
Modality scheduler — Redis-based concurrency semaphore per modality.

Prevents VRAM/RAM overload by limiting how many jobs of each type run simultaneously.
Workers acquire a slot before execution and release it when done (always, even on failure).

Limits are configured via env vars:
  QUEUE_IMAGE_CONCURRENCY=1   # default: only 1 image job at a time (GPU)
  QUEUE_TEXT_CONCURRENCY=4    # text is cheaper, allow more
  etc.

Redis keys:
  sema:modality:{modality}  →  integer counter of currently running jobs
"""
import asyncio
from contextlib import asynccontextmanager

import structlog
from redis.asyncio import Redis

from src.core.config import get_settings
from src.core.domain import JOB_TYPE_TO_MODALITY, JobType, Modality

log = structlog.get_logger(__name__)
settings = get_settings()

# Map modality → configured concurrency limit
_MODALITY_LIMITS: dict[Modality, int] = {
    Modality.TEXT: settings.queue_text_concurrency,
    Modality.AUDIO: settings.queue_audio_concurrency,
    Modality.IMAGE: settings.queue_image_concurrency,
    Modality.VIDEO: settings.queue_video_concurrency,
    Modality.PIPELINE: settings.queue_pipeline_concurrency,
}

_SEMA_KEY = "sema:modality:{modality}"
_SEMA_TTL = 7200  # safety TTL — counters auto-expire if a worker crashes mid-job


class ModalityScheduler:
    """
    Redis-based concurrency gate.

    Usage:
        scheduler = ModalityScheduler(redis)
        async with scheduler.slot(job_type):
            await provider.execute(...)
        # slot released automatically
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def _key(self, modality: Modality) -> str:
        return _SEMA_KEY.format(modality=modality.value)

    def _limit(self, modality: Modality) -> int:
        return _MODALITY_LIMITS.get(modality, 1)

    async def acquire(self, modality: Modality) -> bool:
        """
        Attempts to acquire a concurrency slot for the given modality.
        Returns True if acquired, False if at capacity.
        Uses a Redis Lua script for atomic check-and-increment.
        """
        key = self._key(modality)
        limit = self._limit(modality)

        # Atomic: only increment if current value < limit
        script = """
        local current = tonumber(redis.call('GET', KEYS[1]) or '0')
        if current < tonumber(ARGV[1]) then
            redis.call('INCR', KEYS[1])
            redis.call('EXPIRE', KEYS[1], ARGV[2])
            return 1
        end
        return 0
        """
        result = await self._redis.eval(script, 1, key, limit, _SEMA_TTL)
        acquired = bool(result)

        log.debug(
            "scheduler_acquire",
            modality=modality,
            acquired=acquired,
            limit=limit,
        )
        return acquired

    async def release(self, modality: Modality) -> None:
        """Releases a concurrency slot. Always call this, even on failure."""
        key = self._key(modality)
        # Decrement but never below 0
        script = """
        local current = tonumber(redis.call('GET', KEYS[1]) or '0')
        if current > 0 then
            redis.call('DECR', KEYS[1])
        end
        return 1
        """
        await self._redis.eval(script, 1, key)
        log.debug("scheduler_release", modality=modality)

    @asynccontextmanager
    async def slot(self, job_type: JobType, poll_interval: float = 2.0, max_wait: float = 300.0):
        """
        Async context manager that waits for a slot to become available.
        Polls every `poll_interval` seconds up to `max_wait` seconds.
        Raises TimeoutError if no slot becomes available within max_wait.
        """
        modality = JOB_TYPE_TO_MODALITY[job_type]
        waited = 0.0

        while True:
            acquired = await self.acquire(modality)
            if acquired:
                break
            if waited >= max_wait:
                raise TimeoutError(
                    f"No {modality} slot available after {max_wait}s. "
                    f"Limit: {self._limit(modality)}"
                )
            log.info(
                "scheduler_waiting",
                modality=modality,
                waited_s=waited,
                limit=self._limit(modality),
            )
            await asyncio.sleep(poll_interval)
            waited += poll_interval

        try:
            yield
        finally:
            await self.release(modality)

    async def current_usage(self) -> dict[str, dict[str, int]]:
        """Returns current slot usage for all modalities. Used by /ready endpoint."""
        result = {}
        for modality in Modality:
            key = self._key(modality)
            raw = await self._redis.get(key)
            current = int(raw) if raw else 0
            limit = self._limit(modality)
            result[modality.value] = {"current": current, "limit": limit}
        return result
