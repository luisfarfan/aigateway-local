"""
SSE event publisher — the bridge between workers/services and SSE clients.

Responsibilities:
  1. Serialize SSEEvent to JSON
  2. PUBLISH to Redis channel `job:{job_id}` (crosses process boundary to SSE endpoint)
  3. Persist to `job_events` table (for replay on client reconnect)

Workers and services call `await publisher.publish(event)`.
They don't know about Redis channels or SSE wire format.
"""
from uuid import UUID

import structlog
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.redis import job_channel
from src.modules.events.schemas import SSEEvent
from src.modules.jobs.models import JobEvent

log = structlog.get_logger(__name__)


class EventPublisher:
    """
    Dual-write publisher: Redis pub/sub + DB persistence.

    Instantiated per request/task — receives a session and redis client.
    """

    def __init__(self, redis: Redis, session: AsyncSession) -> None:
        self._redis = redis
        self._session = session

    async def publish(self, event: SSEEvent) -> None:
        """
        Publish an SSE event to Redis and persist it to the DB.
        Both writes happen; failure of either is logged but not fatal
        (workers should not crash because an event failed to persist).
        """
        channel = job_channel(str(event.job_id))
        payload = event.model_dump_json()

        # Publish to Redis (crosses process boundary → SSE endpoint)
        try:
            await self._redis.publish(channel, payload)
        except Exception:
            log.exception("event_redis_publish_failed", job_id=str(event.job_id))

        # Persist for replay (Last-Event-ID reconnection)
        try:
            db_event = JobEvent(
                job_id=event.job_id,
                event_type=event.event_type.value,
                payload=event.model_dump(mode="json"),
            )
            self._session.add(db_event)
            await self._session.flush()
        except Exception:
            log.exception("event_db_persist_failed", job_id=str(event.job_id))

        log.debug(
            "event_published",
            job_id=str(event.job_id),
            event_type=event.event_type,
            status=event.status,
            percent=event.progress_percent,
        )


class RedisOnlyPublisher:
    """
    Lightweight publisher that only writes to Redis — no DB session needed.
    Used inside ARQ tasks where the session lifecycle is managed separately.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def publish(self, event: SSEEvent) -> None:
        channel = job_channel(str(event.job_id))
        try:
            await self._redis.publish(channel, event.model_dump_json())
        except Exception:
            log.exception("event_redis_publish_failed", job_id=str(event.job_id))
