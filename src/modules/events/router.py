"""
SSE event streaming endpoint.

Architecture (per ADR-005):
  Worker → PUBLISH Redis channel → SUBSCRIBE here → SSE stream → Client

Reconnection support:
  If the client sends Last-Event-ID (browser EventSource does this automatically),
  we replay missed events from the job_events table before subscribing to Redis.
"""
import asyncio
import json
from datetime import datetime, timezone
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from src.core.database import get_session
from src.core.redis import get_redis, job_channel
from src.modules.auth.middleware import get_current_client_id
from src.modules.events.schemas import SSEEvent, SSEEvents
from src.modules.jobs.models import Job
from src.modules.jobs.repository import JobRepository

log = structlog.get_logger(__name__)
router = APIRouter(tags=["Events"])

# Polling interval when no Redis message arrives (seconds)
_POLL_INTERVAL = 0.5
# Heartbeat interval — emitted to keep the connection alive (seconds)
_HEARTBEAT_INTERVAL = 15


@router.get(
    "/jobs/{job_id}/events",
    summary="Subscribe to real-time SSE events for a job",
    response_class=EventSourceResponse,
)
async def job_event_stream(
    job_id: UUID,
    request: Request,
    client_id: str = Depends(get_current_client_id),
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    """
    Server-Sent Events stream for a single job.

    Connect with:
        const es = new EventSource('/api/v1/jobs/{id}/events', {
            headers: { Authorization: 'Bearer <key>' }
        });
        es.addEventListener('progress', e => console.log(JSON.parse(e.data)));

    The stream closes automatically on terminal events (completed/failed/cancelled).
    On reconnect, the browser sends `Last-Event-ID` and missed events are replayed.

    Event types: job_created · queued · scheduled · started · progress ·
                 heartbeat · artifact_ready · completed · failed · cancelled
    """
    redis = get_redis()
    repo = JobRepository(session)

    # Ownership check — clients may only subscribe to their own jobs
    job = await repo.get_by_id(job_id)
    if not job or job.client_id != client_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found.")

    # Extract Last-Event-ID for replay (format: "{job_id}:{timestamp_ms}")
    last_event_id = request.headers.get("Last-Event-ID")
    replay_since: datetime | None = None
    if last_event_id:
        try:
            _, ts_ms = last_event_id.rsplit(":", 1)
            replay_since = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
        except (ValueError, IndexError):
            pass  # malformed Last-Event-ID → ignore, stream from now

    return EventSourceResponse(
        _event_generator(job_id, job, redis, repo, request, replay_since),
        ping=_HEARTBEAT_INTERVAL,
    )


async def _event_generator(
    job_id: UUID,
    job: Job,
    redis: Redis,
    repo: JobRepository,
    request: Request,
    replay_since: datetime | None,
):
    """
    Async generator that yields SSE messages.

    1. If replay_since is set, yield missed events from DB first.
    2. If the job is already in a terminal state, close immediately.
    3. Otherwise, subscribe to Redis and stream live events.
    """
    structlog.contextvars.bind_contextvars(job_id=str(job_id))

    # Phase 1: replay missed events
    if replay_since is not None:
        missed = await repo.get_events_since(job_id, since=replay_since)
        for db_event in missed:
            try:
                event = SSEEvent.model_validate(db_event.payload)
                msg = event.to_sse_message()
                yield ServerSentEvent(**msg)
                if event.is_terminal():
                    log.debug("sse_replay_terminal", job_id=str(job_id))
                    return
            except Exception:
                log.exception("sse_replay_parse_error")

    # Phase 2: if job already terminal (no replay needed, client connects late)
    if job.status.is_terminal() and replay_since is None:
        # Emit a synthetic event so the client knows the final state
        if job.status.value == "completed":
            event = SSEEvents.completed(job_id, job.result_summary)
        elif job.status.value == "failed":
            event = SSEEvents.failed(job_id, job.error_message or "Unknown error")
        else:
            event = SSEEvents.cancelled(job_id)
        yield ServerSentEvent(**event.to_sse_message())
        return

    # Phase 3: live stream from Redis pub/sub
    channel = job_channel(str(job_id))
    pubsub = redis.pubsub()

    try:
        await pubsub.subscribe(channel)
        log.debug("sse_subscribed", channel=channel)

        last_heartbeat = asyncio.get_event_loop().time()

        while True:
            # Client disconnected
            if await request.is_disconnected():
                log.debug("sse_client_disconnected", job_id=str(job_id))
                break

            # Try to get a message (non-blocking, short timeout)
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=_POLL_INTERVAL,
                )
            except asyncio.TimeoutError:
                message = None

            now = asyncio.get_event_loop().time()

            if message and message["type"] == "message":
                try:
                    event = SSEEvent.model_validate_json(message["data"])
                    yield ServerSentEvent(**event.to_sse_message())

                    if event.is_terminal():
                        log.info("sse_stream_terminal", job_id=str(job_id), event_type=event.event_type)
                        break

                    last_heartbeat = now
                except Exception:
                    log.exception("sse_message_parse_error")
                    continue

            # Heartbeat — keeps the connection alive and signals the worker is running
            elif now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                heartbeat = SSEEvents.heartbeat(job_id)
                yield ServerSentEvent(**heartbeat.to_sse_message())
                last_heartbeat = now

    except Exception:
        log.exception("sse_generator_error", job_id=str(job_id))
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
        except Exception:
            pass
        log.debug("sse_stream_closed", job_id=str(job_id))
