"""
SSE event contract — every event the system can emit, with a common envelope.

Wire format over the HTTP connection:
    event: job_created
    data: {"event_type": "job_created", "job_id": "...", "timestamp": "...", ...}
    id: {job_id}:{timestamp_ms}

The `id` field is used by the browser's EventSource to send `Last-Event-ID` on reconnect,
enabling the API to replay missed events from the job_events table.
"""
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.core.domain import JobStatus, SSEEventType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SSEEvent(BaseModel):
    """
    Canonical SSE event envelope. All events use this exact structure.

    Workers create these via SSEEvents factory methods and publish them to Redis.
    The SSE endpoint subscribes, serializes to SSE wire format, and streams to the client.
    """
    event_type: SSEEventType
    job_id: UUID
    timestamp: datetime = Field(default_factory=_utcnow)
    status: JobStatus | None = None
    progress_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifact_url: str | None = None

    def to_sse_message(self) -> dict[str, str]:
        """
        Returns the dict consumed by sse-starlette's ServerSentEvent.
        Keys: event, data, id.
        """
        return {
            "event": self.event_type.value,
            "data": self.model_dump_json(),
            "id": f"{self.job_id}:{int(self.timestamp.timestamp() * 1000)}",
        }

    def is_terminal(self) -> bool:
        """Terminal events signal that no more events will be emitted for this job."""
        return self.event_type in (
            SSEEventType.COMPLETED,
            SSEEventType.FAILED,
            SSEEventType.CANCELLED,
        )


class SSEEvents:
    """
    Factory for creating well-formed SSE events.
    Use these instead of constructing SSEEvent directly — they enforce correct field population.
    """

    @staticmethod
    def job_created(job_id: UUID, provider: str, model: str | None) -> SSEEvent:
        return SSEEvent(
            event_type=SSEEventType.JOB_CREATED,
            job_id=job_id,
            status=JobStatus.QUEUED,
            metadata={"provider": provider, "model": model},
        )

    @staticmethod
    def queued(job_id: UUID, queue_position: int | None = None) -> SSEEvent:
        return SSEEvent(
            event_type=SSEEventType.QUEUED,
            job_id=job_id,
            status=JobStatus.QUEUED,
            metadata={"queue_position": queue_position},
        )

    @staticmethod
    def scheduled(job_id: UUID, worker_id: str) -> SSEEvent:
        return SSEEvent(
            event_type=SSEEventType.SCHEDULED,
            job_id=job_id,
            status=JobStatus.SCHEDULED,
            metadata={"worker_id": worker_id},
        )

    @staticmethod
    def started(job_id: UUID, worker_id: str) -> SSEEvent:
        return SSEEvent(
            event_type=SSEEventType.STARTED,
            job_id=job_id,
            status=JobStatus.RUNNING,
            progress_percent=0.0,
            metadata={"worker_id": worker_id},
        )

    @staticmethod
    def progress(
        job_id: UUID,
        percent: float,
        message: str | None = None,
        step: str | None = None,
    ) -> SSEEvent:
        return SSEEvent(
            event_type=SSEEventType.PROGRESS,
            job_id=job_id,
            status=JobStatus.RUNNING,
            progress_percent=percent,
            message=message,
            metadata={"step": step} if step else {},
        )

    @staticmethod
    def heartbeat(job_id: UUID, percent: float | None = None) -> SSEEvent:
        """
        Emitted every ~15s during long-running jobs to keep the SSE connection alive
        and signal that the worker is still active.
        """
        return SSEEvent(
            event_type=SSEEventType.HEARTBEAT,
            job_id=job_id,
            status=JobStatus.RUNNING,
            progress_percent=percent,
        )

    @staticmethod
    def artifact_ready(
        job_id: UUID,
        artifact_url: str,
        artifact_type: str,
        filename: str,
    ) -> SSEEvent:
        return SSEEvent(
            event_type=SSEEventType.ARTIFACT_READY,
            job_id=job_id,
            status=JobStatus.RUNNING,
            artifact_url=artifact_url,
            metadata={"artifact_type": artifact_type, "filename": filename},
        )

    @staticmethod
    def completed(
        job_id: UUID,
        result_summary: dict[str, Any] | None = None,
        artifact_urls: list[str] | None = None,
    ) -> SSEEvent:
        return SSEEvent(
            event_type=SSEEventType.COMPLETED,
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress_percent=100.0,
            metadata={
                "result_summary": result_summary or {},
                "artifact_urls": artifact_urls or [],
            },
        )

    @staticmethod
    def failed(
        job_id: UUID,
        error: str,
        detail: dict[str, Any] | None = None,
    ) -> SSEEvent:
        return SSEEvent(
            event_type=SSEEventType.FAILED,
            job_id=job_id,
            status=JobStatus.FAILED,
            message=error,
            metadata={"detail": detail or {}},
        )

    @staticmethod
    def cancelled(job_id: UUID, reason: str | None = None) -> SSEEvent:
        return SSEEvent(
            event_type=SSEEventType.CANCELLED,
            job_id=job_id,
            status=JobStatus.CANCELLED,
            message=reason,
        )
