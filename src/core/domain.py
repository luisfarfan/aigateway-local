"""
Core domain types — enums, value objects, and constants shared across all modules.

Rules:
  - No imports from any module in this project (only stdlib + pydantic).
  - No infrastructure concerns (no DB, no Redis, no HTTP).
  - Changes here affect the entire system — modify with care.
"""
from enum import StrEnum


class JobType(StrEnum):
    TEXT_GENERATION = "text_generation"
    TEXT_EMBEDDING = "text_embedding"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    VIDEO_GENERATION = "video_generation"
    MULTIMODAL_PIPELINE = "multimodal_pipeline"


class JobStatus(StrEnum):
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """Terminal statuses cannot transition to any other status."""
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    def can_transition_to(self, next_status: "JobStatus") -> bool:
        """Enforces valid state machine transitions."""
        allowed: dict["JobStatus", set["JobStatus"]] = {
            JobStatus.QUEUED: {JobStatus.SCHEDULED, JobStatus.CANCELLED},
            JobStatus.SCHEDULED: {JobStatus.RUNNING, JobStatus.CANCELLED},
            JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED},
            JobStatus.COMPLETED: set(),
            JobStatus.FAILED: {JobStatus.QUEUED},  # retry path
            JobStatus.CANCELLED: set(),
        }
        return next_status in allowed.get(self, set())


class JobPriority(StrEnum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

    def to_arq_queue(self) -> str:
        """Maps priority to the corresponding ARQ queue name."""
        return {
            JobPriority.HIGH: "high",
            JobPriority.NORMAL: "normal",
            JobPriority.LOW: "low",
        }[self]


class Modality(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    PIPELINE = "pipeline"


class ArtifactType(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    JSON = "json"
    LOG = "log"


class SSEEventType(StrEnum):
    JOB_CREATED = "job_created"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    STARTED = "started"
    PROGRESS = "progress"
    HEARTBEAT = "heartbeat"
    ARTIFACT_READY = "artifact_ready"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Maps each job type to its modality.
# Used by the scheduler (concurrency limits per modality) and dispatcher (worker routing).
JOB_TYPE_TO_MODALITY: dict[JobType, Modality] = {
    JobType.TEXT_GENERATION: Modality.TEXT,
    JobType.TEXT_EMBEDDING: Modality.TEXT,
    JobType.SPEECH_TO_TEXT: Modality.AUDIO,
    JobType.TEXT_TO_SPEECH: Modality.AUDIO,
    JobType.IMAGE_GENERATION: Modality.IMAGE,
    JobType.IMAGE_EDIT: Modality.IMAGE,
    JobType.VIDEO_GENERATION: Modality.VIDEO,
    JobType.MULTIMODAL_PIPELINE: Modality.PIPELINE,
}
