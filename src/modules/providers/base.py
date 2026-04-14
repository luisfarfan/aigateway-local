"""
Provider port — the hexagonal interface that every AI engine adapter must implement.

This is the most stable contract in the system.
  - Workers depend on this interface, not on concrete providers.
  - Adding a new AI engine = implementing this interface. Zero other changes.
  - Changing this interface affects every adapter — do so with care.

Concrete adapters live in:
  src/modules/providers/diffusers/provider.py
  src/modules/providers/local_tts/provider.py
  src/modules/providers/local_llm/provider.py
  src/modules/providers/stub/provider.py
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import UUID

from src.core.domain import JobType, Modality


# ─── Capability declaration ────────────────────────────────────────────────────

@dataclass
class ProviderCapability:
    """
    Declares what a provider can do and what resources it requires.
    Registered in the ProviderRegistry at startup; queried by the scheduler
    for resource-aware routing decisions.
    """
    provider_id: str
    supported_job_types: list[JobType]
    supported_models: list[str]
    modality: Modality
    max_concurrent_jobs: int = 1
    requires_gpu: bool = False
    estimated_vram_mb: int | None = None   # None = unknown / CPU-only
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Execution context ─────────────────────────────────────────────────────────

# Callback types — providers call these to report progress and emit artifacts.
ProgressCallback = Callable[[float, str | None], Awaitable[None]]
# args: (percent: float, step_description: str | None)

ArtifactCallback = Callable[[str, str, str], Awaitable[None]]
# args: (storage_key: str, artifact_type: str, mime_type: str)


@dataclass
class ExecutionContext:
    """
    Everything a provider needs to execute a job.
    Constructed by the worker and passed into BaseProvider.execute().
    """
    job_id: UUID
    job_type: JobType
    provider_id: str
    model: str | None
    input_payload: dict[str, Any]
    priority: str
    timeout_seconds: int | None
    worker_id: str

    # Callbacks — providers MUST call these during execution.
    # on_progress: report current progress (0.0–100.0) and optional step name
    # on_artifact: register a generated output file
    on_progress: ProgressCallback
    on_artifact: ArtifactCallback


# ─── Provider result ───────────────────────────────────────────────────────────

@dataclass
class ProviderResult:
    """
    What a provider returns after executing a job.
    Workers translate this into DB updates and SSE events.
    """
    success: bool
    result_summary: dict[str, Any] = field(default_factory=dict)
    # Storage keys of output files (registered via on_artifact during execution)
    artifact_keys: list[str] = field(default_factory=list)
    error_message: str | None = None
    error_detail: dict[str, Any] = field(default_factory=dict)
    # Optional execution telemetry: {"duration_s": 12.4, "vram_peak_mb": 4096, ...}
    execution_metadata: dict[str, Any] = field(default_factory=dict)


# ─── Base provider (the Port) ──────────────────────────────────────────────────

class BaseProvider(ABC):
    """
    The hexagonal Port: the interface every AI engine adapter must implement.

    Lifecycle (called by the worker):
      1. initialize()   — called once at worker startup
      2. execute()      — called per job, may be called concurrently up to max_concurrent_jobs
      3. cancel()       — called if the job is cancelled while running
      4. teardown()     — called at worker shutdown

    Contract for execute():
      - MUST call on_progress() at least once (signals the job started)
      - MUST call on_progress(100.0) before returning on success
      - MUST call on_artifact() for each output file produced
      - MUST respect timeout_seconds (use asyncio.wait_for or internal check)
      - MUST return ProviderResult — never raise an unhandled exception
      - SHOULD call on_progress() every few seconds for long jobs (enables heartbeat)
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique string identifier, e.g. 'diffusers', 'local_tts'. Must be stable."""
        ...

    @property
    @abstractmethod
    def capability(self) -> ProviderCapability:
        """Declares what this provider supports. Read by ProviderRegistry at registration."""
        ...

    @abstractmethod
    def supports(self, job_type: JobType, model: str | None = None) -> bool:
        """
        Returns True if this provider can handle the given job_type and model.
        Used by ProviderRegistry.resolve() for routing validation.
        """
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """
        Load models into memory, allocate GPU resources, warm up.
        Called once at worker startup. Must be idempotent (safe to call again).
        """
        ...

    @abstractmethod
    async def execute(self, context: ExecutionContext) -> ProviderResult:
        """
        Execute the job described by context.
        See class docstring for the full contract.
        """
        ...

    @abstractmethod
    async def cancel(self, job_id: UUID) -> bool:
        """
        Attempt to cancel an in-progress job.
        Returns True if the engine supports cancellation and the job was cancelled.
        Returns False if the engine cannot cancel mid-execution (result will be discarded).
        """
        ...

    async def health_check(self) -> dict[str, Any]:
        """
        Returns current health/status information.
        Called by the /health endpoint. Override for richer diagnostics.
        """
        return {
            "provider": self.provider_id,
            "status": "ok",
            "modality": self.capability.modality,
        }

    async def teardown(self) -> None:
        """
        Release resources, unload models from GPU memory.
        Called at worker shutdown. Override if your provider holds GPU/RAM resources.
        """
        pass
