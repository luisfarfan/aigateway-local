"""
Stub provider — reference implementation of BaseProvider.

Simulates execution of any job type with configurable delays.
Used for:
  - local development without GPU
  - integration tests
  - demonstrating the full E2E flow (POST job → SSE progress → completed)
  - validating new worker or scheduler code

This is a complete, working implementation — not pseudocode.
"""
import asyncio
from typing import Any
from uuid import UUID

from src.core.domain import JobType, Modality
from src.modules.providers.base import (
    BaseProvider,
    ExecutionContext,
    ProviderCapability,
    ProviderResult,
)

# Steps the stub simulates, with (progress_percent, step_description)
_STUB_STEPS: list[tuple[float, str]] = [
    (5.0, "Initializing"),
    (20.0, "Loading resources"),
    (40.0, "Processing input"),
    (60.0, "Generating output"),
    (80.0, "Post-processing"),
    (95.0, "Saving artifacts"),
    (100.0, "Done"),
]


class StubProvider(BaseProvider):
    """
    Adapter: fakes execution of any job type.
    Emits realistic progress events with configurable per-step delays.
    Supports cancellation mid-execution.
    """

    def __init__(self, step_delay_seconds: float = 1.5) -> None:
        self._step_delay = step_delay_seconds
        # Tracks active job IDs — removing one signals cancellation to the running task
        self._active_jobs: set[UUID] = set()

    @property
    def provider_id(self) -> str:
        return "stub"

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id="stub",
            supported_job_types=list(JobType),  # supports everything
            supported_models=["stub-model", "stub-fast"],
            modality=Modality.TEXT,              # stub lives in the text worker
            max_concurrent_jobs=10,
            requires_gpu=False,
            estimated_vram_mb=None,
        )

    def supports(self, job_type: JobType, model: str | None = None) -> bool:
        return True  # stub handles any request

    async def initialize(self) -> None:
        pass  # nothing to load

    async def execute(self, context: ExecutionContext) -> ProviderResult:
        self._active_jobs.add(context.job_id)
        try:
            return await self._run_steps(context)
        finally:
            self._active_jobs.discard(context.job_id)

    async def _run_steps(self, context: ExecutionContext) -> ProviderResult:
        for percent, step_name in _STUB_STEPS:
            # Check cancellation before each step
            if context.job_id not in self._active_jobs:
                return ProviderResult(
                    success=False,
                    error_message="Job was cancelled during execution",
                )

            await context.on_progress(percent, step_name)

            # Simulate work — last step has no delay (we're done)
            if percent < 100.0:
                await asyncio.sleep(self._step_delay)

        # Emit a fake artifact (a text file for any job type)
        fake_key = f"jobs/{context.job_id}/outputs/result.txt"
        await context.on_artifact(fake_key, "text", "text/plain")

        return ProviderResult(
            success=True,
            result_summary={
                "provider": "stub",
                "job_type": context.job_type,
                "model": context.model or "stub-model",
                "message": "Stub execution completed successfully",
                "input_keys": list(context.input_payload.keys()),
            },
            artifact_keys=[fake_key],
            execution_metadata={
                "simulated": True,
                "step_delay_s": self._step_delay,
                "steps": len(_STUB_STEPS),
            },
        )

    async def cancel(self, job_id: UUID) -> bool:
        if job_id in self._active_jobs:
            self._active_jobs.discard(job_id)
            return True
        return False

    async def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "status": "ok",
            "active_jobs": len(self._active_jobs),
            "step_delay_s": self._step_delay,
        }
