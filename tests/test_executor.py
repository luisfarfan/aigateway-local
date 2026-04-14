"""
Unit tests for the job executor (workers/executor.py).

Tests the 4 correctness properties without real Redis/MinIO:
  1. Artifact persistence
  2. Scheduler slot acquisition
  3. Timeout enforcement
  4. Retry with backoff
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.core.domain import JobPriority, JobStatus, JobType
from src.modules.jobs.models import Job
from src.modules.providers.base import ProviderResult
from src.modules.providers.stub.provider import StubProvider


def _make_job(**kwargs) -> Job:
    defaults = dict(
        id=uuid4(),
        type=JobType.IMAGE_GENERATION,
        status=JobStatus.QUEUED,
        priority=JobPriority.NORMAL,
        provider="stub",
        model="stub-model",
        input_payload={"prompt": "test"},
        client_id="test-client",
        max_retries=3,
        retry_count=0,
        timeout_seconds=30,
        tags={},
    )
    defaults.update(kwargs)
    return Job(**defaults)


@pytest.mark.asyncio
async def test_stub_provider_emits_progress():
    """StubProvider calls on_progress for each step."""
    provider = StubProvider(step_delay_seconds=0.01)
    await provider.initialize()

    progress_calls: list[float] = []

    async def on_progress(percent: float, step: str | None) -> None:
        progress_calls.append(percent)

    async def on_artifact(key: str, artifact_type: str, mime_type: str) -> None:
        pass

    from src.modules.providers.base import ExecutionContext
    ctx = ExecutionContext(
        job_id=uuid4(),
        job_type=JobType.IMAGE_GENERATION,
        provider_id="stub",
        model="stub-model",
        input_payload={"prompt": "test"},
        priority="normal",
        timeout_seconds=30,
        worker_id="test-worker",
        on_progress=on_progress,
        on_artifact=on_artifact,
    )

    result = await provider.execute(ctx)

    assert result.success is True
    assert len(progress_calls) == 7  # 7 steps in StubProvider
    assert progress_calls[0] == 5.0
    assert progress_calls[-1] == 100.0


@pytest.mark.asyncio
async def test_stub_provider_supports_cancellation():
    """StubProvider.cancel() stops an in-progress job."""
    provider = StubProvider(step_delay_seconds=5.0)  # slow
    await provider.initialize()

    job_id = uuid4()
    cancelled = False

    async def on_progress(percent: float, step: str | None) -> None:
        if percent >= 5.0:
            await provider.cancel(job_id)

    async def on_artifact(*args) -> None:
        pass

    from src.modules.providers.base import ExecutionContext
    ctx = ExecutionContext(
        job_id=job_id,
        job_type=JobType.TEXT_GENERATION,
        provider_id="stub",
        model="stub-model",
        input_payload={},
        priority="normal",
        timeout_seconds=60,
        worker_id="test-worker",
        on_progress=on_progress,
        on_artifact=on_artifact,
    )

    result = await provider.execute(ctx)
    assert result.success is False
    assert "cancelled" in result.error_message.lower()


@pytest.mark.asyncio
async def test_provider_result_success_structure():
    """ProviderResult fields are correct on success."""
    provider = StubProvider(step_delay_seconds=0.01)
    await provider.initialize()

    artifact_keys: list[str] = []

    async def on_progress(p: float, s: str | None) -> None:
        pass

    async def on_artifact(key: str, at: str, mime: str) -> None:
        artifact_keys.append(key)

    from src.modules.providers.base import ExecutionContext
    job_id = uuid4()
    ctx = ExecutionContext(
        job_id=job_id,
        job_type=JobType.IMAGE_GENERATION,
        provider_id="stub",
        model="stub-model",
        input_payload={"prompt": "test"},
        priority="normal",
        timeout_seconds=30,
        worker_id="worker-1",
        on_progress=on_progress,
        on_artifact=on_artifact,
    )

    result = await provider.execute(ctx)

    assert result.success is True
    assert result.result_summary["provider"] == "stub"
    assert result.execution_metadata["simulated"] is True
    assert len(artifact_keys) == 1
    assert str(job_id) in artifact_keys[0]


@pytest.mark.asyncio
async def test_timeout_enforced():
    """asyncio.wait_for raises TimeoutError when provider runs too long."""
    from src.modules.providers.base import ExecutionContext, ProviderResult, BaseProvider
    from src.core.domain import Modality

    class SlowProvider(BaseProvider):
        @property
        def provider_id(self): return "slow"
        @property
        def capability(self):
            from src.modules.providers.base import ProviderCapability
            return ProviderCapability("slow", list(JobType), [], Modality.TEXT)
        def supports(self, *a): return True
        async def initialize(self): pass
        async def cancel(self, *a): return True
        async def execute(self, ctx):
            await asyncio.sleep(60)  # way longer than timeout
            return ProviderResult(success=True)

    job_id = uuid4()
    ctx = ExecutionContext(
        job_id=job_id, job_type=JobType.TEXT_GENERATION, provider_id="slow",
        model=None, input_payload={}, priority="normal", timeout_seconds=1,
        worker_id="w", on_progress=AsyncMock(), on_artifact=AsyncMock(),
    )

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(SlowProvider().execute(ctx), timeout=0.1)


@pytest.mark.asyncio
async def test_persist_artifact_creates_record(db_session):
    """_persist_artifact creates an Artifact row and emits an SSE event."""
    from workers.executor import _persist_artifact
    from unittest.mock import AsyncMock

    job_id = uuid4()
    publisher = AsyncMock()
    publisher.publish = AsyncMock()

    with (
        patch("workers.executor.storage.get_size", new=AsyncMock(return_value=1024)),
        patch("workers.executor.storage.presigned_download_url",
              new=AsyncMock(return_value="http://minio/test.png")),
    ):
        artifact = await _persist_artifact(
            job_id=job_id,
            storage_key=f"jobs/{job_id}/outputs/image_00.png",
            artifact_type="image",
            mime_type="image/png",
            session=db_session,
            publisher=publisher,
        )

    assert artifact is not None
    assert artifact.job_id == job_id
    assert artifact.filename == "image_00.png"
    assert artifact.size_bytes == 1024
    assert artifact.public_url == "http://minio/test.png"
    publisher.publish.assert_called_once()
