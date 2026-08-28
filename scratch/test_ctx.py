
import uuid
from src.modules.providers.base import ExecutionContext
from src.core.domain import JobType

try:
    ctx = ExecutionContext(
        job_id=uuid.uuid4(),
        job_type=JobType.TEXT_GENERATION,
        provider_id="test",
        model=None,
        input_payload={},
        priority="normal",
        timeout_seconds=30,
        worker_id="worker-1",
        registry=None,
        on_progress=None,
        on_artifact=None
    )
    print("Success instantiating ExecutionContext with keywords")
except Exception as e:
    print(f"Error with keywords: {e}")

try:
    ctx = ExecutionContext(
        uuid.uuid4(),
        JobType.TEXT_GENERATION,
        "test",
        None,
        {},
        "normal",
        30,
        "worker-1",
        None,
        None,
        None
    )
    print("Success instantiating ExecutionContext with positionals")
except Exception as e:
    print(f"Error with positionals: {e}")
