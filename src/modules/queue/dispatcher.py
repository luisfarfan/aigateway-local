"""
Job dispatcher — enqueues jobs into the ARQ priority queues.

The dispatcher is the only place that knows about ARQ internals.
The rest of the system (service, workers) uses the dispatcher and doesn't
interact with ARQ directly.

Priority → ARQ queue name mapping:
  high   → "high"
  normal → "normal"  (default)
  low    → "low"

Workers listen to all three queues in priority order.
"""
from uuid import UUID

import structlog
from arq.connections import ArqRedis

from src.core.domain import JobPriority

log = structlog.get_logger(__name__)

# ARQ task name — must match the function name in workers/main.py
EXECUTE_JOB_TASK = "execute_job"


async def enqueue_job(
    arq: ArqRedis,
    job_id: UUID,
    priority: JobPriority,
) -> str:
    """
    Enqueues a job into the appropriate ARQ priority queue.

    Returns the ARQ job ID (different from our domain job_id).
    The domain job_id is passed as the task argument so workers can look it up.
    """
    queue_name = priority.to_arq_queue()

    arq_job = await arq.enqueue_job(
        EXECUTE_JOB_TASK,
        str(job_id),           # workers receive the domain job_id as a string
        _queue_name=queue_name,
    )

    log.info(
        "job_enqueued",
        job_id=str(job_id),
        priority=priority,
        queue=queue_name,
        arq_job_id=arq_job.job_id if arq_job else None,
    )

    return arq_job.job_id if arq_job else ""


async def enqueue_pipeline_step(
    arq: ArqRedis,
    parent_job_id: UUID,
    step_job_id: UUID,
    priority: JobPriority,
) -> str:
    """
    Enqueues a pipeline subjob. Same mechanism as a regular job — the worker
    handles the parent/child relationship via the job record.
    """
    return await enqueue_job(arq, step_job_id, priority)
