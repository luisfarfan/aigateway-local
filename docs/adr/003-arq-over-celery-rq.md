# ADR-003: ARQ over Celery / RQ for Job Queue

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

Jobs must be queued and processed asynchronously by background workers. Multiple queue
options exist in the Python ecosystem. The choice affects how workers are structured,
how priorities are handled, and how the system integrates with FastAPI's async model.

## Decision

Use **ARQ (Async Redis Queue)** as the job queue and worker framework.

## Comparison

| Criterion | ARQ | Celery | RQ |
|---|---|---|---|
| Async-native | Yes (asyncio) | No (gevent/threads) | No (sync) |
| Redis as only dependency | Yes | Yes (+ broker/backend split) | Yes |
| Priority queues | Yes (multiple queues) | Yes (complex config) | Yes (limited) |
| Job cancellation | Yes | Yes (revoke) | Limited |
| Job result storage | Yes (Redis TTL) | Yes | Yes |
| Heartbeat / health | Yes | Yes | Limited |
| Ops complexity | Low | High | Low |
| FastAPI integration | Native | Requires workarounds | Requires workarounds |

## Rationale

FastAPI's entire design is async. Mixing sync workers (Celery, RQ) with async FastAPI
code creates impedance: you end up wrapping async code in `asyncio.run()` or using
thread executors, losing the benefits of async entirely.

ARQ tasks are plain `async def` functions. The worker runs an asyncio event loop.
The FastAPI app enqueues jobs with `await arq_pool.enqueue_job(...)`. No context
switching, no thread pools, no gevent monkey-patching.

Priority is handled by using multiple Redis queues (high, normal, low) and configuring
workers to pull from them in order.

## Consequences

- All worker logic must be written as `async def` functions.
- Job results are stored in Redis with configurable TTL (default 24h). Final state is
  also persisted in PostgreSQL for durable querying.
- Worker processes are started independently (`python -m workers.main`).
- ARQ's built-in health check and job introspection are used for the `/health` endpoint.
- Priority queues: `arq:queue:high`, `arq:queue:normal`, `arq:queue:low`.
