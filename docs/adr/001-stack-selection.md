# ADR-001: Stack Selection

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

Building a local AI gateway on Ubuntu serving multiple AI modalities (text, audio, image, video).
The system needs to be async-first, strongly typed, persist state, queue jobs, and stream
progress events to remote clients over HTTP.

## Decision

| Layer | Choice | Rejected alternatives |
|---|---|---|
| Runtime | Python 3.12 | — |
| API framework | FastAPI | Flask, Django REST |
| ORM / models | SQLModel | SQLAlchemy + Pydantic separately |
| DB driver | asyncpg | psycopg2 (sync) |
| Database | PostgreSQL 16 | SQLite (not prod-ready), MySQL |
| Migrations | Alembic | — |
| Queue / workers | ARQ | Celery (sync-biased), RQ (sync), Dramatiq |
| Pub-Sub / cache | Redis | RabbitMQ (heavier ops) |
| Object storage | MinIO | Raw filesystem (not abstracted) |
| SSE | sse-starlette | Polling (not real-time), WebSockets (overkill) |
| Logging | structlog | stdlib logging (not structured) |
| Metrics | prometheus-fastapi-instrumentator | Custom (reinventing the wheel) |
| Config | Pydantic Settings | python-decouple, dynaconf |
| Packaging | pyproject.toml | setup.py, requirements.txt |

## Rationale

- **SQLModel over SQLAlchemy + Pydantic separately:** Eliminates the dual-model problem
  (one ORM model + one schema per entity). Single source of truth. Same author as FastAPI.

- **ARQ over Celery:** ARQ is fully async-native (asyncio). Celery is fundamentally sync
  and requires extra setup (gevent/eventlet) to work with async code. ARQ integrates
  naturally with FastAPI's event loop. Simpler ops: just Redis, no broker + backend split.

- **MinIO over raw filesystem:** Zero code change to migrate to S3 or Cloudflare R2.
  S3-compatible API from day one. Handles presigned URLs, multipart, etc. Runs locally
  in Docker with no cloud dependency.

- **Redis for both queue and pub-sub:** ARQ uses Redis as its queue backend. SSE uses
  Redis pub-sub channels (one channel per job_id). Single service, two roles. Avoids
  adding a message broker.

## Consequences

- Full async stack: HTTP → DB → queue → storage — no sync blocking points.
- Strong typing via Pydantic v2 throughout.
- Local storage abstracted behind S3-compatible interface from day one.
- Easy migration to managed services (RDS, ElastiCache, S3) without code changes.
- ARQ requires Python async functions as tasks — all worker logic must be async.
