# ADR-005: SSE with Redis Pub/Sub for Real-Time Job Progress

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

Clients need real-time progress updates while jobs execute. The update source (worker process)
and the update sink (HTTP endpoint serving the client) run in separate processes.
Inter-process communication is required.

## Decision

Use **Server-Sent Events (SSE)** as the client protocol, with **Redis Pub/Sub** as the
inter-process event bus between workers and the API process.

## Architecture

```
[Worker Process]                    [API Process]                  [Client]
  ARQ Task
    │
    ├─ updates DB
    │
    └─ PUBLISH ──→ Redis channel ──→ SUBSCRIBE ──→ SSE endpoint ──→ EventSource
                   job:{job_id}        async gen         /jobs/{id}/events
```

**Channel naming:** `job:{job_id}` — one channel per job.
**Global channel:** `client:{client_id}` — for clients subscribing to all their jobs.

## Why SSE over WebSockets

| Criterion | SSE | WebSockets |
|---|---|---|
| Direction | Server → Client (unidirectional) | Bidirectional |
| Reconnection | Built-in (browser EventSource) | Manual |
| HTTP/2 multiplexing | Yes | No |
| Load balancer support | Standard HTTP | Requires sticky sessions or upgrade |
| Complexity | Low | Higher |

For this use case, the client only needs to *receive* events. SSE is the correct tool.
WebSockets would add bidirectional complexity with no benefit.

## Why Redis Pub/Sub over in-process queues

Workers run in separate processes from the API. In-process queues (`asyncio.Queue`) don't
cross process boundaries. Redis Pub/Sub is the natural IPC bus since Redis is already
a dependency (ARQ queue backend).

## SSE Event Envelope

Every event follows this structure:
```
event: {event_type}
data: {"event_type": "...", "job_id": "...", "timestamp": "...", "status": "...", ...}
id: {job_id}:{timestamp_ms}
```

The `id` field enables browser `EventSource` to send `Last-Event-ID` on reconnect,
allowing the API to replay missed events from the `job_events` table.

## Consequences

- Workers call `redis.publish(f"job:{job_id}", event_json)` to emit events.
- The SSE endpoint subscribes to the channel and streams to the client.
- Events are also persisted to `job_events` table for replay on reconnect.
- `sse-starlette` library handles the SSE wire format and keepalive pings.
- On terminal events (completed/failed/cancelled), the channel is closed and cleaned up.
