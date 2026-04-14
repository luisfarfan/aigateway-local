# ADR-002: Modular Monolith with Hexagonal Architecture for Providers

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

The system must support multiple AI engines (Diffusers, local TTS, local LLM, future engines)
without coupling the core to any specific engine. New providers must be addable without
modifying existing modules. The codebase must stay navigable as it grows.

## Decision

**Modular Monolith** organized by vertical slices (one directory per domain concept),
with **Hexagonal Architecture (Ports & Adapters)** applied to the provider layer.

```
src/
  core/          ← shared kernel, no module dependencies
  modules/
    jobs/        ← vertical slice: all job domain logic
    queue/       ← vertical slice: scheduling and dispatching
    workers/     ← vertical slice: job execution runtime
    providers/   ← hexagonal layer: port + all adapters
      base.py    ← Port (abstract interface)
      diffusers/ ← Adapter
      local_tts/ ← Adapter
      stub/      ← Adapter (testing/dev)
    events/      ← vertical slice: SSE pub-sub
    artifacts/   ← vertical slice: output storage
    registry/    ← vertical slice: model catalog
    auth/        ← vertical slice: API key auth
  api/           ← app composition only, no business logic
```

**Dependency rule (strictly enforced):**
```
core ← modules ← providers ← workers ← api
```
Modules do NOT import each other directly. Cross-module communication goes through
`core` types or explicit service interfaces.

## Rationale

- **Why not Clean Architecture (horizontal layers)?**
  A change like "add image_edit support to Diffusers" would touch `domain/`, `application/`,
  `infrastructure/`, and `presentation/` — four directories for one feature. High friction.

- **Why not full DDD?**
  DDD shines with complex, evolving business rules. This system's complexity is in
  *integration* (many AI engines) not in *business logic*. Full DDD adds significant
  boilerplate for moderate gain here.

- **Why Hexagonal for providers specifically?**
  The system IS a gateway. Its core job is adapting diverse AI engines to a common interface.
  Hexagonal architecture is literally designed for this pattern: ports define what the system
  needs, adapters implement it per engine. Adding a new engine = adding one adapter file.

- **Why Modular Monolith?**
  Single deployment unit keeps ops simple (one process, one DB, one Redis).
  Module boundaries are enforced by convention, not by network. If needed, modules can be
  extracted to microservices later — the boundaries are already clean.

## Consequences

- Adding a new AI engine: create one adapter implementing `BaseProvider`. Zero changes elsewhere.
- Adding a new modality: add a worker type + provider port extension. Existing code untouched.
- `providers/base.py` is the most stable contract. It must be changed with care.
- Modules must communicate through well-defined interfaces, not direct imports.
- Easy to extract modules to microservices if scaling requires it.
