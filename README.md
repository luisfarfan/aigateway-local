# Local AI Gateway

A multimodal local AI gateway — runs on your Ubuntu machine and accepts inference requests from any HTTP client (MacBook, another server, etc.).

Think of it as a self-hosted OpenRouter: a unified API for text, audio, image, and video generation using locally installed models, with job queuing, real-time progress via SSE, and artifact storage.

---

## What it does

- **Unified API** — one endpoint for text generation (LLMs), TTS, STT, image generation (Diffusers), and video generation
- **Job queue** — requests are enqueued, never executed directly, preventing VRAM/RAM overload
- **Real-time progress** — subscribe to `GET /jobs/{id}/events` and receive SSE events as the model processes
- **Priority lanes** — jobs are queued as `high`, `normal`, or `low` priority
- **Modality concurrency limits** — configurable max concurrent jobs per type (e.g. only 1 image job at a time on GPU)
- **Artifact storage** — outputs saved to MinIO (S3-compatible), served via presigned URLs
- **Extensible** — adding a new AI engine = implementing one interface (`BaseProvider`)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Client (MacBook / any HTTP client)                          │
│    POST /api/v1/jobs          →  create & enqueue job        │
│    GET  /api/v1/jobs/{id}/events  →  SSE real-time progress  │
│    GET  /api/v1/jobs/{id}     →  final result + artifacts    │
└─────────────────────────┬────────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼────────────────────────────────────┐
│  FastAPI (async)                                             │
│    Auth · Rate limiting · Validation · SSE endpoint          │
└──────┬────────────────────────────┬───────────────────────────┘
       │ persist job                │ enqueue to ARQ
┌──────▼──────┐            ┌────────▼────────┐
│ PostgreSQL  │            │  Redis          │
│  jobs       │            │  queue (ARQ)    │
│  artifacts  │            │  pub/sub (SSE)  │
│  events     │            └────────┬────────┘
└─────────────┘                     │ dequeue
                           ┌────────▼────────────────────────┐
                           │  ARQ Worker                     │
                           │    ModalityScheduler (semaphore) │
                           │    ProviderRegistry.resolve()    │
                           │    Provider.execute()            │
                           │    PUBLISH events → Redis        │
                           └────────┬────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
             ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
             │  Diffusers  │ │  Ollama/HF  │ │  XTTS/Piper │
             │  (images,   │ │  (LLMs)     │ │  (TTS)      │
             │   video)    │ └─────────────┘ └─────────────┘
             └─────────────┘
                           │ save artifacts
                    ┌──────▼──────┐
                    │   MinIO     │
                    │  (S3-compat)│
                    └─────────────┘
```

**Pattern:** Modular Monolith + Hexagonal Architecture (ports & adapters for providers).  
**Stack:** FastAPI · SQLModel · asyncpg · PostgreSQL · ARQ · Redis · MinIO · structlog · Prometheus

---

## Project structure

```
local-ai-gateway/
├── src/
│   ├── core/                   # shared kernel (config, db, redis, storage, exceptions)
│   ├── api/                    # FastAPI app factory, lifespan, middleware
│   └── modules/
│       ├── jobs/               # job domain: models, schemas, repository, service, router
│       ├── events/             # SSE: publisher, subscriber, router
│       ├── queue/              # dispatcher (ARQ), scheduler (modality semaphore)
│       ├── workers/            # (reserved for future per-modality worker classes)
│       ├── artifacts/          # artifact schemas
│       ├── auth/               # API key middleware
│       └── providers/          # hexagonal layer
│           ├── base.py         # Port: BaseProvider interface
│           ├── registry.py     # provider catalog
│           ├── stub/           # always-available test/dev adapter
│           ├── diffusers/      # HuggingFace Diffusers (image, video)
│           ├── local_llm/      # Ollama + HF Transformers (text)
│           └── local_tts/      # XTTS / Kokoro / Piper (audio)
├── workers/
│   └── main.py                 # ARQ worker entrypoint
├── migrations/                 # Alembic migrations
├── docs/adr/                   # Architecture Decision Records
├── tests/
├── docker-compose.yml          # PostgreSQL + Redis + MinIO
├── .env.example
├── Makefile
└── pyproject.toml
```

---

## Quick start (development — Mac or Ubuntu without GPU)

### 1. Clone and install

```bash
git clone <repo-url> local-ai-gateway
cd local-ai-gateway

python3.12 -m venv .venv
source .venv/bin/activate

make dev-install
```

### 2. Configure

```bash
make cp-env    # copies .env.example → .env
# No changes needed for local dev — defaults work out of the box
```

### 3. Start infrastructure

```bash
make up
# Starts: PostgreSQL 16 · Redis 7 · MinIO
# MinIO console: http://localhost:9001 (minioadmin / minioadmin123)
```

### 4. Apply database migrations

```bash
make db-upgrade
```

### 5. Start the API and worker

```bash
# Terminal 1
make api
# → http://localhost:8000/api/v1/docs

# Terminal 2
make worker
```

### 6. Test the full flow

```bash
# Create a job (uses the stub provider — no GPU needed)
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "type": "image_generation",
    "provider": "stub",
    "model": "stub-model",
    "input": {
      "prompt": "A cyberpunk motorcycle in the rain",
      "width": 1024,
      "height": 1024,
      "steps": 30
    }
  }'
# → {"id": "abc-123", "status": "queued", ...}

# Subscribe to SSE progress (in another terminal)
curl -N http://localhost:8000/api/v1/jobs/abc-123/events

# Check final result
curl http://localhost:8000/api/v1/jobs/abc-123
```

---

## Ubuntu setup (with real AI engines)

### 1. Clone and install base

```bash
git clone <repo-url> local-ai-gateway
cd local-ai-gateway

python3.12 -m venv .venv
source .venv/bin/activate
make dev-install
```

### 2. Configure for Ubuntu

```bash
make cp-env
nano .env   # or your editor of choice
```

Enable the providers you have installed:

```bash
# In .env:
ENABLE_PROVIDER_DIFFUSERS=true
ENABLE_PROVIDER_LOCAL_LLM=true
ENABLE_PROVIDER_LOCAL_TTS=true

# Point to your already-downloaded models:
DIFFUSERS_MODEL_PATH_STABLE_DIFFUSION_XL=/path/to/your/sdxl
LOCAL_LLM_BACKEND=ollama        # if Ollama is running
LOCAL_TTS_ENGINE=xtts
```

### 3. Diffusers models

The provider reads model paths from env vars. If a path isn't set, it downloads from HuggingFace Hub on first use.

```bash
# Env var convention: DIFFUSERS_MODEL_PATH_{MODEL_ID_UPPERCASED}
DIFFUSERS_MODEL_PATH_STABLE_DIFFUSION_XL=/data/models/sdxl
DIFFUSERS_MODEL_PATH_STABLE_DIFFUSION_V1_5=/data/models/sd15
DIFFUSERS_MODEL_PATH_SDXL_TURBO=/data/models/sdxl-turbo
```

Supported Diffusers models out of the box:

| model_id | Pipeline | Job type |
|---|---|---|
| `stable-diffusion-xl` | StableDiffusionXLPipeline | image_generation |
| `stable-diffusion-v1-5` | StableDiffusionPipeline | image_generation |
| `sdxl-turbo` | AutoPipelineForText2Image | image_generation |
| `stable-diffusion-inpaint` | StableDiffusionInpaintPipeline | image_edit |
| `stable-diffusion-xl-refiner` | StableDiffusionXLImg2ImgPipeline | image_edit |
| `zeroscope-v2` | TextToVideoSDPipeline | video_generation |
| `stable-video-diffusion` | StableVideoDiffusionPipeline | video_generation |

### 4. LLM models (Ollama)

```bash
ollama pull llama3.2
ollama pull mistral
ollama pull qwen2.5
# ...any model from ollama.com/library
```

Supported model IDs (add more in `src/modules/providers/local_llm/config.py`):
`llama3.2`, `llama3.2:3b`, `mistral`, `mixtral`, `qwen2.5`, `deepseek-r1`, `codellama`

### 5. Start everything

```bash
make up          # infra services
make db-upgrade  # migrations

# Terminal 1
make api

# Terminal 2
make worker
```

---

## API reference

Interactive docs: `http://localhost:8000/api/v1/docs`

### Create a job

```
POST /api/v1/jobs
Authorization: Bearer <api-key>
```

```json
{
  "type": "image_generation",
  "priority": "normal",
  "provider": "diffusers",
  "model": "stable-diffusion-xl",
  "input": {
    "prompt": "A cinematic cyberpunk motorcycle in the rain",
    "negative_prompt": "blurry, distorted",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 7.5
  }
}
```

Returns `202 Accepted` with the job object.

### Subscribe to progress (SSE)

```
GET /api/v1/jobs/{job_id}/events
Authorization: Bearer <api-key>
```

Events stream:

```
event: job_created
data: {"event_type":"job_created","job_id":"...","status":"queued",...}

event: started
data: {"event_type":"started","job_id":"...","status":"running","progress_percent":0.0,...}

event: progress
data: {"event_type":"progress","job_id":"...","progress_percent":45.0,"message":"Denoising step 14/30",...}

event: artifact_ready
data: {"event_type":"artifact_ready","job_id":"...","artifact_url":"http://...","metadata":{"artifact_type":"image"},...}

event: completed
data: {"event_type":"completed","job_id":"...","status":"completed","progress_percent":100.0,...}
```

The connection closes automatically on terminal events (`completed`, `failed`, `cancelled`).  
On reconnect, send `Last-Event-ID` to replay missed events.

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/jobs` | List jobs (filterable by status, type, priority, provider) |
| `GET` | `/api/v1/jobs/{id}` | Get job with artifacts |
| `DELETE` | `/api/v1/jobs/{id}` | Cancel a job |
| `GET` | `/api/v1/providers` | List registered providers and capabilities |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks DB, Redis) |
| `GET` | `/metrics` | Prometheus metrics |

### Example payloads

**Text generation (Ollama):**
```json
{
  "type": "text_generation",
  "provider": "local_llm",
  "model": "llama3.2",
  "input": {
    "prompt": "Explain quantum entanglement in simple terms",
    "system_prompt": "You are a helpful assistant.",
    "max_tokens": 1024,
    "temperature": 0.7
  }
}
```

**Text to speech (XTTS):**
```json
{
  "type": "text_to_speech",
  "provider": "local_tts",
  "model": "xtts_v2",
  "input": {
    "text": "Hello, this is a test audio.",
    "voice": "es_male_01",
    "language": "es",
    "output_format": "wav"
  }
}
```

**Multimodal pipeline:**
```json
{
  "type": "multimodal_pipeline",
  "priority": "high",
  "provider": "stub",
  "input": {
    "goal": "Generate a narrated video about a KTM motorcycle on a mountain road",
    "steps": [
      {"step_type": "script_generation", "provider": "local_llm", "model": "llama3.2"},
      {"step_type": "tts", "provider": "local_tts", "model": "xtts_v2"},
      {"step_type": "image_generation", "provider": "diffusers", "model": "stable-diffusion-xl"},
      {"step_type": "video_assembly"}
    ]
  }
}
```

---

## Authentication

Set `API_KEYS` in `.env` as a comma-separated list:

```bash
API_KEYS=your-key-here,optional-second-key
```

Generate a key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Send with requests:
```bash
# Bearer token
curl -H "Authorization: Bearer your-key-here" ...

# Or API key header
curl -H "X-API-Key: your-key-here" ...
```

If `API_KEYS` is empty, auth is disabled (development mode).

---

## Adding a new provider

1. Create `src/modules/providers/{name}/provider.py` implementing `BaseProvider`
2. Add a feature flag to `.env.example`: `ENABLE_PROVIDER_{NAME}=false`
3. Register it in `src/api/lifespan.py` under the flag check
4. Add models to `WorkerSettings.functions` if needed

The rest of the system (API, queue, SSE, storage) requires zero changes.

---

## Development

```bash
make lint       # ruff linter
make format     # auto-format
make typecheck  # mypy
make test       # pytest with coverage

# Generate a new DB migration after changing models
make db-migrate m="add new field to jobs"
make db-upgrade
```

---

## Architecture decisions

See [`docs/adr/`](docs/adr/) for the full rationale behind every major decision:

- [001 — Stack selection](docs/adr/001-stack-selection.md)
- [002 — Modular + Hexagonal architecture](docs/adr/002-modular-hexagonal-pattern.md)
- [003 — ARQ over Celery/RQ](docs/adr/003-arq-over-celery-rq.md)
- [004 — MinIO for storage](docs/adr/004-minio-for-storage.md)
- [005 — SSE with Redis Pub/Sub](docs/adr/005-sse-with-redis-pubsub.md)
