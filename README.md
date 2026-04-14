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
│   ├── core/                   # shared kernel (config, db, redis, storage, metrics, exceptions)
│   ├── api/                    # FastAPI app factory, lifespan, middleware
│   └── modules/
│       ├── jobs/               # job domain: models, schemas, repository, service, router
│       ├── events/             # SSE: publisher, subscriber, router
│       ├── queue/              # dispatcher (ARQ), scheduler (modality semaphore)
│       ├── artifacts/          # artifact CRUD + presigned URL refresh
│       ├── uploads/            # multipart upload + presigned PUT URL
│       ├── status/             # GET /status — providers, job counts, GPU info
│       ├── auth/               # API key middleware
│       └── providers/          # hexagonal layer
│           ├── base.py         # Port: BaseProvider interface
│           ├── registry.py     # provider catalog
│           ├── stub/           # always-available test/dev adapter
│           ├── diffusers/      # HuggingFace Diffusers (image, video)
│           ├── local_llm/      # Ollama + HF Transformers (text)
│           ├── local_tts/      # XTTS / Kokoro / Piper (TTS)
│           └── local_stt/      # faster-whisper / openai-whisper (STT)
├── workers/
│   └── main.py                 # ARQ worker entrypoint
├── migrations/                 # Alembic migrations
├── grafana/
│   ├── provisioning/           # auto-provisioned datasource + dashboard loader
│   └── dashboards/             # local_ai_gateway.json — pre-built Grafana dashboard
├── prometheus.yml              # scrape config (targets host API at :8000/metrics)
├── docs/adr/                   # Architecture Decision Records
├── tests/
├── docker-compose.yml          # Postgres + Redis + MinIO + Prometheus + Grafana
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
# Starts: PostgreSQL 16 · Redis 7 · MinIO · Prometheus · Grafana
# MinIO console:  http://localhost:9001  (minioadmin / minioadmin123)
# Grafana:        http://localhost:3000  (admin / admin)
# Prometheus:     http://localhost:9090
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

### 5. Speech-to-text (Whisper)

```bash
pip install faster-whisper   # recommended — 4× faster than openai-whisper
# or: pip install openai-whisper  (fallback)
```

In `.env`:
```bash
ENABLE_PROVIDER_LOCAL_STT=true
STT_MODEL_SIZE=base          # tiny | base | small | medium | large-v3
STT_DEVICE=auto              # auto | cuda | cpu
STT_COMPUTE_TYPE=float16     # float16 (GPU) | int8 (CPU)
# STT_MODEL_PATH=/data/models/whisper   # optional local path
```

### 6. Start everything

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

### Upload a file (input for STT / img2img)

```bash
# Direct multipart upload (max 500 MB)
curl -X POST http://localhost:8000/api/v1/uploads \
  -H "Authorization: Bearer your-key" \
  -F "file=@/path/to/audio.wav"
# → {"storage_key": "uploads/client-id/abc123/audio.wav"}

# Or get a presigned PUT URL and upload directly to MinIO
curl -X POST http://localhost:8000/api/v1/uploads/presigned \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"filename": "audio.wav", "content_type": "audio/wav"}'
# → {"upload_url": "http://...", "storage_key": "..."}
```

Use the `storage_key` as input when creating a job (e.g. STT transcription).

### Download artifacts

```bash
# Metadata + fresh presigned URL
GET /api/v1/artifacts/{id}

# 302 redirect → direct download (browser / curl friendly)
GET /api/v1/artifacts/{id}/download

# All artifacts for a job
GET /api/v1/jobs/{id}/artifacts
```

### System status

```bash
curl http://localhost:8000/api/v1/status
# → { "providers": [...], "jobs": {"queued": 2, "running": 1, ...}, "gpu": {...} }
```

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/jobs` | List jobs (filterable by status, type, priority, provider) |
| `GET` | `/api/v1/jobs/{id}` | Get job with artifacts |
| `DELETE` | `/api/v1/jobs/{id}` | Cancel a job |
| `GET` | `/api/v1/providers` | List registered providers and capabilities |
| `GET` | `/api/v1/status` | Providers, job counts, GPU/VRAM info |
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

## Observability

Prometheus + Grafana are included in `docker-compose.yml` — no extra setup needed.

```bash
make up   # starts everything including Prometheus and Grafana
```

| Service | URL | Credentials |
|---|---|---|
| **Grafana dashboard** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | — |
| **Metrics endpoint** | http://localhost:8000/metrics | — |

Grafana opens directly on the pre-built **Local AI Gateway** dashboard with:

- **Jobs completed / Active / Queue depth / Success rate** — live stat panels
- **Job throughput** — rate per second by status (completed / failed / retried)
- **Active jobs by provider** — see when your GPU is busy
- **Inference duration p50 / p95 / p99** — latency percentiles per provider
- **Queue depth by priority** — high / normal / low lanes
- **Failed jobs over time** — spikes = provider problems

Custom metrics exposed (all prefixed `gateway_`):

| Metric | Type | Labels |
|---|---|---|
| `gateway_jobs_total` | Counter | `job_type`, `status`, `provider` |
| `gateway_active_jobs` | Gauge | `provider`, `job_type` |
| `gateway_queue_depth` | Gauge | `priority` |
| `gateway_inference_duration_seconds` | Histogram | `provider`, `job_type` |

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
