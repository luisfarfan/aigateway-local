# aigateway-local

A self-hosted multimodal AI gateway — runs on your Ubuntu machine and accepts inference requests from any HTTP client (MacBook, another server, etc.).

Think of it as a self-hosted OpenRouter: a unified API for LLMs, TTS, STT, image and video generation using locally installed models, with job queuing, real-time progress via SSE, and artifact storage.

---

## What it does

- **Unified API** — one endpoint for text generation (LLMs), TTS, STT, image generation (Diffusers), and video generation
- **Job queue** — requests are enqueued, never executed directly, preventing VRAM/RAM overload
- **Real-time progress** — subscribe to `GET /jobs/{id}/events` and receive SSE events as the model processes
- **Priority lanes** — jobs are queued as `high`, `normal`, or `low` priority
- **Modality concurrency limits** — configurable max concurrent jobs per type (e.g. only 1 image job at a time on GPU)
- **Artifact storage** — outputs saved to MinIO (S3-compatible), served via presigned URLs
- **Autonomous Orquestration** — powered by **CrewAI**, handles multi-step complex requests autonomously
- **Video Assembly** — stitch images and audio with hardcoded subtitles
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

### 2. Configure System Dependencies

```bash
make sys-deps    # Installs ffmpeg and imagemagick (requires sudo)
```

### 3. Configure

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

    ]
  }
}
```

**Autonomous Mission (CrewAI):**
```json
{
  "type": "autonomous_mission",
  "priority": "normal",
  "provider": "orchestrator",
  "input": {
    "prompt": "Genera un video de 3 escenas sobre la historia de Machu Picchu con voz y subtítulos"
  }
}
```
The orchestrator will plan the scenes, generate scripts, images, audio, and finally assemble the video.

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

---

## Gateway de modelos (cloud + local)

Además del plano de jobs, este repo expone un **plano síncrono OpenAI-compatible**
para que cualquier proyecto llame a modelos sin instalar nada: ni CLIProxyAPI, ni
cuentas, ni Ollama, ni Langfuse. Todo eso vive acá una sola vez.

Un mismo endpoint cubre Gemini, Codex y Claude (por CLIProxyAPI, con OAuth) y los
modelos locales de Ollama. Si se agota la cuota cloud, la petición cae sola a la GPU
de esta máquina.

> Las secciones anteriores (en inglés) documentan el plano de jobs original y la
> arquitectura del repo. Esta y las siguientes cubren el gateway de modelos. Son el
> mismo servicio: dos puertas, no dos sistemas.

### Índice

- [Arranque](#arranque)
- [Los dos planos](#los-dos-planos)
- [Plano síncrono `/v1/*`](#plano-síncrono-v1)
  - [Autenticación](#autenticación) · [Chat](#chat) · [Búsqueda web](#búsqueda-web)
  - [Visión](#visión) · [Salida estructurada](#salida-estructurada) · [Imágenes](#imágenes)
  - [Streaming](#streaming) · [Function calling](#function-calling)
  - [Embeddings](#embeddings) · [Modelos locales](#modelos-locales)
  - [Errores](#errores-y-reintentos)
- [Plano de jobs `/api/v1/jobs`](#plano-de-jobs-apiv1jobs)
- [Configuración](#configuración)
  - [Cadenas y fallback](#cadenas-de-modelos-y-fallback) · [Circuit breaker](#circuit-breaker)
  - [Watchdog](#watchdog) · [Precios](#precios) · [Variables](#variables-de-entorno)
- [SDK de Python](#sdk-de-python)
- [Observabilidad](#observabilidad)
- [Comparar modelos](#comparar-modelos-evals)
- [Clientes agénticos](#clientes-agénticos) · [Servicio nativo](#servicio-nativo-que-funcione-siempre)
- [Extender](#extender) · [Resolución de problemas](#resolución-de-problemas) · [Desarrollo](#desarrollo)
- [Qué está verificado y qué no](#qué-está-verificado-y-qué-no)

---

## Arranque

```bash
docker compose up -d          # cliproxy, postgres, redis, minio, clickhouse, langfuse
make serve                    # el gateway en 0.0.0.0:8000
make worker                   # el worker de jobs (sólo si vas a usar el plano async)
```

Eso sirve para trabajar. Para que quede sirviendo **siempre**, incluso tras reiniciar
la máquina, ver [Servicio nativo](#servicio-nativo-que-funcione-siempre).

Antes del primer uso hay que conectar las cuentas cloud una vez —
ver [CLIPROXY-AUTH.md](docs/CLIPROXY-AUTH.md).

Para usarlo desde otros dispositivos del WiFi, ver [RED-LOCAL.md](docs/RED-LOCAL.md).

---

## Los dos planos

| | `/v1/*` (síncrono) | `/api/v1/jobs` (asíncrono) |
|---|---|---|
| Respuesta | inmediata | encolada, con progreso por SSE |
| Para qué | chat, búsqueda, visión, JSON tipado, imagen | lotes, tareas largas, salidas que son archivos |
| Salidas | en el cuerpo de la respuesta | artefactos en MinIO, con URL firmada |
| Concurrencia | 16 simultáneas hacia cloud | semáforo por modalidad, para no reventar la VRAM |

Los dos comparten los mismos proveedores y la misma traducción. No son dos sistemas:
son dos puertas.

---

## Plano síncrono `/v1/*`

```
POST /v1/chat/completions     chat, búsqueda web, visión y salida estructurada
POST /v1/images/generations   generación de imagen
POST /v1/embeddings           vectores para búsqueda semántica y RAG
GET  /v1/models               inventario (cloud + local)
```

### Autenticación

Todas las rutas exigen `Authorization: Bearer <clave>`, con las claves de `API_KEYS`
en el `.env` (varias separadas por coma, para poder revocar una sola).

**Con `API_KEYS` vacío el gateway queda abierto.** Es el modo cómodo para desarrollo
en local, y es exactamente lo que no debe usarse escuchando en la red.

### Chat

```bash
curl -X POST http://192.168.1.12:8000/v1/chat/completions \
  -H "Authorization: Bearer $CLAVE" \
  -H "Content-Type: application/json" \
  -H "X-Proxima-Project: mi-app" \
  -d '{
    "messages": [{"role": "user", "content": "¿Capital de Perú?"}],
    "max_tokens": 100
  }'
```

`model` es **opcional**: sin él manda la cadena de `routing.yaml`. Un consumidor no
debería tener que saber qué modelo está vivo hoy.

`X-Proxima-Project` no es decorativo: separa el costo y las trazas por consumidor.
Sin él, todo el gasto cae en un mismo balde y los reportes no sirven.

| Cabecera | Efecto |
|---|---|
| `X-Proxima-Project` | dimensiona costo y trazas |
| `X-Proxima-No-Cache` | salta el cache en esta llamada |
| `X-Proxima-Client` | etiqueta libre para el histórico |
| `X-Proxima-No-Fallback` | falla en vez de probar el siguiente modelo de la cadena |

### Búsqueda web

Se pide con un bloque de tool, igual que en OpenAI:

```json
{
  "messages": [{"role": "user", "content": "Precio del Bitcoin hoy. Cita la URL."}],
  "tools": [{"type": "web_search"}],
  "max_tokens": 300
}
```

**Acá está buena parte del valor del gateway.** Cada proveedor expone la búsqueda de
una forma distinta e incompatible:

| Proveedor | Cómo se pide de verdad |
|---|---|
| Google | superficie nativa `/v1beta/models/{m}:generateContent` con `tools:[{googleSearch:{}}]` |
| OpenAI | otro endpoint entero, `/v1/responses` con `web_search_preview` |
| Anthropic | `web_search_20250305` + una cabecera beta |

Tú mandas **una sola forma** — cualquiera de las tres, el gateway las acepta todas — y
él traduce a la que corresponda al modelo que resuelva. Medido: el bloque de Gemini
sobre `/v1/chat/completions` lo **descarta** el proxy, y el modelo contesta "no tengo
acceso a información en tiempo real". Por eso hace falta la traducción.

La respuesta trae las fuentes ya extraídas:

```json
{
  "choices": [{"message": {"role": "assistant", "content": "..."}}],
  "proxima": {
    "searched": true,
    "sources": [
      {"uri": "https://binance.com/...", "title": "binance.com"},
      {"uri": "https://coinmarketcap.com/...", "title": "coinmarketcap.com"}
    ]
  }
}
```

`searched` y `sources` son señales **distintas**: un modelo puede buscar y luego citar
en prosa sin emitir la referencia estructurada. Deducir una de la otra da falsos
negativos.

**En la práctica:** cuando la sirve Gemini vienen 3-4 fuentes con `uri` y `title`.
Si Gemini está saturado y la cadena cae a Codex, `searched` sigue en `true` pero
`sources` puede volver **vacío** — Codex a menudo cita dentro del texto sin emitir la
anotación. Si tu caso depende de tener fuentes estructuradas, fuerza
`model: "gemini-3-flash"` en vez de dejar que decida la cadena, y maneja el error si
está caído.

### Visión

Contenido multimodal en la forma de OpenAI. El gateway lo pasa tal cual, así que
funciona con cualquier modelo que soporte imágenes:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "¿De qué color es esta imagen?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}}
    ]
  }],
  "max_tokens": 100
}
```

En Python:

```python
import base64, pathlib
uri = "data:image/png;base64," + base64.b64encode(pathlib.Path("foto.png").read_bytes()).decode()

gw.chat([{"role": "user", "content": [
    {"type": "text", "text": "¿Qué hay en la foto?"},
    {"type": "image_url", "image_url": {"url": uri}},
]}])
```

Verificado con `gemini-3-flash` y `gpt-5.4-mini`. También acepta URLs públicas en vez
de data URIs, según lo que soporte el modelo de arriba.

### Salida estructurada

Se pide con el `response_format` estándar de OpenAI, y el gateway **hace que se
cumpla también en los proveedores que descartan ese campo**, que son la mayoría:

```json
{
  "messages": [{"role": "user", "content": "Extrae: 'iPhone 15 Pro Max de Apple'"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "Producto",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "nombre": {"type": "string"},
          "categoria": {"type": "string", "enum": ["telefono", "laptop", "otro"]}
        },
        "required": ["nombre", "categoria"]
      }
    }
  }
}
```

El objeto **ya validado** viene aparte, para no obligarte a re-parsear el texto:

```json
{
  "choices": [{"message": {"content": "{\"nombre\":\"iPhone 15 Pro Max\",...}"}}],
  "proxima": {"parsed": {"nombre": "iPhone 15 Pro Max", "categoria": "telefono"},
              "repairs": 0, "cache": "miss"}
}
```

Qué hace por dentro, y por qué (ver [F2-GUARD-EVIDENCE.md](docs/F2-GUARD-EVIDENCE.md)):

1. Manda el schema en `response_format` **y** dentro de la conversación, con los
   valores de cada enum escritos. Gemini y Claude no ven el primero; el segundo es el
   único sitio donde les llega el contrato.
2. Si la respuesta no valida, **repara**: cita el error exacto y vuelve a enunciar los
   campos.
3. Si sigue fallando, reintenta **sin la conversación original**: cuando el contexto es
   lo que descarriló al modelo, insistir sobre él lo vuelve a descarrilar.
4. Agotados los tres intentos, **levanta** con el detalle de cada uno. Nunca un objeto
   a medias ni defaults inventados.

Medido: con schema simple `response_format` basta; con unión discriminada y enums de
valores arbitrarios, el modelo escribe prosa donde va la constante — y con el guard
sale bien al primer intento.

Si el schema no se cumple, la respuesta es **422** (el upstream contestó, lo que no
cumple es el contenido) con los intentos detallados. No un 502.

El resultado se **cachea**: la misma petición con el mismo schema no se vuelve a pagar.
La clave incluye el proyecto, así que dos consumidores no comparten entrada.

### Imágenes

```bash
curl -X POST http://192.168.1.12:8000/v1/images/generations \
  -H "Authorization: Bearer $CLAVE" -H "Content-Type: application/json" \
  -d '{"prompt": "un cubo rojo sobre fondo blanco", "size": "1024x1024"}'
```

Respuesta, siempre normalizada a data URI:

```json
{"model": "gemini-3.1-flash-image", "data": [{"url": "data:image/jpeg;base64,/9j/4AAQ..."}]}
```

Por debajo hay dos vías incompatibles y el gateway las esconde: Gemini entrega la
imagen **dentro de un mensaje de chat**, mientras que los modelos de OpenAI usan
`/v1/images/generations` con base64 pelado.

**La imagen tarda mucho más que el chat** — medido, entre 30 s y 148 s para la misma
petición según la carga de arriba. Por eso tiene su propio timeout
(`CLIPROXY_IMAGE_TIMEOUT_S`, 420 s por defecto): con el de chat se cancelaría a mitad,
gastando la cuota sin traer nada.

Para lotes o cuando no quieras esperar, usa el [plano de jobs](#plano-de-jobs-apiv1jobs):
la imagen queda como artefacto en MinIO con URL firmada.

### Streaming

```json
{"messages": [{"role": "user", "content": "..."}], "stream": true}
```

Devuelve SSE estándar (`text/event-stream`) con los chunks **del upstream, sin
reserializar**: cada proveedor mete campos propios (`native_finish_reason`,
`system_fingerprint`) y reconstruir el JSON sólo podría perderlos. La cabecera
`X-Proxima-Served-By` dice qué modelo respondió, porque en streaming el cuerpo no es
nuestro.

Es un camino más estrecho que el normal, por razones que no son pereza:

| | por qué |
|---|---|
| **Fallback sólo hasta el primer byte** | después, cambiar de modelo cosería una respuesta de dos modelos distintos. La cadena se recorre al abrir el stream; a partir de ahí, lo que salga sale |
| **`response_format` no se acepta** → 400 | no se puede validar contra un schema lo que aún no terminó de llegar |
| **Búsqueda web no se acepta** → 501 | usa superficies que no emiten SSE (la nativa de Gemini, `/v1/responses`) |
| **Sin cache** | guardar exigiría acumular la respuesta entera, lo contrario de lo pedido |

Los tokens **sí** se contabilizan: el gateway manda `stream_options.include_usage` y
lee el `usage` del último chunk. Se cuenta incluso si el cliente corta a mitad — quien
abandona el turno igual consumió.

### Function calling

Las herramientas de función viajan **verbatim**, en la forma estándar de OpenAI:

```json
{
  "messages": [{"role": "user", "content": "¿Qué tiempo hace en Lima?"}],
  "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"]}}}],
  "tool_choice": "auto"
}
```

La respuesta trae `tool_calls` y, sobre todo, `finish_reason: "tool_calls"` — que es
el campo por el que un agente decide si ejecutar herramientas o dar el turno por
cerrado:

```json
{"choices": [{"finish_reason": "tool_calls",
  "message": {"role": "assistant", "content": null, "tool_calls": [
    {"id": "call_1", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\"city\":\"Lima\"}"}}]}}]}
```

El round-trip multi-turno funciona: se devuelve el mensaje del asistente más uno con
`role: "tool"` y su `tool_call_id`, y el modelo cierra con la respuesta final.

**Búsqueda web y function calling se distinguen por el contenido de `tools`.** Si
todas las entradas son de búsqueda, va por la superficie nativa del proveedor. Si hay
al menos una función, va por el camino de chat con todo reenviado — la superficie
nativa de búsqueda de Gemini no acepta funciones y las perdería en silencio.

### Embeddings

```bash
curl -X POST http://192.168.1.12:8000/v1/embeddings \
  -H "Authorization: Bearer $CLAVE" -H "Content-Type: application/json" \
  -d '{"input": ["el gato duerme", "the cat sleeps"]}'
```

```json
{"object": "list", "model": "bge-m3",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.031, ...]}]}
```

`input` acepta un texto suelto o una lista. Los vectores vuelven **ordenados por
índice**, en el mismo orden que los textos: la API permite entregarlos desordenados,
y un RAG que asocie el vector equivocado al texto equivocado falla en silencio.

**Los sirve siempre un modelo local.** CLIProxyAPI no expone esta superficie
(comprobado: devuelve vacío), y es además donde más sentido tiene — un reindexado son
decenas de miles de llamadas, y localmente no cuestan cuota ni salen de la máquina.

**El modelo por defecto salió de medir, no de elegir el primero de la lista.**
Similitud coseno entre "el gato duerme en el sofá" y su traducción al inglés, contra
una frase sin relación:

| modelo | par que debe parecerse | par que no | |
|---|---|---|---|
| `bge-m3` (por defecto) | **0.849** | 0.294 | ✅ 1024 dims |
| `qwen3-embedding:0.6b` | 0.796 | 0.266 | ✅ 1024 dims |
| `nomic-embed-text` | 0.355 | **0.601** | ❌ invertido |

`nomic-embed-text` es monolingüe inglés y espera prefijos de tarea
(`search_document:`, `search_query:`). Sin ellos y con español da un resultado que
**parece** funcionar y no funciona — por eso quedó fuera de la cadena.

### Modelos locales

Prefijo `ollama/`:

```json
{"model": "ollama/qwen2.5:7b", "messages": [{"role": "user", "content": "Di PONG"}]}
```

Es un prefijo explícito y no una heurística a propósito: adivinar por el nombre —"si
dice qwen es local"— se rompe el día que un proveedor cloud sirva un Qwen. Ya pasa:
`gpt-oss-120b` lo sirve antigravity y no es de OpenAI.

Ollama no hace búsqueda web ni genera imágenes. Si un modelo local sale como candidato
para eso, el gateway lo salta y prueba el siguiente en vez de fallar.

### Errores y reintentos

Todos los errores traen `retryable`, que **lo decide el gateway**: sólo él sabe si ya
agotó la cadena de modelos o si el problema es transitorio.

```json
{"error": {"message": "...", "type": "upstream_unavailable", "retryable": true}}
```

| HTTP | `type` | Qué pasó | ¿Reintentar? |
|---|---|---|---|
| 400 | `invalid_request` | petición mal formada | no, arregla la petición |
| 401 | — | falta la API key o es inválida | no |
| 422 | `invalid_structured_output` | no cumplió el schema en 3 intentos | no, cambia prompt o schema |
| 501 | `unsupported_capability` | ningún backend puede hacer eso | no |
| 502 | `no_credential` | ninguna cuenta cubre ese modelo | no, conecta la cuenta |
| 503 | `upstream_unavailable` | cuota, 5xx, cooldown | **sí** |
| 504 | `upstream_timeout` | se agotó la espera | **sí** |

Un `no_credential` sale **502 y no 503** a propósito: esperar no lo arregla, y un
cliente que dedujera "5xx → reintento" quedaría reintentando para siempre.

---

## Plano de jobs `/api/v1/jobs`

Para lo que no cabe en una respuesta HTTP: lotes, tareas largas, salidas que son
archivos.

```bash
# 1. crear
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Authorization: Bearer $CLAVE" -H "Content-Type: application/json" \
  -d '{
    "type": "image_generation",
    "provider": "cliproxy",
    "model": "gemini-3.1-flash-image",
    "input": {"prompt": "un cubo rojo sobre fondo blanco"}
  }'
# → 202 {"id": "6b0686cd-...", "status": "queued"}

# 2. seguir el progreso en vivo (SSE)
curl -N -H "Authorization: Bearer $CLAVE" \
  http://localhost:8000/api/v1/jobs/6b0686cd-.../events

# 3. resultado y artefactos
curl -H "Authorization: Bearer $CLAVE" http://localhost:8000/api/v1/jobs/6b0686cd-...
curl -H "Authorization: Bearer $CLAVE" http://localhost:8000/api/v1/jobs/6b0686cd-.../artifacts
```

Los artefactos se sirven con URL firmada de MinIO. El job de arriba deja un
`image/jpeg` de ~440 KB.

### Tipos de job y qué está disponible hoy

| Tipo | Provider | Estado |
|---|---|---|
| `text_generation` | `cliproxy` (cloud) · `local_llm` (Ollama) | ✅ verificado |
| `image_generation` | `cliproxy` | ✅ verificado, con artefacto en MinIO |
| `text_embedding` | `local_llm` (Ollama) | disponible, sin verificar — para embeddings usa mejor `POST /v1/embeddings` |
| `video_assembly` | `video_editor` (moviepy) | disponible, sin verificar |
| `autonomous_mission` | `orchestrator` (CrewAI) | disponible, sin verificar |
| `text_to_speech` | `local_tts` | ⚠️ falta instalar `pip install -e ".[tts]"` |
| `speech_to_text` | `local_stt` | ⚠️ falta instalar `pip install -e ".[stt]"` |
| generación de imagen/video **local** | `diffusers` | ⚠️ falta `pip install -e ".[diffusers]"` (torch) |

### Video

Hay dos cosas distintas y conviene no confundirlas:

- **Montaje de video** (`video_assembly`, con moviepy): **instalado**. Une imágenes y
  audio que ya estén en MinIO, con subtítulos incrustados.
- **Generación de video** (Diffusers): **no instalado** — requiere torch y GPU.

Montaje, a partir de claves de MinIO:

```json
{
  "type": "video_assembly",
  "provider": "video_editor",
  "model": "moviepy-v1",
  "input": {
    "fps": 24,
    "scenes": [
      {"image_key": "uploads/escena1.png", "audio_key": "uploads/voz1.mp3",
       "subtitle": "Primera escena"},
      {"image_key": "uploads/escena2.png", "audio_key": "uploads/voz2.mp3"}
    ]
  }
}
```

La duración de cada escena la marca su audio. Los archivos se suben antes por
`POST /api/v1/uploads`.

### Concurrencia

Un semáforo por modalidad evita reventar la VRAM: por defecto 1 job de imagen a la
vez, 4 de texto (`QUEUE_IMAGE_CONCURRENCY`, `QUEUE_TEXT_CONCURRENCY`, …).

**Limitación conocida:** el semáforo mira el *tipo de job*, no quién lo ejecuta. Un
`image_generation` servido por `cliproxy` no toca la GPU y aun así ocupa el único
carril de imagen. Documentado en [PROXIMA_GATEWAY_PLAN.md](docs/PROXIMA_GATEWAY_PLAN.md#riesgos)
con tres salidas posibles; sin resolver porque afecta un camino compartido con los
providers locales.

---

## Configuración

### Cadenas de modelos y fallback

`config/routing.yaml`. Una cadena por capacidad: se prueba el primero, y si falla por
algo que otro modelo podría resolver, se prueba el siguiente.

```yaml
routes:
  chat:
    - gemini-3-flash
    - gpt-5.4-mini
    - claude-sonnet-4-6
    - ollama/qwen2.5:7b      # último recurso: GPU local, sin cuota ni internet
  websearch:
    - gemini-3-flash         # primero por las FUENTES: devuelve grounding consistente
    - gpt-5.4-mini
  structured:
    - gemini-3-flash
    - gpt-5.4-mini
    - ollama/qwen2.5:7b
  image:
    - gemini-3.1-flash-image
    - gpt-image-2
  embeddings:
    - ollama/bge-m3            # sólo local: CLIProxyAPI no expone embeddings
    - ollama/qwen3-embedding:0.6b
```

Si el cliente nombra un modelo, ese va **primero** y la cadena queda detrás como red.
El fallback **nunca es silencioso**:

```json
"proxima": {"fell_back_from": "gemini-3-flash", "served_by": "ollama/qwen2.5:7b"}
```

Qué justifica saltar al siguiente:

```yaml
fallback_on:
  - upstream_unavailable    # 429, 5xx, cooldown
  - upstream_timeout
  - no_credential           # ninguna cuenta cubre ese modelo
  - invalid_output          # el guard agotó los intentos; otro modelo puede acertar
  - unsupported_capability  # ese backend no puede hacer eso
```

**`invalid_request` no está, a propósito.** Si la petición está mal formada, el
siguiente modelo la rechaza igual: reintentar sólo gasta cuota y latencia para llegar
al mismo 400.

### Circuit breaker

Corta el tráfico hacia un modelo que viene fallando, para no pagar la latencia de
descubrirlo en cada petición.

```yaml
circuit_breaker:
  window_s: 60           # ventana en la que se cuentan los fallos
  failure_threshold: 5   # fallos en la ventana que abren el circuito
  open_for_s: 120        # cuánto se salta el modelo antes de reintentarlo
```

El estado vive en **Redis**, no en memoria: la API y los workers son procesos
distintos, y un breaker por proceso obliga a cada uno a quemarse por su cuenta.

Degrada hacia **dejar pasar**: sin Redis, nada se bloquea. Bloquear por no poder
consultar el estado sería inventarse una caída.

### Watchdog

Comprueba periódicamente qué modelos responden **de verdad** y abre el circuito de los
que no.

```yaml
watchdog:
  skip_routes:
    - image        # la sonda es una llamada de chat: un modelo de sólo imagen la
                   # rechaza aunque esté vivo
```

```bash
WATCHDOG_ENABLED=true
WATCHDOG_INTERVAL_S=900
```

Existe por un hallazgo concreto: **`GET /v1/models` no es prueba de vida**. Una
instancia llegó a anunciar 62 modelos de los cuales 32 devolvían `token revoked` o
`403`. Un selector que filtre por esa lista elige con confianza un modelo muerto.

Cuesta una llamada de pocos tokens por modelo y barrido. Bastante menos que descubrir
un modelo caído en medio del tráfico real, una petición a la vez.

### Precios

`config/pricing.yaml`, en USD por millón de tokens. **Ningún upstream devuelve el
costo**, así que se calcula acá o no se calcula.

```yaml
models:
  gemini-3-flash: {input: 0.30, output: 2.50}
families:
  openai: {input: 1.25, output: 10.00}
  local: {input: 0, output: 0}      # explícito, no omitido: ver abajo
images:
  gpt-image-2: {per_image: 0.04}
billing:
  oauth: {charged: false}    # suscripción: el costo marginal es cero
  api_key: {charged: true}
```

Tres estados, no dos — y la distinción importa:

| | `priced` | `cost_usd` | `cost_equivalent_usd` |
|---|---|---|---|
| Con tarifa, pagado por API | `true` | el costo real | igual |
| Con tarifa, vía suscripción OAuth | `true` | `0.0` | precio de lista |
| **Sin tarifa en la tabla** | `false` | `null` | `null` |

El tercero incrementa `gateway_llm_unpriced_total`. **Un costo desconocido no es un
costo cero**: un panel que suma ceros por modelos sin tarifa muestra un gasto que
nadie paga y esconde la parte que sí. Por eso los modelos locales se listan con cero
explícito en vez de omitirse.

> Los precios del repo son **estimados, sin verificar contra las tarifas oficiales**.
> Sirven para que el mecanismo funcione; revísalos antes de tomar decisiones con esas
> cifras.

### Variables de entorno

Todas en `.env` (no se versiona).

```bash
# Seguridad — sin esto el gateway queda abierto a la red
API_KEYS=pxg-...                       # varias separadas por coma

# CLIProxyAPI
CLIPROXY_BASE_URL=http://localhost:8417
CLIPROXY_API_KEY=...                   # el de docker/cliproxy/config.yaml
CLIPROXY_TIMEOUT_S=120
CLIPROXY_IMAGE_TIMEOUT_S=420           # la imagen tarda mucho más que el chat
CLIPROXY_DEFAULT_MODEL=gemini-3-flash

# Backend local
ENABLE_BACKEND_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT_S=300

# Cache
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL_S=3600
LLM_DEFAULT_PROJECT=default

# Observabilidad
LANGFUSE_OTLP_ENDPOINT=http://localhost:3778/api/public/otel/v1/traces
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LLM_HISTORY_ENABLED=true               # escribir el histórico en Postgres

# Routing
WATCHDOG_ENABLED=true
WATCHDOG_INTERVAL_S=900

# Puertos remapeados: los de siempre están ocupados en esta máquina
POSTGRES_PORT=5442
MINIO_ENDPOINT=http://localhost:9010
```

---

## SDK de Python

```bash
pip install "proxima-llm @ git+ssh://git@github.com/luisfarfan/aigateway-local.git#subdirectory=sdk/python"
```

Dos superficies, porque los proyectos que lo consumen usan una cada uno:

```python
from proxima_llm import Gateway, SyncGateway, ProximaError

gw = SyncGateway("http://192.168.1.12:8000", api_key="pxg-...", project="tienda")

gw.chat("¿Capital de Perú?").text
gw.search("Precio del Bitcoin hoy").sources        # [Source(uri=..., title=...)]
gw.structured("clasifica esto", schema=SCHEMA).parsed
gw.image("un cubo rojo").images[0].url             # data URI
gw.embed(["texto uno", "texto dos"]).vectors      # para RAG; acepta un str suelto
gw.models()

# visión
gw.chat([{"role": "user", "content": [
    {"type": "text", "text": "¿Qué hay acá?"},
    {"type": "image_url", "image_url": {"url": data_uri}},
]}])

# forzar un modelo, o uno local
gw.chat("hola", model="ollama/qwen2.5:7b")
```

Async, la misma API con `await`:

```python
async with Gateway("http://192.168.1.12:8000", api_key="pxg-...", project="tienda") as gw:
    r = await gw.chat("hola")
```

`SyncGateway` es síncrono **de verdad** (`httpx.Client`), no un `asyncio.run`
envolviendo al async: envolver rompe dentro de un bucle ya corriendo.

Errores:

```python
try:
    r = gw.structured("...", schema=SCHEMA)
except ProximaError as e:
    e.kind        # "upstream_unavailable" | "no_credential" | "invalid_structured_output" | ...
    e.retryable   # lo dice el gateway, no se deduce del status
    e.attempts    # qué pasó en cada intento, si falló la salida estructurada
```

Lo que devuelve:

```python
r.text · r.model · r.parsed · r.sources · r.images
# embed() devuelve Embeddings: .vectors · .dimensions · .model · len(e)
r.prompt_tokens · r.completion_tokens · r.total_tokens
r.searched · r.cache · r.fell_back · r.fell_back_from
r.raw                     # la respuesta cruda, por si hace falta algo no expuesto
```

---

## Observabilidad

Tres capas, y ninguna puede tumbar una petición: si Postgres o Langfuse no están, se
registra un aviso y la respuesta sale igual.

### Trazas — Langfuse (`http://localhost:3778`)

Una traza por petición, con prompt, respuesta, tokens, costo, modelo que respondió y
los intentos del guard como eventos.

Instrumentado con **OpenTelemetry y convención GenAI**, no con el SDK de Langfuse: el
mismo span sirve para Phoenix o Tempo cambiando un endpoint. Langfuse los clasifica
como `GENERATION` con modelo y tokens sin configuración adicional.

### Métricas — Prometheus (`/metrics`)

```
gateway_llm_requests_total{project,route,model,outcome}
gateway_llm_tokens_total{project,model,direction}
gateway_llm_cost_usd_total{project,model}              # lo cobrado de verdad
gateway_llm_cost_equivalent_usd_total{project,model}   # el precio de lista
gateway_llm_unpriced_total{model}                      # huecos en pricing.yaml
gateway_llm_fallback_total{route,from_model,to_model,reason}
gateway_llm_cache_total{project,result}
gateway_llm_guard_attempts_total{model,outcome}
gateway_llm_duration_seconds{route,model}
```

### Histórico — Postgres

- `llm_requests` — una fila por petición **lógica**. Unidad de facturación.
- `llm_attempts` — una fila por llamada **física**. Unidad de diagnóstico.

Dos tablas y no una porque con una habría que elegir entre contar el costo dos veces
o perder el detalle de por qué una petición necesitó tres llamadas.

```sql
SELECT project, count(*), sum(cost_equivalent_usd) AS equivalente
FROM llm_requests WHERE created_at > now() - interval '7 days'
GROUP BY project ORDER BY equivalente DESC;
```

---

## Comparar modelos (evals)

Convierte "¿este modelo sirve?" en un número que puedes mirar **antes** de cambiar
`routing.yaml`, y volver a correr después para ver si algo empeoró.

```bash
python evals/run.py --models gemini-3-flash gpt-5.4-mini ollama/qwen2.5:7b
```

```
modelo                         acierto  reparac.   mediana    tokens
gemini-3-flash               4/4  100%        0      1.4s      1116
gpt-5.4-mini                 4/4  100%        0      2.5s      1468
ollama/qwen2.5:7b            4/4  100%        0      3.2s      1191
```

Los casos de `evals/datasets/structured.yaml` no son ejemplos bonitos: cada uno
reproduce un modo de fallo observado. Añadir uno es editar ese YAML.

Ya sirvieron: en su primera corrida encontraron un HTTP 400 de OpenAI que ningún test
unitario veía, porque el schema estricto no llevaba `type` en los nodos `const`.

---

## Clientes agénticos

Un agente con bucle de herramientas (tipo Strix) pide cosas que un chat simple no.
Lo que conviene saber antes de apuntarlo aquí:

**1. Nombra el modelo explícitamente.** La cadena de `chat` termina en
`ollama/qwen2.5:7b` como red de seguridad. Para una respuesta suelta está bien; para
un bucle agéntico es mal candidato — daría muchos turnos de ruido en vez de fallar
rápido. Un modelo nombrado va **primero** en la cadena, y si además quieres que falle
en vez de degradar:

```
X-Proxima-No-Fallback: 1
```

**2. Comprueba quién respondió.** En JSON está en `model`, y si hubo salto,
`proxima.fell_back_from`. En streaming, la cabecera `X-Proxima-Served-By`.

**3. Desde un contenedor, `localhost` no es esta máquina.** Si el agente corre en su
propio Docker, apunta a la IP de la LAN (`http://192.168.1.12:8000`), no a
`localhost` ni a `127.0.0.1`.

**4. Streaming y salida estructurada son excluyentes.** Ver [Streaming](#streaming).

---

## Servicio nativo (que funcione siempre)

Los contenedores llevan `restart: unless-stopped`, así que vuelven solos al reiniciar
la máquina. El gateway y el worker no, porque no están containerizados: si los lanzas
a mano en un terminal, mueren al cerrarlo. Resultado: reinicias y todo vuelve **menos
la puerta de entrada**.

Se arregla con dos servicios de usuario de systemd — con tu cuenta, tu venv y tu
`.env`, sin root:

```bash
mkdir -p ~/.config/systemd/user
cp ops/aigateway.service ops/aigateway-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now aigateway aigateway-worker

# Sin esto systemd los mata al cerrar sesión, y no arrancan hasta que entres.
sudo loginctl enable-linger $USER
```

Comprobar y operar:

```bash
systemctl --user status aigateway
journalctl --user -u aigateway -f          # logs en vivo
systemctl --user restart aigateway         # tras tocar .env o config/*.yaml
systemctl --user disable --now aigateway aigateway-worker
```

`Restart=always` los levanta si se caen. Verificado con un `kill -9`: vuelve solo en
~10 s. `StartLimitIntervalSec=0` evita que systemd se rinda si Postgres tarda en
arrancar tras un reinicio.

**El worker es opcional pero conviene:** sin él, los jobs de `/api/v1/jobs` se encolan
y nadie los ejecuta — se quedan en `queued` para siempre. El plano `/v1/*` funciona
sin worker.

### Dos trampas del `.env` que sólo aparecen bajo systemd

**1. systemd no quita los comentarios de la misma línea.** Bash sí. Con un `.env`
que trae:

```bash
LOG_JSON=false                   # true in production
```

`EnvironmentFile=` le pasaría a la app el string `false                   # true in
production` y pydantic falla al arrancar. Por eso los units **no** usan
`EnvironmentFile`: hacen que bash lea el archivo.

```ini
ExecStart=/bin/bash -c 'set -a; . ./.env; set +a; exec .venv/bin/uvicorn ...'
```

**2. Todo lo que el servicio necesita tiene que estar EN el `.env`.** Lanzándolo a
mano es fácil escribir `POSTGRES_PORT=5442 make serve` y no notar que la variable
falta del archivo. El servicio no tiene esa línea de comandos.

Pasó tres veces al instalarlo, y cada una se veía distinta:

| Falta | Síntoma |
|---|---|
| `POSTGRES_PORT` apuntaba a 5432 | `password authentication failed for user "gateway"` — se conectaba al PostgreSQL del sistema, no al nuestro |
| `MINIO_ENDPOINT` apuntaba a otro MinIO | arranca, pero los artefactos irían al servidor equivocado |
| `CLIPROXY_API_KEY` no estaba | `Illegal header value b'Bearer '` |

Ninguna se veía en las pruebas manuales, porque en todas yo pasaba el valor correcto
por línea de comandos. Si cambias algo del entorno, **pruébalo reiniciando el
servicio**, no lanzándolo a mano.

---

## Extender

### Añadir un modelo a una cadena

Editar `config/routing.yaml` y reiniciar. Nada más: no hay lista de modelos en el
código, a propósito — fijarla obligaría a un despliegue cada vez que un proveedor
renombra algo.

```yaml
routes:
  chat:
    - gemini-3-flash
    - mi-modelo-nuevo      # cloud, servido por CLIProxyAPI
    - ollama/mistral       # local, servido por Ollama
```

Antes de ponerlo primero, **mídelo**:

```bash
python evals/run.py --models gemini-3-flash mi-modelo-nuevo
```

### Añadir un backend nuevo

Un backend es cualquier cosa que sirva modelos: otro proxy, un proveedor con API
propia, un servidor de inferencia. Implementar el protocolo de
`src/modules/backends/base.py` — `chat`, `search`, `image`, `embed`, `family_of`,
`name` — y registrarlo en `BackendRegistry`.

Lo que no pueda hacer se declara con `BackendCapabilityError`, no con una excepción
genérica: así el routing lo trata como candidato inservible y prueba el siguiente, en
vez de dar por muerta la petición.

Hay un test que comprueba las implementaciones **reales** contra el protocolo. No es
decorativo: `CliproxyClient` llegó a producción sin el atributo `name`, y como los
dobles de test sí lo tenían, nadie lo vio hasta probarlo contra la red.

### Añadir un caso a los evals

Editar `evals/datasets/structured.yaml`. Cada caso lleva `prompt`, `schema`, y
`expect` con afirmaciones sobre el contenido — no sólo sobre si valida. Un JSON que
cumple el schema puede traer el dato equivocado, y eso es justo lo que interesa medir.

---

## Resolución de problemas

Fallos reales que salieron construyendo esto, con su síntoma exacto.

### `503 no_candidates: todos los modelos tienen el circuito abierto`

El breaker cerró el paso a toda la cadena. Ver el estado y limpiarlo:

```bash
docker exec agl_redis redis-cli --scan --pattern 'breaker:*'
docker exec agl_redis redis-cli --scan --pattern 'breaker:*' | xargs -r docker exec -i agl_redis redis-cli DEL
```

Causa habitual: **el watchdog sondeando un modelo que no hace chat**. La sonda es una
llamada de chat, y un modelo de sólo imagen o de embeddings la rechaza aunque esté
perfectamente vivo. Si añades una ruta con modelos así, exclúyela:

```yaml
watchdog:
  skip_routes: [image, embeddings]
```

Pasó dos veces: con `gpt-image-2` y con los modelos de embeddings. Los circuitos
abiertos **sobreviven al arreglo** — hay que limpiarlos a mano.

### `504 upstream_timeout` en imagen

La generación de imagen tarda mucho más que el chat: medido, entre 30 s y 148 s para
la misma petición según la carga de arriba. Tiene su propio timeout:

```bash
CLIPROXY_IMAGE_TIMEOUT_S=420
```

Si aun así se corta, súbelo. Cancelar a mitad gasta la cuota sin traer nada.

### `502 no_credential: auth_not_found`

Ninguna cuenta conectada cubre ese modelo. **No se arregla esperando ni reintentando**
— hay que conectar la cuenta ([CLIPROXY-AUTH.md](docs/CLIPROXY-AUTH.md)) o quitar el
modelo de la cadena.

Ojo: `GET /v1/models` lo lista igual. Listar no es prueba de vida.

### Las trazas no llegan a Langfuse

Si el log dice `tracing.exporter_ready` pero Langfuse está vacío, casi seguro otra
librería instaló su propio `TracerProvider` primero — CrewAI lo hace al importarse.
OpenTelemetry **no permite reemplazarlo** y lo rechaza con un warning que nadie mira;
los spans se crean contra un proveedor sin salida y todo parece funcionar.

`configure_tracing` va lo primero en el lifespan y, si ya hay uno instalado, le
**añade** el exportador en vez de intentar reemplazarlo. Si tocas ese orden, esto
vuelve.

### Puertos ocupados al levantar el compose

Tres puertos habituales estaban tomados por otros servicios de la máquina, y se
remapearon. Dentro de la red de compose nada cambia:

| | estándar | acá |
|---|---|---|
| Postgres | 5432 | `127.0.0.1:5442` |
| MinIO API | 9000 | `127.0.0.1:9010` |
| MinIO consola | 9001 | `127.0.0.1:9011` |
| Langfuse | 3000/3777 | `127.0.0.1:3778` |

Procesos que corras en el host necesitan `POSTGRES_PORT=5442` y
`MINIO_ENDPOINT=http://localhost:9010`.

### Un job se queda en `running` para siempre

Mira el semáforo de modalidad: puede estar esperando un carril que nadie libera.

```bash
docker exec agl_redis redis-cli --scan --pattern 'sema:*'
docker exec agl_redis redis-cli DEL sema:modality:image
```

### `401` desde otro dispositivo

Falta la cabecera `Authorization: Bearer <clave de API_KEYS>`. Si `API_KEYS` está
vacío el gateway queda **abierto**, que es peor.

---

## Desarrollo

```bash
pytest tests/ -q                    # 150 tests, sin red, ~9s
ruff check src/ tests/ sdk/ evals/
ruff format src/ tests/ sdk/ evals/

CLIPROXY_API_KEY=... pytest tests/e2e -v      # e2e real; se salta sin la clave
```

Los tests no tocan la red: releen `tests/fixtures/cliproxy/*.json`, grabados contra
una instancia real. Para re-grabarlos tras un cambio de comportamiento de arriba:

```bash
CLIPROXY_BASE_URL=http://127.0.0.1:8417 CLIPROXY_API_KEY=... python scripts/record_fixtures.py
```

El grabador compara lo observado con lo esperado y avisa si un caso cambió de
resultado: un fixture que pasa de `ok` a `error` es una señal, no un re-grabado
silencioso.

---

## Qué está verificado y qué no

Probado end-to-end contra el gateway expuesto en la red, con modelos reales:

| | |
|---|---|
| chat, búsqueda con fuentes, visión, salida estructurada | ✅ |
| streaming SSE, con tokens contabilizados | ✅ |
| function calling multi-turno (`tool_calls` → `role: tool` → respuesta) | ✅ |
| imagen por `/v1/*` y por jobs (con artefacto en MinIO) | ✅ |
| embeddings (`bge-m3`, 1024 dims) | ✅ |
| modelos locales de Ollama, y fallback cloud → local | ✅ |
| autenticación (401 sin clave, 200 con clave, desde la LAN) | ✅ |
| cache, fallback declarado en la respuesta, watchdog | ✅ |
| histórico en Postgres, métricas, trazas en Langfuse | ✅ |
| SDK sync y async | ✅ |

Sin verificar por mí: `video_assembly`, `autonomous_mission` y `text_embedding` —
están instalados y registrados, pero no los he ejercitado.

No instalados: `diffusers`/torch (imagen y video locales), `TTS`, `faster-whisper`.

Sin cuenta conectada: Anthropic nativo (los Claude que se ven llegan por antigravity,
vía Vertex) y `gpt-image-2` (la cuenta Codex es de plan gratuito).

### Documentación

| | |
|---|---|
| [PROXIMA_GATEWAY_PLAN.md](docs/PROXIMA_GATEWAY_PLAN.md) | el plan y su estado |
| [RED-LOCAL.md](docs/RED-LOCAL.md) | usarlo desde cualquier dispositivo del WiFi |
| [CLIPROXY-AUTH.md](docs/CLIPROXY-AUTH.md) | conectar cuentas cloud |
| [F0-AUDIT-ADAPTERS.md](docs/F0-AUDIT-ADAPTERS.md) | auditoría que sirvió de especificación |
| [F0-VANILLA-CAPABILITIES.md](docs/F0-VANILLA-CAPABILITIES.md) | qué hace CLIProxyAPI vanilla, medido |
| [F2-GUARD-EVIDENCE.md](docs/F2-GUARD-EVIDENCE.md) | por qué hace falta el guard |
| [F3-OBSERVABILITY.md](docs/F3-OBSERVABILITY.md) | costo, métricas, trazas |
| [F4-ROUTING.md](docs/F4-ROUTING.md) | fallback, breaker, watchdog |
| [F5-LOCAL-FALLBACK.md](docs/F5-LOCAL-FALLBACK.md) | cloud → local |
| [F6-SDK-Y-EVALS.md](docs/F6-SDK-Y-EVALS.md) | el cliente y cómo comparar modelos |
