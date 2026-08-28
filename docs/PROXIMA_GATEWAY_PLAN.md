# Proxima Gateway — plan de consolidación

> Evolución de `aigateway-local` al gateway único de IA de Proxima: cloud (CLIProxyAPI)
> + local (GPU), con observabilidad OTel → Langfuse, fallback y SDK compartido.
>
> Estado: **propuesta**. Nada implementado todavía.
> Fecha: 2026-08-26

---

## 1. Por qué

Hoy existen **tres implementaciones independientes** del mismo adaptador a CLIProxyAPI,
y ya divergieron en capacidades:

| Ubicación | LOC aprox | Tiene | Le falta |
|---|---|---|---|
| `proxima-intelligence-v2/src/proxima_intelligence/llm/` | ~1.100 (`guard.py` 21KB) | guard de structured output con reparación, cache Redis con TTL por `search_type`, `model_selector` vía Redis + watchdog 30min, websearch por proveedor, image models por prioridad | `to_strict_json_schema`, stripping de code fences |
| `make-montages/src/make_montages/infrastructure/cliproxyapi_*.py` | 980 | `to_strict_json_schema` (strict mode OpenAI: `additionalProperties:false`, `oneOf`→`anyOf`, drop `discriminator`), `strip_markdown_code_fence`, `truncated_repr`, adapters vision/search/image separados | guard con reparación, cache, watchdog |
| `aigateway-local` | 5.991 total | plano async de jobs, ARQ, `ModalityScheduler` (semáforos VRAM), MinIO, SSE, Prometheus + Grafana, providers locales | cualquier acceso a modelos cloud |

Cada bug de traducción de `web_search` se arregla hasta tres veces. Cada mejora de
structured output vive en un solo repo. Y el costo real de todo esto **no se mide en
ningún lado**: CLIProxyAPI expone `usage-statistics-enabled` con retención de 60s en
memoria (`redis-usage-queue-retention-seconds: 60`) — cero histórico.

## 2. Qué queda cuando esté hecho

```
                    ┌────────────────────── Proxima Gateway ──────────────────────┐
   intel-v2  ──┐    │                                                              │
   make-mont ──┼──► │  /v1/*  (sync, OpenAI-compatible)                            │
   rag-api   ──┤    │     routing + fallback + guard + cache + tracing              │
   otros     ──┘    │        ├──► cliproxy provider ──► CLIProxyAPI ──► Gemini/    │
                    │        │                                          Claude/     │
                    │        │                                          Codex/Grok  │
                    │        └──► local_llm provider ──► Ollama (GPU)              │
                    │                                                              │
                    │  /api/v1/jobs  (async, cola ARQ + SSE)                       │
                    │        └──► diffusers · tts · stt · video_editor · crew      │
                    └──────────────────────────────────────────────────────────────┘
                              │ OTel spans          │ métricas
                              ▼                     ▼
                          Langfuse             Prometheus/Grafana
```

**Dos planos, una sola política.** El plano sync mantiene el contrato que
`proxima-intelligence-v2` ya consume (`POST /v1/chat/completions`) — migrar es cambiar
`cliproxy_base_url`. El plano async es el que ya existe hoy, intacto.

**Lo que sólo se puede hacer unificado:** fallback **cloud → local**. Se agota la cuota de
Gemini, en vez de fallar cae a Ollama en la GPU. Hoy no existe en ningún proyecto.

## 3. Alcance y decisiones

### Alcance de este trabajo

**Sólo `aigateway-local`.** No se modifica ningún otro repositorio. `proxima-intelligence-v2`,
`make-montages`, `proxima-rag-api` y el fork Go de CLIProxyAPI quedan intactos.

La auditoría en `F0-AUDIT-ADAPTERS.md` es lectura de esos repos y sirve de **especificación**:
lo que allí ya funciona se reimplementa aquí, no se importa ni se mueve. Migrar a los
consumidores para que llamen a este gateway es un **paso posterior**, fuera de alcance,
y no empieza hasta que este repo esté completo y verificado por su cuenta.

Consecuencia: la definición de "listo" no puede depender de los tests de otro repo. Este
repo tiene que probar su propia corrección — tests de contrato contra la superficie
`/v1/*`, fixtures grabados de respuestas reales de cada proveedor, y un smoke contra una
instancia real de CLIProxyAPI.

### Decisiones

| Decisión | Elección | Razón |
|---|---|---|
| Repo | evolucionar `aigateway-local` in-place | conserva historia, infra ya corriendo (Postgres/Redis/MinIO/Prom/Grafana), `BaseProvider` ya es el puerto correcto |
| Observabilidad | OTel GenAI semconv → Langfuse self-host | portable; Prometheus se queda para métricas operativas; Phoenix se puede añadir después por OTLP sin recodificar |
| Traducciones por proveedor | **en Python, en este repo** (`translate.py`) | ver abajo — evita depender del fork Go y respeta el alcance |

### El fork Go: por qué no lo usamos

Verificado contra el remoto (`git ls-remote`), hoy:

```
fork/main            bc5e903   ← no contiene ninguno de los dos parches
fork/proxima/main    bc5e903   ← MISMO commit que main
fork/feat/web-search-gemini-claude   c37510bc   (sin mergear)
fork/feat/gemini-images-endpoint     54c0dd24   (sin mergear)
origin/main (upstream)  4b5f1eab  ← el fork está atrasado
```

`git merge-base --is-ancestor` confirma que **ninguno** de los dos parches está en
`proxima/main`. Usarlo como base significaría o mergear en el fork (otro repo, fuera de
alcance) o quedarnos sin las traducciones.

Por eso este repo habla con **CLIProxyAPI vanilla** y hace las traducciones per-provider
en Python, dentro de `src/modules/providers/cliproxy/translate.py`:

- websearch Gemini → `tools: [{"type":"web_search"}]` en `/v1/chat/completions`
- websearch Claude → `tools: [{"type":"web_search_20250305","name":"web_search"}]`
- websearch Codex → `/v1/responses` con `web_search_preview`
- imagen Gemini → `/v1/chat/completions`, extraer el data URI de
  `choices[0].message.images[0].image_url.url`
- imagen Codex → `/v1/images/generations` con `response_format: b64_json`

Beneficio secundario: se acaba el mantenimiento del fork y sus rebases contra upstream.
Si alguna traducción resulta imposible desde Python, se documenta como excepción y ahí
sí se evalúa parchear Go — pero como decisión aparte, no como premisa.

## 4. Contratos

### 4.1 Plano sync — `/v1/*`

Superficie OpenAI-compatible, idéntica a la que CLIProxyAPI expone hoy, para que la
migración sea un cambio de `base_url`:

| Endpoint | Notas |
|---|---|
| `POST /v1/chat/completions` | superficie **por defecto**; texto, websearch, visión |
| `POST /v1/responses` | websearch de Codex (`web_search_preview`) |
| `POST /v1/messages` | escape hatch Claude-nativo; **evitar para tools** |
| `POST /v1/images/generations` | `gemini-3.1-flash-image`, `gpt-image-2` |
| `POST /v1/embeddings` | |
| `GET  /v1/models` | unión de modelos cloud + locales disponibles |

Extensiones propias, todas opcionales y con default seguro:

| Header | Efecto |
|---|---|
| `X-Proxima-Project` | dimensiona costo/uso por proyecto en Langfuse y Prometheus |
| `X-Proxima-Route` | fuerza una cadena de routing concreta (`chat`, `websearch`, `vision`, …) |
| `X-Proxima-No-Cache` | salta el cache para esta llamada |
| `X-Proxima-Bypass` | passthrough puro a CLIProxyAPI, sin routing/guard/cache (escape hatch ante incidente) |

Campo de body opcional `proxima.response_model` (JSON Schema) → activa el guard de
structured output con reparación.

### 4.2 Plano async — `/api/v1/jobs`

Sin cambios. `JobType.TEXT_GENERATION` y `IMAGE_GENERATION` ganan un provider más
(`cliproxy`), así que un job de texto puede resolverse en cloud o local según la política.

### 4.3 Provider nuevo

```
src/modules/providers/cliproxy/
├── provider.py     # implementa BaseProvider
├── config.py       # base_url, api_key, timeouts, modelos por familia
├── translate.py    # websearch/vision/image por proveedor (de intel-v2 + make-montages)
└── errors.py       # 401/403/408/429/5xx → retryable; 4xx restantes → fatal
```

`ProviderCapability(provider_id="cliproxy", requires_gpu=False, max_concurrent_jobs=16,
modality=TEXT|IMAGE)` — no compite por VRAM, así que no pasa por el semáforo de GPU.

## 5. Modelo de datos del tracing

Un span por request en el plano sync, y por `provider.execute()` en el async, con
atributos OTel GenAI semconv más los propios:

```
gen_ai.system            = "cliproxy" | "ollama" | "diffusers" | …
gen_ai.request.model     = modelo pedido
gen_ai.response.model    = modelo que realmente respondió (≠ si hubo fallback)
gen_ai.usage.input_tokens / output_tokens
proxima.project          = X-Proxima-Project
proxima.route            = cadena de routing usada
proxima.attempt          = 0,1,2…      (0 = primer intento)
proxima.fallback_reason  = "http_429" | "timeout" | "schema_invalid" | …
proxima.cache            = "hit" | "miss" | "bypass"
proxima.guard.repairs    = nº de reintentos de reparación del structured output
proxima.cost_usd         = costo calculado
proxima.cost_usd_saved   = costo equivalente evitado por usar OAuth en vez de API de pago
```

Tablas nuevas en Postgres (histórico propio, independiente de Langfuse):
`llm_requests` (una fila por request lógico) y `llm_attempts` (una por intento físico,
FK a la anterior). Langfuse es la UI; Postgres es la fuente para reportes y evals.

Métricas Prometheus nuevas, junto a las `gateway_*` existentes:
`gateway_llm_tokens_total{direction,model,project}`,
`gateway_llm_cost_usd_total{model,project}`,
`gateway_llm_fallback_total{from_model,to_model,reason}`,
`gateway_llm_cache_total{result}`,
`gateway_llm_guard_repairs_total{model}`.

Precios en `config/pricing.yaml`, versionado.

## 6. Routing y fallback

`config/routing.yaml`, recargable en caliente:

```yaml
routes:
  chat:
    - {provider: cliproxy, model: gemini-3-flash}
    - {provider: cliproxy, model: claude-sonnet-4}
    - {provider: local_llm, model: qwen2.5:14b}     # último recurso, GPU local
  websearch:
    - {provider: cliproxy, model: gemini-3-flash, tool: web_search}
    - {provider: cliproxy, model: claude-sonnet-4, tool: web_search_20250305}
fallback_on:
  http: [401, 403, 408, 429, 500, 502, 503, 504]
  transport: [timeout, connect_error]
  semantic: [schema_invalid, empty_response, refusal]
circuit_breaker:
  window_s: 60
  failure_threshold: 5
  open_for_s: 120
```

El `model-watchdog` de intel-v2 (cada 30 min, escribe a Redis `proxima:model:chat`) se
mueve aquí y pasa a alimentar el orden de las cadenas para todos los consumidores, no
sólo intel-v2.

## 7. Guard y cache unificados

`src/modules/structured/` — reimplementado aquí, tomando como especificación lo mejor de
cada implementación auditada (nada se copia desde otro repo; ambos quedan intactos):

- patrón de **intel-v2**: bucle de reparación de 3 intentos, extracción de JSON,
  log persistente por intento, cache antes de la llamada
- patrón de **make-montages**: `to_strict_json_schema` (recursivo, `$defs`/`anyOf`/`items`,
  `oneOf`→`anyOf`), stripping de code fences, truncado de blobs en logs, y sobre todo la
  **inyección del schema completo en la conversación** con los valores reales de los enums
  — Gemini y Claude descartan `response_format`, así que es el único sitio donde el
  contrato les llega (auditoría §3)

`src/modules/cache/`: `derive_cache_key` generalizado — hoy el namespace es
`proxima:enrich:v2:{search_type}:{hash}`, acoplado al dominio de enriquecimiento. Pasa a
`proxima:llm:v1:{project}:{route}:{hash}` con TTL por ruta configurable.

## 8. SDK compartido

`sdk/python/proxima_llm/` — vive en este repo, con superficie **sync y async** (los
consumidores auditados usan una cada uno). Nadie lo consume todavía; adoptarlo es el paso
posterior, fuera de alcance:

```python
from proxima_llm import Gateway

gw = Gateway(project="mi-proyecto")
await gw.chat(messages)
await gw.structured(messages, response_model=ProductSpecs)   # guard incluido
await gw.search(messages)                                    # websearch por proveedor
await gw.vision(messages, images=[...])
await gw.image(prompt, size="1024x1024")
await gw.embed(texts)
```

Instalable por git ref (`proxima-llm @ git+ssh://…#subdirectory=sdk/python`).

Criterio de listo: el SDK pasa **la misma suite de contrato** que la API HTTP, en sync y
en async. Que un consumidor real lo adopte es otra historia, y otra fase.

## 9. Fases

Todas dentro de `aigateway-local`. Cada una deja el repo funcionando y verificable **sin
depender de ningún otro proyecto**.

| # | Qué | Días | Criterio de aceptación |
|---|---|---|---|
| **F0** | ~~Auditoría + fijar CLIProxyAPI vanilla con tag fijo + banco de fixtures~~ **HECHO** — ver `F0-AUDIT-ADAPTERS.md`, `F0-VANILLA-CAPABILITIES.md`, `CLIPROXY-AUTH.md`, `scripts/record_fixtures.py`, `tests/test_cliproxy_fixtures.py` | ✔ | 7 fixtures grabados, 7 tests de contrato pasan sin red |
| **F1** | ~~Provider `cliproxy` + `translate.py` + plano sync `/v1/*`~~ **HECHO** | ✔ | 63 tests sin red + 4 e2e reales; websearch devuelve fuentes; `base_url` tolera `/v1`; provider registrado en API y worker |
| **F2** | ~~Guard de structured output unificado + cache~~ **HECHO** — ver `F2-GUARD-EVIDENCE.md` | ✔ | 86 tests sin red; en vivo: unión discriminada que falla sin guard sale bien al primer intento con él |
| **F3** | ~~OTel + Langfuse + `pricing.yaml` + tablas + métricas~~ **HECHO** — ver `F3-OBSERVABILITY.md` | ✔ | Langfuse propio en el compose; trazas reales clasificadas como GENERATION con tokens |
| **F4** | ~~`routing.yaml`, fallback, circuit breaker, watchdog~~ **HECHO** — ver `F4-ROUTING.md` | ✔ | 119 tests; en vivo: fallback `gpt-image-2 → gemini-3-flash` declarado en la respuesta; watchdog sondeando |
| **F5** | ~~Fallback cloud → local (Ollama)~~ **HECHO** — ver `F5-LOCAL-FALLBACK.md` | ✔ | 122 tests; en vivo: los 3 modelos cloud caídos y la petición servida por `qwen2.5:7b` en la GPU |
| **F6** | ~~SDK sync+async + evals~~ **HECHO** — ver `F6-SDK-Y-EVALS.md` | ✔ | 141 tests; evals 12/12 tras arreglar un HTTP 400 que sólo ellos encontraron |

Total: ~3 semanas. Al terminar F6 el repo está listo y recién entonces se planifica,
como trabajo aparte, migrar a los consumidores.

Cambio respecto a la versión anterior del plan: el guard sube de F4 a **F2**, y la
migración de consumidores sale del plan.

## 10. Riesgos

| Riesgo | Mitigación |
|---|---|
| Reimplementar las traducciones sin poder probar contra los repos originales | banco de fixtures en F0 grabado de respuestas reales; la auditoría documenta la forma exacta de cada payload |
| Alguna traducción no sea posible desde Python | se documenta como excepción y se decide el parche Go aparte; no bloquea el resto |
| Hop extra de latencia (~2–5 ms) | header `X-Proxima-Bypass`; medido como métrica propia |
| Gateway como punto único de fallo | health check; el SDK (F6) puede caer a CLIProxyAPI directo |
| Rate limit compartido entre proyectos | cuotas por `X-Proxima-Project` en el routing |
| Directorio de tokens OAuth | bind-mount host en un path neutral, documentado; no reutilizar el de otro proyecto |
| Deriva futura entre planos sync y async | ambos pasan por el mismo `ProviderRegistry` y el mismo decorador de tracing |
| **Los jobs de imagen cloud se serializan tras la GPU** | ver abajo — decidir antes de que se note en producción |

### El semáforo de modalidad ignora quién ejecuta

`ModalityScheduler.slot()` resuelve el límite con `JOB_TYPE_TO_MODALITY[job_type]`
(`scheduler.py:111`), sin mirar el provider. El semáforo existe para proteger la VRAM,
pero un `IMAGE_GENERATION` servido por `cliproxy` no toca la GPU y aun así consume el
único slot de `queue_image_concurrency=1`: queda en fila detrás de Diffusers, y bloquea
a Diffusers mientras espera una respuesta de red.

`ProviderCapability` ya declara `requires_gpu`, así que el dato está; sólo no se usa en
esa decisión. Opciones, de menos a más invasiva:

1. Que el executor pida el slot sólo si `provider.capability.requires_gpu` — dos líneas,
   pero cambia un camino compartido con los providers locales.
2. Un carril propio para lo remoto (`Modality.TEXT` o una modalidad `REMOTE` con su
   propio límite), elegido por capability y no por tipo de job.
3. Dejarlo: hoy no molesta porque no hay volumen de imagen cloud.

Es una decisión sobre el plano de jobs que ya existía, no algo que traiga F1, así que
queda anotada en vez de resuelta por cuenta propia.

## 11. Decisiones tomadas y descartadas

### Voyage AI queda FUERA del gateway — 2026-08-28

`proxima-api` y `proxima-rag-api` usan Voyage (`voyage-4`, 1024 dims, en pgvector)
para búsqueda semántica. Se evaluó meterlo como backend de la ruta `embeddings`.

**Decisión del usuario: no.** Razón: llamar a Voyage es una petición HTTP simple, su
cliente son ~100 líneas, y el hop no compensa.

Se deja constancia de lo que se acepta al decidirlo así, para no re-discutirlo sin
los mismos datos delante:

- El gasto de Voyage **no se mide**. Es lo único que se paga por token en todo el
  sistema; el resto va por suscripción OAuth con costo marginal cero. La capa de
  costos del gateway seguirá reportando `$0.00` en todo.
- **No hay cache** de la consulta del comprador, que se repite mucho.
- **Voyage caído = búsqueda semántica caída.** No hay fallback a `bge-m3` local.
- Dos consumidores mantienen **dos clientes**, dos claves y dos políticas de
  reintento. Es el mismo patrón que produjo las tres copias divergentes del
  adaptador de CLIProxyAPI (ver `F0-AUDIT-ADAPTERS.md`).

Qué haría cambiar la decisión: un tercer consumidor, una factura de Voyage que
sorprenda, o una caída de Voyage que tumbe la búsqueda en producción.

Lo que sí quedó hecho: `/v1/embeddings` con modelos locales (`bge-m3`), para quien no
tenga ya un índice construido con otro modelo.

## 12. Pendiente de confirmar

- ¿El repo se renombra a `proxima-gateway` o se queda como `aigateway-local`?
- ¿Langfuse self-host en la misma máquina que el gateway, o en otra?
- ¿Hay cuota/presupuesto real que aplicar por proyecto, o el objetivo es sólo visibilidad?
- Para F0: ¿hay una instancia de CLIProxyAPI con sesiones OAuth vivas contra la que grabar
  los fixtures, o hay que autenticar una nueva?

Fuera de alcance, para más adelante: dónde vive `proxima-rag-api` y en qué orden se migran
los consumidores.
