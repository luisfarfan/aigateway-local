"""
Prometheus custom metrics — all prefixed with "gateway_".

Import this module anywhere; prometheus_client uses a global registry
so metrics are singletons — safe to import in both API and worker processes.

Metrics exposed at GET /metrics (via prometheus-fastapi-instrumentator):
  gateway_jobs_total                  Counter   jobs by job_type + status + provider
  gateway_active_jobs                 Gauge     jobs currently executing by provider + job_type
  gateway_queue_depth                 Gauge     jobs in QUEUED state by priority
  gateway_inference_duration_seconds  Histogram wall-clock time inside provider.execute()
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Counters ──────────────────────────────────────────────────────────────────

jobs_total = Counter(
    "gateway_jobs_total",
    "Total jobs that reached a terminal state, labelled by job_type, status and provider.",
    ["job_type", "status", "provider"],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

active_jobs = Gauge(
    "gateway_active_jobs",
    "Number of jobs currently being executed by a provider.",
    ["provider", "job_type"],
)

queue_depth = Gauge(
    "gateway_queue_depth",
    "Approximate number of jobs waiting in QUEUED state by ARQ priority queue.",
    ["priority"],
)

# ── Histograms ────────────────────────────────────────────────────────────────

inference_duration_seconds = Histogram(
    "gateway_inference_duration_seconds",
    "Wall-clock time from RUNNING start to provider.execute() returning.",
    ["provider", "job_type"],
    # Buckets cover: 1s (stub/fast) → 1h (video generation)
    buckets=[1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600],
)


# ─── LLM (F3) ─────────────────────────────────────────────────────────────────
# Etiquetadas por `project` a propósito: la pregunta que estas métricas existen
# para responder es "cuánto gasta cada consumidor", no "cuánto gasta el gateway".

llm_requests_total = Counter(
    "gateway_llm_requests_total",
    "Peticiones lógicas a un modelo, por resultado.",
    ["project", "route", "model", "outcome"],
)

llm_tokens_total = Counter(
    "gateway_llm_tokens_total",
    "Tokens consumidos, por dirección.",
    ["project", "model", "direction"],
)

llm_cost_usd_total = Counter(
    "gateway_llm_cost_usd_total",
    "Costo realmente cobrado. Cero mientras se use OAuth contra suscripción.",
    ["project", "model"],
)

llm_cost_equivalent_usd_total = Counter(
    "gateway_llm_cost_equivalent_usd_total",
    "Precio de lista de lo consumido, se pague o no. Dimensiona el ahorro de "
    "las suscripciones y permite comparar modelos entre sí.",
    ["project", "model"],
)

llm_unpriced_total = Counter(
    "gateway_llm_unpriced_total",
    "Llamadas de modelos sin precio en config/pricing.yaml. Si sube, el reporte "
    "de costos está incompleto y no se nota por ningún otro lado.",
    ["model"],
)

llm_cache_total = Counter(
    "gateway_llm_cache_total",
    "Consultas al cache por resultado (hit/miss/disabled/bypass).",
    ["project", "result"],
)

llm_guard_attempts_total = Counter(
    "gateway_llm_guard_attempts_total",
    "Intentos del guard de salida estructurada, por desenlace. Un `ok` en el "
    "intento 1 es lo normal; el resto es prompt o schema que hay que revisar.",
    ["model", "outcome"],
)

llm_duration_seconds = Histogram(
    "gateway_llm_duration_seconds",
    "Tiempo de pared de una petición lógica, reparaciones incluidas.",
    ["route", "model"],
    buckets=[0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160],
)


llm_fallback_total = Counter(
    "gateway_llm_fallback_total",
    "Saltos de un modelo al siguiente de la cadena, con el motivo. El motivo "
    "importa: caer por `no_credential` es una cuenta sin conectar, caer por "
    "`upstream_unavailable` es cuota. Se arreglan distinto.",
    ["route", "from_model", "to_model", "reason"],
)
