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
