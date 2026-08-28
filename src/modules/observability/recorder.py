"""
Un solo sitio donde una llamada a un modelo se convierte en traza, métricas y
fila de histórico.

La alternativa —que cada endpoint arme su span, sume sus contadores y escriba
sus filas— garantiza que los tres se desincronicen: alguien agrega una ruta y
olvida el contador, o cuenta el costo dos veces. Acá se hace una vez.

Todo lo que no sea servir la petición es **best-effort**. Si Postgres no está,
se pierde una fila de histórico y se registra; la respuesta sale igual. El
mismo criterio que el cache: la observabilidad nunca puede ser el motivo de que
algo falle.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog

from src.core import metrics
from src.core.config import get_settings
from src.core.database import AsyncSessionLocal
from src.modules.observability import tracing
from src.modules.observability.models import LLMAttempt, LLMRequest
from src.modules.observability.pricing import Cost, cost_of_images, cost_of_tokens

log = structlog.get_logger(__name__)


@dataclass
class AttemptRecord:
    number: int
    outcome: str
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


@dataclass
class Observation:
    """Lo que el endpoint va llenando durante la petición.

    Arranca como fallo (`outcome="upstream_error"`) a propósito: si algo revienta
    en el camino y nadie marca el éxito, el registro dice que falló. El sesgo
    tiene que ir hacia reportar de más, no hacia perder errores.
    """

    project: str
    route: str
    requested_model: str
    family: str = "unknown"
    auth_mode: str = "oauth"

    response_model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    image_count: int = 0
    cache: str = "disabled"
    searched: bool = False

    outcome: str = "upstream_error"
    error_kind: str | None = None
    error_message: str | None = None
    retryable: bool = False

    client_id: str | None = None
    job_id: UUID | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def succeeded(self, *, model: str | None = None) -> None:
        self.outcome = "ok"
        if model:
            self.response_model = model

    def failed(self, *, kind: str, message: str, retryable: bool, outcome: str) -> None:
        self.outcome = outcome
        self.error_kind = kind
        self.error_message = message[:2000]
        self.retryable = retryable

    def cost(self) -> Cost:
        """Imagen se cobra por unidad; el resto por token."""
        if self.route == "image" and self.image_count:
            return cost_of_images(
                model=self.response_model or self.requested_model,
                image_count=self.image_count,
                auth_mode=self.auth_mode,
            )
        return cost_of_tokens(
            model=self.response_model or self.requested_model,
            family=self.family,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            auth_mode=self.auth_mode,
        )


@asynccontextmanager
async def observe(observation: Observation):
    """Envuelve una petición: abre el span, y al cerrar publica todo.

    El bloque `finally` corre incluso si el endpoint levanta, así que una
    excepción inesperada igual queda contabilizada como fallo en vez de
    desaparecer de las cifras.
    """
    started = time.monotonic()
    with tracing.llm_span(
        f"llm.{observation.route}",
        system="cliproxy",
        model=observation.requested_model,
        project=observation.project,
        route=observation.route,
    ) as span:
        try:
            yield observation
        finally:
            duration = time.monotonic() - started
            cost = observation.cost()
            _annotate_span(span, observation, cost)
            _publish_metrics(observation, cost, duration)
            await _persist(observation, cost, duration, span)


def _annotate_span(span: Any, obs: Observation, cost: Cost) -> None:
    tracing.record_result(
        span,
        response_model=obs.response_model or obs.requested_model,
        prompt_tokens=obs.prompt_tokens,
        completion_tokens=obs.completion_tokens,
        cost_usd=cost.amount_usd,
        cost_equivalent_usd=cost.equivalent_usd,
        priced=cost.priced,
        cache=obs.cache,
        attempts=max(1, len(obs.attempts)),
        searched=obs.searched,
    )
    for attempt in obs.attempts:
        tracing.record_attempt(
            span, number=attempt.number, outcome=attempt.outcome, error=attempt.error
        )
    if obs.outcome != "ok":
        tracing.record_error(
            span,
            kind=obs.error_kind or obs.outcome,
            message=obs.error_message or obs.outcome,
            retryable=obs.retryable,
        )


def _publish_metrics(obs: Observation, cost: Cost, duration: float) -> None:
    model = obs.response_model or obs.requested_model

    metrics.llm_requests_total.labels(obs.project, obs.route, model, obs.outcome).inc()
    metrics.llm_duration_seconds.labels(obs.route, model).observe(duration)
    metrics.llm_cache_total.labels(obs.project, obs.cache).inc()

    if obs.prompt_tokens:
        metrics.llm_tokens_total.labels(obs.project, model, "input").inc(obs.prompt_tokens)
    if obs.completion_tokens:
        metrics.llm_tokens_total.labels(obs.project, model, "output").inc(obs.completion_tokens)

    for attempt in obs.attempts:
        metrics.llm_guard_attempts_total.labels(model, attempt.outcome).inc()

    if not cost.priced:
        # Sin esta señal, un modelo sin tarifa se ve idéntico a uno gratis y el
        # reporte de costos queda incompleto en silencio.
        metrics.llm_unpriced_total.labels(model).inc()
        return

    if cost.amount_usd:
        metrics.llm_cost_usd_total.labels(obs.project, model).inc(cost.amount_usd)
    if cost.equivalent_usd:
        metrics.llm_cost_equivalent_usd_total.labels(obs.project, model).inc(cost.equivalent_usd)


async def _persist(obs: Observation, cost: Cost, duration: float, span: Any) -> None:
    trace_id = None
    try:
        context = span.get_span_context()
        if context and context.trace_id:
            trace_id = format(context.trace_id, "032x")
    except Exception:  # noqa: BLE001 — un span sin contexto no vale una excepción
        trace_id = None

    request = LLMRequest(
        project=obs.project,
        route=obs.route,
        client_id=obs.client_id,
        job_id=obs.job_id,
        requested_model=obs.requested_model,
        response_model=obs.response_model,
        family=obs.family,
        prompt_tokens=obs.prompt_tokens,
        completion_tokens=obs.completion_tokens,
        cost_usd=cost.amount_usd,
        cost_equivalent_usd=cost.equivalent_usd,
        priced=cost.priced,
        cache=obs.cache,
        attempts=max(1, len(obs.attempts)),
        searched=obs.searched,
        outcome=obs.outcome,
        error_kind=obs.error_kind,
        error_message=obs.error_message,
        duration_s=duration,
        trace_id=trace_id,
        meta=obs.meta,
    )

    if not get_settings().llm_history_enabled:
        return

    try:
        async with AsyncSessionLocal() as session:
            session.add(request)
            for attempt in obs.attempts:
                session.add(
                    LLMAttempt(
                        request_id=request.id,
                        number=attempt.number,
                        model=obs.response_model or obs.requested_model,
                        outcome=attempt.outcome,
                        prompt_tokens=attempt.prompt_tokens,
                        completion_tokens=attempt.completion_tokens,
                        duration_s=attempt.duration_s,
                        error=attempt.error,
                    )
                )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 — el histórico nunca rompe la petición
        log.warning("observability.persist_failed", route=obs.route, error=str(exc))
