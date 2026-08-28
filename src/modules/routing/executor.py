"""
Ejecuta una petición sobre una cadena de modelos, cayendo al siguiente cuando
corresponde.

Las tres reglas que definen el comportamiento:

1. **No todo fallo justifica un fallback.** Una petición mal formada la va a
   rechazar el siguiente modelo también; reintentar sólo gasta cuota y latencia
   para llegar al mismo 400. Qué sí justifica saltar está en `routing.yaml`.
2. **Un modelo con el circuito abierto ni se intenta.** Ya se sabe que viene
   fallando; descubrirlo otra vez cuesta una llamada perdida por petición.
3. **El fallback nunca es silencioso.** El modelo que respondió viaja en la
   respuesta y en la traza, y cada salto queda como intento registrado.

Si toda la cadena falla, se levanta el último error **sustantivo**: el que dice
algo del upstream. Un candidato cuyo backend no está configurado produce
`unsupported_capability`, que describe la configuración, no el problema; si ese
fuera el último de la cadena, taparía el 429 que causó todo y mandaría a
diagnosticar el lado equivocado.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.core import metrics
from src.modules.routing.breaker import CircuitBreaker
from src.modules.routing.config import RoutingTable
from src.modules.routing.errors import UNSUPPORTED_CAPABILITY, kind_of

log = structlog.get_logger(__name__)


class NoCandidatesError(RuntimeError):
    """No quedó ningún modelo por probar.

    Distinto de "todos fallaron": acá ni siquiera se pudo intentar, casi siempre
    porque toda la cadena tiene el circuito abierto o la ruta está vacía.
    """


@dataclass
class ModelAttempt:
    model: str
    outcome: str  # ok | skipped_open | <error kind>
    duration_s: float = 0.0
    error: str | None = None


@dataclass
class RouteResult[T]:
    value: T
    model: str
    attempts: list[ModelAttempt] = field(default_factory=list)

    @property
    def fell_back(self) -> bool:
        return any(a.outcome != "ok" for a in self.attempts)

    @property
    def first_choice(self) -> str | None:
        return self.attempts[0].model if self.attempts else None


async def run_with_fallback[T](
    call: Callable[[str], Awaitable[T]],
    *,
    route: str,
    table: RoutingTable,
    breaker: CircuitBreaker,
    requested_model: str | None = None,
) -> RouteResult[T]:
    """Prueba `call(model)` sobre la cadena hasta que uno responda."""
    candidates = table.candidates(route, requested_model)
    if not candidates:
        raise NoCandidatesError(f"La ruta {route!r} no tiene modelos configurados")

    attempts: list[ModelAttempt] = []
    last_error: Exception | None = None
    # El último que dice algo del upstream, ignorando candidatos inservibles.
    last_substantive: Exception | None = None

    for model in candidates:
        if await breaker.is_open(model):
            attempts.append(ModelAttempt(model, "skipped_open"))
            log.info("routing.skipped_open", route=route, model=model)
            continue

        started = time.monotonic()
        try:
            value = await call(model)
        except Exception as exc:  # noqa: BLE001 — se clasifica abajo
            elapsed = time.monotonic() - started
            kind = kind_of(exc)
            attempts.append(ModelAttempt(model, kind, elapsed, str(exc)[:300]))
            last_error = exc
            if kind != UNSUPPORTED_CAPABILITY:
                last_substantive = exc

            await breaker.record_failure(model)

            if not table.should_fallback(kind):
                # El siguiente modelo fallaría igual. Se corta acá para no gastar
                # cuota ni latencia llegando al mismo error.
                log.info("routing.no_fallback", route=route, model=model, kind=kind)
                raise

            log.warning("routing.fallback", route=route, from_model=model, kind=kind)
            continue

        await breaker.record_success(model)
        attempts.append(ModelAttempt(model, "ok", time.monotonic() - started))

        result = RouteResult(value=value, model=model, attempts=attempts)
        if result.fell_back:
            _count_fallback(route, attempts, model)
        return result

    if (final := last_substantive or last_error) is not None:
        raise final
    raise NoCandidatesError(
        f"Todos los modelos de {route!r} tienen el circuito abierto: {candidates}"
    )


def _count_fallback(route: str, attempts: list[ModelAttempt], winner: str) -> None:
    """Una métrica por salto, con el motivo.

    Interesa el motivo y no sólo el número: una cadena que cae por
    `no_credential` es una cuenta que hay que conectar, y una que cae por
    `upstream_unavailable` es cuota. Se arreglan de formas distintas.
    """
    for attempt in attempts:
        if attempt.outcome in ("ok", "skipped_open"):
            continue
        metrics.llm_fallback_total.labels(route, attempt.model, winner, attempt.outcome).inc()


def summarize(attempts: list[ModelAttempt]) -> list[dict[str, Any]]:
    """Los intentos en forma serializable, para la respuesta y el histórico."""
    return [{"model": a.model, "outcome": a.outcome, "error": a.error} for a in attempts]
