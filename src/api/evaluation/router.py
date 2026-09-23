"""
Plano de evaluación — Jev (y lo que venga después) como servicio para los
proyectos de la red.

Dos puertas al mismo camino, porque cada una tiene clientes que ya existen:

  * `POST /v1/evaluate` — la forma nativa de Vercel AI Gateway
    (`boolean` → `probability`). La recomendada para código nuevo.
  * `POST /typesafe/v1/systemone` — la forma de TypeSafe (`noul` → `noul`).
    El SDK oficial de TypeSafe funciona cambiando sólo su `base_url`.

Lo que un proyecto gana llamando acá y no a Vercel directo es lo mismo que en
`/v1/chat/completions`: la key de Vercel no sale de esta máquina, el gasto queda
atribuido por `X-Proxima-Project`, y la ruta `evaluate` de `routing.yaml` da
fallback con circuit breaker cuando se agregue un segundo backend (Kev local).

No pasa por la cola de jobs a propósito: Jev responde en décimas de segundo, y
encolarlo agregaría más latencia que la llamada entera.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.api.openai_compat.router import (
    _annotate_routing,
    _authenticate,
    _client_id_of,
    _project_of,
    _routing,
)
from src.modules.backends.base import BackendCapabilityError
from src.modules.evaluation.registry import EvaluatorRegistry
from src.modules.evaluation.schema import (
    EvaluateRequest,
    EvaluationResult,
    TypeSafeTranslationError,
    native_to_typesafe,
    typesafe_to_native,
)
from src.modules.observability.pricing import Cost
from src.modules.observability.recorder import Observation, observe
from src.modules.routing import errors as routing_errors
from src.modules.routing.config import RoutingTable
from src.modules.routing.executor import NoCandidatesError, RouteResult, run_with_fallback

log = structlog.get_logger(__name__)

ROUTE = "evaluate"
# Si `routing.yaml` no tiene la ruta, se sigue sirviendo Jev en vez de fallar
# por config faltante — el mismo criterio que `cliproxy_default_model` en chat.
DEFAULT_MODEL = "jev"

router = APIRouter(tags=["Evaluation"], dependencies=[Depends(_authenticate)])


class _EvaluationFailed(Exception):
    """Fallo ya clasificado, listo para escribirse en cualquiera de los dos dialectos."""

    def __init__(self, *, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.message = message

    @property
    def retryable(self) -> bool:
        return routing_errors.is_retryable(self.kind)


def _evaluators(request: Request) -> EvaluatorRegistry:
    registry = getattr(request.app.state, "evaluators", None)
    return registry if registry is not None else EvaluatorRegistry()


def _candidates(table: RoutingTable, registry: EvaluatorRegistry, model: str | None) -> list[str]:
    """Cadena de `routing.yaml`, con el modelo pedido primero y todo normalizado.

    Se normaliza para que `typesafe-ai/jev` pedido por el cliente y
    `vercel/typesafe-ai/jev` de la cadena no cuenten como dos candidatos.
    """
    requested = registry.canonical(model) if model else None
    chain = table.candidates(ROUTE, requested)
    if not chain:
        chain = [requested or registry.canonical(DEFAULT_MODEL)]

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in chain:
        canonical = registry.canonical(candidate)
        if canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered[:1] if table.single_candidate else ordered


def _reported_cost(result: EvaluationResult) -> Cost | None:
    """El costo que dijo el upstream, si lo dijo. Si no, decide `pricing.yaml`."""
    if result.cost_usd is None:
        return None
    equivalent = result.market_cost_usd if result.market_cost_usd is not None else result.cost_usd
    return Cost(
        amount_usd=result.cost_usd,
        equivalent_usd=equivalent,
        priced=True,
        charged=True,
        source="upstream",
    )


async def _run(request: Request, body: EvaluateRequest) -> dict[str, Any]:
    """Evalúa por la cadena y devuelve la respuesta en forma nativa.

    Levanta `_EvaluationFailed` con el fallo ya clasificado; cada endpoint lo
    escribe en su dialecto.
    """
    registry = _evaluators(request)
    table, breaker = _routing(request)
    candidates = _candidates(table, registry, body.model)
    questions = body.questions_payload()

    obs = Observation(
        project=_project_of(request),
        route=ROUTE,
        requested_model=candidates[0],
        client_id=_client_id_of(request),
        # Todo backend de evaluación cloud se paga por token con una key.
        auth_mode="api_key",
    )
    obs.meta["questions"] = len(questions)

    async with observe(obs):

        async def attempt(model: str) -> EvaluationResult:
            resolved = registry.resolve(model)
            if resolved is None:
                raise BackendCapabilityError(f"No hay backend de evaluación para {model!r}")
            obs.family = "typesafe"
            obs.meta["backend"] = resolved.backend.name
            return await resolved.backend.evaluate(
                model=resolved.model, state=body.state, questions=questions
            )

        try:
            routed: RouteResult[EvaluationResult] = await run_with_fallback(
                attempt, route=ROUTE, table=table, breaker=breaker, candidates=candidates
            )
        except NoCandidatesError as exc:
            obs.failed(
                kind="no_candidates", message=str(exc), retryable=True, outcome="upstream_error"
            )
            raise _EvaluationFailed(status=503, kind="no_candidates", message=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — toda la cadena falló
            kind = routing_errors.kind_of(exc)
            message = getattr(exc, "message", str(exc))
            obs.failed(
                kind=kind,
                message=message,
                retryable=routing_errors.is_retryable(kind),
                outcome=routing_errors.outcome_of(kind),
            )
            obs.meta["routing_exhausted"] = True
            log.warning("evaluation.failed", kind=kind, error=message)
            raise _EvaluationFailed(
                status=routing_errors.http_status_of(kind), kind=kind, message=message
            ) from exc

        _annotate_routing(obs, routed)
        result = routed.value
        obs.prompt_tokens = result.input_tokens
        obs.completion_tokens = result.output_tokens
        obs.reported_cost = _reported_cost(result)
        obs.succeeded(model=routed.model)

        payload: dict[str, Any] = {
            "model": result.model,
            "answers": result.answers,
            "usage": {"inputTokens": result.input_tokens, "outputTokens": result.output_tokens},
        }
        if result.provider_metadata:
            payload["providerMetadata"] = result.provider_metadata
        if routed.fell_back:
            # Nunca silencioso: quien llamó tiene derecho a saber que respondió otro.
            payload["proxima"] = {"fell_back_from": routed.first_choice, "served_by": routed.model}
        return payload


# ─── Forma nativa (Vercel) ────────────────────────────────────────────────────


@router.post("/v1/evaluate", summary="Evaluación: estado + preguntas tipadas → probabilidades")
async def evaluate(request: Request, body: EvaluateRequest) -> Any:
    try:
        return await _run(request, body)
    except _EvaluationFailed as exc:
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {"message": exc.message, "type": exc.kind, "retryable": exc.retryable}
            },
        )


# ─── Forma TypeSafe ───────────────────────────────────────────────────────────


def _typesafe_error(status: int, message: str, error_type: str) -> JSONResponse:
    """La forma de error de TypeSafe: un cliente suyo ya sabe leerla."""
    return JSONResponse(status_code=status, content={"message": message, "error_type": error_type})


@router.post("/typesafe/v1/systemone", summary="Evaluación con la API de TypeSafe (System One)")
async def systemone(request: Request) -> Any:
    try:
        raw = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _typesafe_error(400, "el cuerpo no es JSON válido", "invalid_request")

    try:
        body = typesafe_to_native(raw)
    except TypeSafeTranslationError as exc:
        return _typesafe_error(400, str(exc), "invalid_request")
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        return _typesafe_error(400, f"{where}: {first.get('msg')}", "invalid_request")

    try:
        payload = await _run(request, body)
    except _EvaluationFailed as exc:
        return _typesafe_error(exc.status, exc.message, exc.kind)
    return native_to_typesafe(payload, body.questions)
