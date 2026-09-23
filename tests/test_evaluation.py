"""
Plano de evaluación (Jev), sin red.

Tres capas, cada una contra su doble:

  * La traducción TypeSafe ↔ nativa, pura.
  * `VercelEvaluator` contra un `MockTransport` que devuelve las formas reales
    medidas contra `ai-gateway.vercel.sh` el 2026-09-23 (éxito, 400, 401, 403
    por ZDR en plan Hobby).
  * El router, con un evaluador falso: contrato HTTP, fallback y costo.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.evaluation.router import router
from src.api.middleware import register_exception_handlers
from src.modules.evaluation.openrouter import OpenRouterEvaluator
from src.modules.evaluation.registry import EvaluatorRegistry
from src.modules.evaluation.schema import (
    EvaluateRequest,
    EvaluationResult,
    TypeSafeTranslationError,
    native_to_typesafe,
    typesafe_to_native,
)
from src.modules.evaluation.vercel import VercelEvaluator
from src.modules.observability.pricing import Cost
from src.modules.observability.recorder import Observation
from src.modules.providers.cliproxy.errors import (
    CliproxyNoCredentialError,
    CliproxyRequestError,
    CliproxyRetryableError,
    CliproxyTransportError,
)

# Respuesta real de `POST /v1/evaluate`, recortada en `routing`.
VERCEL_OK = {
    "model": "typesafe-ai/jev",
    "answers": {
        "refund": {"type": "boolean", "probability": 0.98},
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 1, "technical": 0},
            "confidence": 1,
        },
        "urgency": {
            "type": "score",
            "score": 2,
            "probabilities": {"0": 0, "1": 0, "2": 1},
            "confidence": 1,
        },
    },
    "usage": {"inputTokens": 393, "outputTokens": 62},
    "providerMetadata": {
        "typesafe": {"confidence": {"team": 1, "urgency": 1}},
        "gateway": {"cost": "0", "marketCost": "0.000016506", "generationId": "gen_x"},
    },
}

QUESTIONS = {
    "refund": {"type": "boolean", "instructions": "¿Pide un reembolso?"},
    "team": {
        "type": "choice",
        "instructions": "¿Qué equipo?",
        "criteria": {"billing": "cobros", "technical": "errores"},
    },
    "urgency": {
        "type": "score",
        "instructions": "¿Qué tan urgente?",
        "criteria": ["baja", "media", "alta"],
    },
}


@pytest.fixture(autouse=True)
def settings_aisladas():
    from src.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ─── Traducción TypeSafe ↔ nativa ─────────────────────────────────────────────


def test_typesafe_noul_se_traduce_a_boolean():
    body = typesafe_to_native(
        {
            "state": "x",
            "questions": {"refund": {"type": "noul", "instructions": "¿Reembolso?"}},
        }
    )
    assert body.questions["refund"].type == "boolean"


def test_typesafe_rechaza_boolean():
    """`boolean` no existe en TypeSafe. Aceptarlo haría que el cliente funcione
    acá y falle contra TypeSafe directo."""
    with pytest.raises(TypeSafeTranslationError):
        typesafe_to_native(
            {"state": "x", "questions": {"q": {"type": "boolean", "instructions": "?"}}}
        )


def test_respuesta_nativa_a_typesafe():
    request = EvaluateRequest.model_validate({"state": "x", "questions": QUESTIONS})
    out = native_to_typesafe(VERCEL_OK, request.questions)

    assert out["answers"]["refund"] == {"type": "noul", "noul": 0.98}
    assert out["answers"]["team"]["choice"] == "billing"
    # TypeSafe devuelve la leyenda del score; el nativo no, se reconstruye.
    assert out["answers"]["urgency"]["legend"] == {"0": "baja", "1": "media", "2": "alta"}
    assert out["usage"] == {"input_tokens": 393, "output_tokens": 62}
    assert "provider_metadata" in out


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "?"},  # sin opciones
        {"type": "score", "instructions": "?", "criteria": ["solo-uno"]},
        {"type": "score", "instructions": "?", "criteria": [str(i) for i in range(11)]},
        {"type": "nope", "instructions": "?"},
        {"type": "boolean", "instructions": ""},
    ],
)
def test_preguntas_invalidas_se_rechazan_antes_de_la_cadena(question: dict[str, Any]):
    with pytest.raises(ValueError):
        EvaluateRequest.model_validate({"state": "x", "questions": {"q": question}})


# ─── Registro ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["jev", "typesafe-ai/jev", "vercel/typesafe-ai/jev"])
def test_los_nombres_de_jev_son_el_mismo_modelo(model: str):
    assert EvaluatorRegistry.canonical(model) == "vercel/typesafe-ai/jev"


def test_sin_backend_vercel_no_resuelve():
    assert EvaluatorRegistry().resolve("jev") is None


# ─── VercelEvaluator ──────────────────────────────────────────────────────────


def _evaluator(handler, **kwargs: Any) -> VercelEvaluator:
    return VercelEvaluator(api_key="k", transport=httpx.MockTransport(handler), **kwargs)


@pytest.mark.asyncio
async def test_vercel_parsea_respuesta_y_costo():
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["path"] = request.url.path
        sent["auth"] = request.headers["authorization"]
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json=VERCEL_OK)

    result = await _evaluator(handler).evaluate(
        model="typesafe-ai/jev", state="x", questions=QUESTIONS
    )

    assert sent["path"] == "/v1/evaluate"
    assert sent["auth"] == "Bearer k"
    # ZDR apagado por defecto: con plan Hobby, pedirlo rompe la petición.
    assert "providerOptions" not in sent["body"]
    assert result.answers["refund"]["probability"] == 0.98
    assert (result.input_tokens, result.output_tokens) == (393, 62)
    assert result.cost_usd == 0.0
    assert result.market_cost_usd == pytest.approx(0.000016506)


@pytest.mark.asyncio
async def test_vercel_zdr_se_pide_si_esta_encendido():
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=VERCEL_OK)

    await _evaluator(handler, zero_data_retention=True).evaluate(
        model="typesafe-ai/jev", state="x", questions=QUESTIONS
    )
    assert sent["providerOptions"] == {"gateway": {"zeroDataRetention": True}}


@pytest.mark.parametrize(
    ("status", "payload", "headers", "expected"),
    [
        (
            400,
            {"error": {"message": "questions.q.type: Invalid", "type": "invalid_request_error"}},
            {},
            CliproxyRequestError,
        ),
        (
            401,
            {"error": {"message": "Authentication failed", "type": "authentication_error"}},
            {},
            CliproxyNoCredentialError,
        ),
        (
            403,
            {"error": {"message": "Zero Data Retention (ZDR) is only available for Pro"}},
            {},
            CliproxyNoCredentialError,
        ),
        (
            429,
            {"error": {"message": "rate limited"}},
            {"retry-after": "30"},
            CliproxyRetryableError,
        ),
        (503, {"error": {"message": "down"}}, {}, CliproxyRetryableError),
    ],
)
@pytest.mark.asyncio
async def test_vercel_clasifica_errores(status, payload, headers, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, headers=headers)

    with pytest.raises(expected) as info:
        await _evaluator(handler).evaluate(model="typesafe-ai/jev", state="x", questions=QUESTIONS)
    if headers.get("retry-after"):
        assert info.value.retry_after_s == 30


@pytest.mark.asyncio
async def test_vercel_timeout_es_transporte():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("lento", request=request)

    with pytest.raises(CliproxyTransportError):
        await _evaluator(handler).evaluate(model="typesafe-ai/jev", state="x", questions=QUESTIONS)


# ─── Costo ────────────────────────────────────────────────────────────────────


def test_el_costo_informado_gana_sobre_la_tabla():
    reported = Cost(0.0, 0.000016506, priced=True, charged=True, source="upstream")
    obs = Observation(project="p", route="evaluate", requested_model="vercel/typesafe-ai/jev")
    obs.reported_cost = reported
    assert obs.cost() is reported


def test_sin_costo_informado_usa_la_tabla():
    obs = Observation(
        project="p",
        route="evaluate",
        requested_model="vercel/typesafe-ai/jev",
        auth_mode="api_key",
        prompt_tokens=1_000_000,
    )
    cost = obs.cost()
    assert cost.priced and cost.charged
    assert cost.amount_usd == pytest.approx(0.042)


# ─── Router ───────────────────────────────────────────────────────────────────


class FakeEvaluator:
    name = "fake"

    def __init__(self, raises: Exception | None = None) -> None:
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def evaluate(self, *, model: str, state: Any, questions: dict[str, Any]):
        self.calls.append({"model": model, "state": state, "questions": questions})
        if self._raises:
            raise self._raises
        return EvaluationResult(
            model=model,
            answers=VERCEL_OK["answers"],
            input_tokens=393,
            output_tokens=62,
            cost_usd=0.0,
            market_cost_usd=0.000016506,
            provider_metadata=VERCEL_OK["providerMetadata"],
        )


class FakeSettings:
    llm_default_project = "tests"
    llm_require_project = True


def build_app(
    evaluator: FakeEvaluator | None, *, openrouter: FakeEvaluator | None = None
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    register_exception_handlers(app)
    app.state.evaluators = EvaluatorRegistry(vercel=evaluator, openrouter=openrouter)
    app.state.settings = FakeSettings()
    return app


async def call(app: FastAPI, path: str, body: dict[str, Any], **kwargs: Any):
    headers = {"X-Proxima-Project": "tests-eval", **(kwargs.pop("headers", None) or {})}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        return await http.post(path, json=body, headers=headers)


@pytest.mark.asyncio
async def test_evaluate_forma_nativa():
    fake = FakeEvaluator()
    response = await call(
        build_app(fake),
        "/v1/evaluate",
        {"model": "typesafe-ai/jev", "state": "x", "questions": QUESTIONS},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answers"]["refund"] == {"type": "boolean", "probability": 0.98}
    assert data["usage"] == {"inputTokens": 393, "outputTokens": 62}
    assert "proxima" not in data  # sin fallback
    # Pedido como `typesafe-ai/jev` y en la cadena como `vercel/…`: una sola llamada.
    assert [c["model"] for c in fake.calls] == ["typesafe-ai/jev"]


@pytest.mark.asyncio
async def test_evaluate_sin_modelo_usa_la_cadena():
    fake = FakeEvaluator()
    response = await call(build_app(fake), "/v1/evaluate", {"state": "x", "questions": QUESTIONS})
    assert response.status_code == 200
    assert fake.calls[0]["model"] == "typesafe-ai/jev"


@pytest.mark.asyncio
async def test_evaluate_exige_proyecto():
    response = await call(
        build_app(FakeEvaluator()),
        "/v1/evaluate",
        {"state": "x", "questions": QUESTIONS},
        headers={"X-Proxima-Project": ""},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_evaluate_pregunta_invalida_no_llega_al_backend():
    fake = FakeEvaluator()
    response = await call(
        build_app(fake),
        "/v1/evaluate",
        {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?"}}},
    )
    assert response.status_code in (400, 422)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_evaluate_sin_backend_configurado():
    response = await call(build_app(None), "/v1/evaluate", {"state": "x", "questions": QUESTIONS})
    assert response.status_code == 501
    assert response.json()["error"]["type"] == "unsupported_capability"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "status", "kind", "retryable"),
    [
        (CliproxyRequestError("mala"), 400, "invalid_request", False),
        (CliproxyNoCredentialError("key"), 502, "no_credential", False),
        (CliproxyRetryableError("429"), 503, "upstream_unavailable", True),
        (CliproxyTransportError("timeout"), 504, "upstream_timeout", True),
    ],
)
async def test_evaluate_errores_del_upstream(exc, status, kind, retryable):
    response = await call(
        build_app(FakeEvaluator(raises=exc)), "/v1/evaluate", {"state": "x", "questions": QUESTIONS}
    )
    assert response.status_code == status
    assert response.json()["error"]["type"] == kind
    assert response.json()["error"]["retryable"] is retryable


@pytest.mark.asyncio
async def test_systemone_forma_typesafe():
    fake = FakeEvaluator()
    questions = {**QUESTIONS, "refund": {"type": "noul", "instructions": "¿Reembolso?"}}
    response = await call(
        build_app(fake), "/typesafe/v1/systemone", {"state": "x", "questions": questions}
    )

    assert response.status_code == 200
    data = response.json()
    # El backend recibió la forma nativa…
    assert fake.calls[0]["questions"]["refund"]["type"] == "boolean"
    # …y el cliente TypeSafe recibe la suya.
    assert data["answers"]["refund"] == {"type": "noul", "noul": 0.98}
    assert data["usage"] == {"input_tokens": 393, "output_tokens": 62}


@pytest.mark.asyncio
async def test_systemone_error_en_forma_typesafe():
    response = await call(
        build_app(FakeEvaluator()),
        "/typesafe/v1/systemone",
        {"state": "x", "questions": {"q": {"type": "boolean", "instructions": "?"}}},
    )
    assert response.status_code == 400
    assert response.json()["error_type"] == "invalid_request"
    assert "noul" in response.json()["message"]


@pytest.mark.asyncio
async def test_systemone_fallo_upstream_en_forma_typesafe():
    response = await call(
        build_app(FakeEvaluator(raises=CliproxyRetryableError("429"))),
        "/typesafe/v1/systemone",
        {"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}},
    )
    assert response.status_code == 503
    assert response.json()["error_type"] == "upstream_unavailable"


# ─── SDK ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sdk_evaluate_contra_el_router_real():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk" / "python"))
    from proxima_llm import Gateway, ProximaError

    fake = FakeEvaluator()
    gw = Gateway("http://gw", transport=ASGITransport(app=build_app(fake)), project="tienda")
    ev = await gw.evaluate("Me cobraron dos veces", QUESTIONS)

    assert ev.probability("refund") == 0.98
    assert ev.choice("team") == "billing"
    assert ev.score("urgency") == 2
    assert ev.input_tokens == 393
    assert not ev.fell_back

    failing = Gateway(
        "http://gw",
        transport=ASGITransport(app=build_app(FakeEvaluator(raises=CliproxyRetryableError("429")))),
        project="tienda",
    )
    with pytest.raises(ProximaError) as info:
        await failing.evaluate("x", QUESTIONS)
    assert info.value.retryable


# ─── OpenRouter ───────────────────────────────────────────────────────────────

# Respuesta real de `POST https://openrouter.ai/api/v1/systemone` (2026-09-23).
OPENROUTER_OK = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "refund": {"type": "noul", "noul": 0.98},
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"technical": 0, "billing": 1},
            "confidence": 1,
        },
        "urgency": {
            "type": "score",
            "score": 2,
            "legend": {"0": "baja", "1": "media", "2": "alta"},
            "probabilities": {"0": 0, "1": 0, "2": 1},
            "confidence": 1,
        },
    },
    "usage": {"input_tokens": 380, "output_tokens": 62, "cost": 0.00001596},
    "id": "gen-dec-1790192023-bD5GHna32IeHUf6jL8TU",
    "provider": "TypeSafe",
}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("typesafe/jev-1.13", "openrouter/typesafe/jev-1.13"),
        ("~typesafe/jev-latest", "openrouter/~typesafe/jev-latest"),
        ("openrouter/typesafe/jev-1.13", "openrouter/typesafe/jev-1.13"),
    ],
)
def test_namespace_de_openrouter(model: str, expected: str):
    assert EvaluatorRegistry.canonical(model) == expected


@pytest.mark.asyncio
async def test_openrouter_traduce_ida_y_vuelta():
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["path"] = request.url.path
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json=OPENROUTER_OK)

    evaluator = OpenRouterEvaluator(api_key="k", transport=httpx.MockTransport(handler))
    result = await evaluator.evaluate(model="typesafe/jev-1.13", state="x", questions=QUESTIONS)

    assert sent["path"] == "/api/v1/systemone"
    # A OpenRouter le llega TypeSafe…
    assert sent["body"]["questions"]["refund"]["type"] == "noul"
    assert sent["body"]["questions"]["team"]["type"] == "choice"
    # …y el gateway recibe la forma nativa, igual que de Vercel.
    assert result.answers["refund"] == {"type": "boolean", "probability": 0.98}
    assert "legend" not in result.answers["urgency"]
    assert result.answers["urgency"]["score"] == 2
    assert (result.input_tokens, result.output_tokens) == (380, 62)
    assert result.cost_usd == pytest.approx(0.00001596)


@pytest.mark.asyncio
async def test_openrouter_tope_de_key_es_no_credential():
    """Medido: con el tope de gasto de la key agotado, 403 "Key limit exceeded"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error": {"message": "Key limit exceeded (total limit).", "code": 403}}
        )

    evaluator = OpenRouterEvaluator(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(CliproxyNoCredentialError):
        await evaluator.evaluate(model="typesafe/jev-1.13", state="x", questions=QUESTIONS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vercel_error",
    [CliproxyRetryableError("429"), CliproxyNoCredentialError("key"), CliproxyTransportError("t")],
)
async def test_si_vercel_falla_responde_openrouter(vercel_error: Exception):
    vercel = FakeEvaluator(raises=vercel_error)
    openrouter = FakeEvaluator()
    response = await call(
        build_app(vercel, openrouter=openrouter),
        "/v1/evaluate",
        {"state": "x", "questions": QUESTIONS},
    )

    assert response.status_code == 200
    assert [c["model"] for c in vercel.calls] == ["typesafe-ai/jev"]
    assert [c["model"] for c in openrouter.calls] == ["typesafe/jev-1.13"]
    # Nunca silencioso.
    assert response.json()["proxima"] == {
        "fell_back_from": "vercel/typesafe-ai/jev",
        "served_by": "openrouter/typesafe/jev-1.13",
    }


@pytest.mark.asyncio
async def test_peticion_invalida_no_salta_a_openrouter():
    """Un 400 lo rechazaría OpenRouter igual: saltar sólo gasta."""
    vercel = FakeEvaluator(raises=CliproxyRequestError("mala"))
    openrouter = FakeEvaluator()
    response = await call(
        build_app(vercel, openrouter=openrouter),
        "/v1/evaluate",
        {"state": "x", "questions": QUESTIONS},
    )
    assert response.status_code == 400
    assert openrouter.calls == []


@pytest.mark.asyncio
async def test_no_fallback_se_respeta():
    vercel = FakeEvaluator(raises=CliproxyRetryableError("429"))
    openrouter = FakeEvaluator()
    response = await call(
        build_app(vercel, openrouter=openrouter),
        "/v1/evaluate",
        {"state": "x", "questions": QUESTIONS},
        headers={"X-Proxima-No-Fallback": "1"},
    )
    assert response.status_code == 503
    assert openrouter.calls == []


@pytest.mark.asyncio
async def test_pedir_openrouter_explicito():
    vercel = FakeEvaluator()
    openrouter = FakeEvaluator()
    response = await call(
        build_app(vercel, openrouter=openrouter),
        "/v1/evaluate",
        {"model": "typesafe/jev-1.13", "state": "x", "questions": QUESTIONS},
    )
    assert response.status_code == 200
    assert vercel.calls == []
    assert [c["model"] for c in openrouter.calls] == ["typesafe/jev-1.13"]
