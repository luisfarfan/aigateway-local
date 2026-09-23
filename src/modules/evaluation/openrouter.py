"""
Evaluación por OpenRouter — Jev (`typesafe/jev-1.13`) con la key de OpenRouter.

Habla `POST /api/v1/systemone`, la superficie TypeSafe de OpenRouter. Existe
también `/api/alpha/decisions`, que devuelve lo mismo (medido el 2026-09-23),
pero está marcada alpha; la de TypeSafe es un contrato publicado.

Por qué un segundo agregador para el mismo modelo: si Vercel se cae, se queda
sin cuota o su key deja de servir, Jev sigue respondiendo por acá. Medido con
10 llamadas por lado, las dos vías tienen latencias parecidas (mediana 0.4-1.9
s, dominada por la carga de TypeSafe) y el mismo precio por token.

Diferencias con Vercel que resuelve este módulo:

  * **Dialecto.** OpenRouter habla TypeSafe (`noul`); el gateway usa por dentro
    la forma nativa (`boolean`). Se traduce a la ida y a la vuelta.
  * **Costo.** Viene en `usage.cost`, como número. No hay precio de lista
    aparte, así que el equivalente es el mismo monto.
  * **Tope de la key.** OpenRouter permite fijar un límite de gasto por key;
    agotado, responde 403 "Key limit exceeded". Se clasifica como
    `no_credential`: no se arregla esperando.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.modules.evaluation.http import money, post_json
from src.modules.evaluation.schema import (
    EvaluationResult,
    native_questions_to_typesafe,
    typesafe_answers_to_native,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_TIMEOUT_SECONDS = 20.0

# OpenRouter identifica la app que llama con estas cabeceras (opcionales). Se
# mandan para que el gasto del gateway se distinga en su panel del de otras
# herramientas que usen la misma cuenta.
_APP_HEADERS = {"X-Title": "aigateway-local"}


class OpenRouterEvaluator:
    """Cliente mínimo de la API System One de OpenRouter."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **_APP_HEADERS,
            },
            timeout=timeout_seconds,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return "openrouter"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def evaluate(
        self,
        *,
        model: str,
        state: Any,
        questions: dict[str, Any],
    ) -> EvaluationResult:
        body = {
            "model": model,
            "state": state,
            "questions": native_questions_to_typesafe(questions),
        }
        payload = await post_json(
            self._client, "/v1/systemone", body, where="openrouter /v1/systemone"
        )

        usage = payload.get("usage") or {}
        cost = money(usage.get("cost"))
        return EvaluationResult(
            model=str(payload.get("model") or model),
            answers=typesafe_answers_to_native(payload.get("answers") or {}),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cost_usd=cost,
            market_cost_usd=cost,
            provider_metadata={
                "openrouter": {
                    "id": payload.get("id"),
                    "provider": payload.get("provider"),
                    "cost": cost,
                }
            },
        )
