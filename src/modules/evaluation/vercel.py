"""
Evaluación por Vercel AI Gateway — hoy, Jev de TypeSafe AI.

Habla `POST /v1/evaluate`, la superficie nativa de Vercel. No la de TypeSafe
(`/typesafe/v1/systemone`), aunque también existe: el gateway usa por dentro la
forma nativa (ver `schema.py`) y traducir dos veces sólo agrega dónde romperse.

Lo que este backend resuelve y el cliente no tiene por qué saber:

  * **La credencial.** La key de Vercel vive acá; los proyectos usan la del
    gateway. Si un proyecto se filtra, no se filtra la de Vercel.
  * **El costo real.** Vercel lo informa en cada respuesta
    (`providerMetadata.gateway.cost`), así que no hace falta adivinarlo con la
    tabla de precios. Medido el 2026-09-23: `cost: "0"` durante la promoción y
    `marketCost: "0.000016506"` para 393 tokens de entrada.
  * **Qué fallo es cuál.** Ver `http.classify`.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.modules.evaluation.http import money, post_json
from src.modules.evaluation.schema import EvaluationResult

DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh"

# Jev responde en ~0.2 s; el viaje completo medido va de 0.4 a 5 s según la
# carga de TypeSafe. Un timeout largo sólo retrasaría el salto al siguiente
# candidato cuando Vercel no responde.
DEFAULT_TIMEOUT_SECONDS = 20.0


class VercelEvaluator:
    """Cliente mínimo de la API de evaluación de Vercel AI Gateway."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        zero_data_retention: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # Apagado por defecto porque NO es gratis: con el plan Hobby Vercel
        # rechaza la petición entera con 403 (`ZdrUnauthorizedError`). Se
        # enciende por config cuando la cuenta lo permite, y entonces aplica a
        # todas las llamadas — no es algo que cada proyecto deba recordar pedir.
        self._zero_data_retention = zero_data_retention
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout_seconds,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return "vercel"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def evaluate(
        self,
        *,
        model: str,
        state: Any,
        questions: dict[str, Any],
    ) -> EvaluationResult:
        body: dict[str, Any] = {"model": model, "state": state, "questions": questions}
        if self._zero_data_retention:
            body["providerOptions"] = {"gateway": {"zeroDataRetention": True}}

        payload = await post_json(self._client, "/v1/evaluate", body, where="vercel /v1/evaluate")
        return _parse(payload, requested=model)


def _parse(payload: dict[str, Any], *, requested: str) -> EvaluationResult:
    usage = payload.get("usage") or {}
    metadata = payload.get("providerMetadata") or {}
    gateway = metadata.get("gateway") or {}
    return EvaluationResult(
        model=str(payload.get("model") or requested),
        answers=payload.get("answers") or {},
        input_tokens=int(usage.get("inputTokens") or 0),
        output_tokens=int(usage.get("outputTokens") or 0),
        cost_usd=money(gateway.get("cost")),
        market_cost_usd=money(gateway.get("marketCost")),
        provider_metadata=metadata,
    )
