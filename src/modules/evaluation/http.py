"""
Lo que comparten los backends de evaluación que hablan HTTP con un agregador.

Vercel AI Gateway y OpenRouter fallan con las mismas categorías —key inválida,
sin saldo, cuota, caída— aunque cada uno las escriba distinto. Clasificarlas en
un solo lugar garantiza que el mismo problema abra el mismo circuito y le
llegue al cliente con el mismo código, venga de donde venga.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.modules.providers.cliproxy.errors import (
    MAX_RETRY_AFTER_S,
    CliproxyError,
    CliproxyNoCredentialError,
    CliproxyRequestError,
    CliproxyRetryableError,
    CliproxyTransportError,
)

# Estados que no se arreglan esperando ni cambiando la petición: la cuenta no
# puede hacer eso. Medidos:
#   401 — key inválida (Vercel y OpenRouter).
#   403 — función fuera del plan (Vercel: ZDR con Hobby) o tope de gasto de la
#         key agotado (OpenRouter: "Key limit exceeded").
#   402 — sin saldo.
# Se clasifican como `no_credential` para que la cadena salte al siguiente
# candidato y el cliente NO reintente: hace falta una persona.
_ACCOUNT_STATUS = frozenset({401, 402, 403})
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


async def post_json(
    client: httpx.AsyncClient, path: str, body: dict[str, Any], *, where: str
) -> Any:
    """POST y cuerpo JSON, o la excepción clasificada.

    Un 200 con `error` en el cuerpo también se trata como fallo: es lo que hacen
    varios agregadores cuando el proveedor de fondo falla a mitad de camino.
    """
    try:
        response = await client.post(path, json=body)
    except httpx.TimeoutException as exc:
        raise CliproxyTransportError(f"{where} no respondió a tiempo: {exc}") from exc
    except httpx.TransportError as exc:
        raise CliproxyTransportError(f"No se pudo hablar con {where}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text[:500]}

    if response.status_code >= 400 or (isinstance(payload, dict) and payload.get("error")):
        raise classify(response, payload, where=where)
    return payload


def classify(response: httpx.Response, payload: Any, *, where: str) -> CliproxyError:
    status = response.status_code
    error = payload.get("error") if isinstance(payload, dict) else None
    message = ""
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("type") or "")
    elif isinstance(error, str):
        message = error
    detail = f"{where} → {status}" + (f": {message[:500]}" if message else "")
    body = payload if isinstance(payload, dict) else {"raw": payload}

    if status in _ACCOUNT_STATUS:
        return CliproxyNoCredentialError(detail, status_code=status, payload=body)
    if status in _RETRYABLE_STATUS or status >= 500:
        return CliproxyRetryableError(
            detail,
            status_code=status,
            payload=body,
            retry_after_s=retry_after(response.headers.get("retry-after")),
        )
    return CliproxyRequestError(detail, status_code=status, payload=body)


def retry_after(raw: str | None) -> int | None:
    """Sólo la forma en segundos; la de fecha HTTP no se ha visto en estos upstreams."""
    if not raw:
        return None
    try:
        return min(max(0, int(float(raw))), MAX_RETRY_AFTER_S)
    except ValueError:
        return None


def money(raw: Any) -> float | None:
    """Monto en USD. Vercel lo manda como string (`"0.00001155"`), OpenRouter como número."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
