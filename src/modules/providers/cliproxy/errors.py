"""
Clasificación de errores de CLIProxyAPI.

El routing (F4) sólo puede decidir bien si sabe *por qué* falló algo, y los
códigos HTTP no alcanzan: `auth_not_found` (ninguna credencial cubre el modelo),
una cuota agotada y una caída de red pintan las tres como 5xx, pero sólo una se
arregla reintentando en otro modelo, y sólo otra se arregla reintentando el
mismo dentro de un rato.

La forma exacta de estos errores está fijada en `tests/fixtures/cliproxy/`.
"""

from __future__ import annotations

from typing import Any

from src.core.exceptions import GatewayError


class CliproxyError(GatewayError):
    """Base de los errores hablando con CLIProxyAPI."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class CliproxyTransportError(CliproxyError):
    """No hubo respuesta: timeout, DNS, conexión rechazada.

    Reintentable con el mismo modelo — el problema es el camino, no la elección.
    """


class CliproxyRetryableError(CliproxyError):
    """El upstream respondió, pero con algo que puede resolverse solo.

    Incluye 429 (cuota), 5xx, y los 401/403 transitorios de Gemini bajo
    concurrencia (`auth_unavailable`), que no significan credencial inválida.
    """


class CliproxyNoCredentialError(CliproxyError):
    """Ninguna credencial cargada cubre este modelo (`auth_not_found`).

    Reintentar el mismo modelo no sirve por más que se espere: hay que cambiar
    de modelo o conectar la cuenta. Distinto de una cuota agotada.
    """


class CliproxyRequestError(CliproxyError):
    """El request está mal formado o el modelo lo rechaza. No reintentar."""


# Estados HTTP que valen un reintento. 408 y 429 son explícitos; 401/403
# entran porque Gemini los devuelve de forma transitoria bajo carga.
_RETRYABLE_STATUS = frozenset({401, 403, 408, 409, 429, 500, 502, 503, 504})

# Marcadores dentro del cuerpo del error, más precisos que el status.
_NO_CREDENTIAL_MARKERS = ("auth_not_found", "no auth available")
_RETRYABLE_MARKERS = (
    "auth_unavailable",
    "cooling down",
    "model_cooldown",
    "resource_exhausted",
    "resource has been exhausted",
    "rate limit",
)


def _error_message(payload: Any) -> str:
    """Mensaje de error del cuerpo, venga en la forma que venga.

    CLIProxyAPI reenvía la forma del proveedor de arriba tal cual, así que el
    mismo campo aparece como `error.message` (OpenAI, Google) o como `error`
    a secas. Ninguna de las dos es más correcta; hay que aceptar ambas.
    """
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("status") or "")
    if isinstance(error, str):
        return error
    return ""


def classify(status_code: int, payload: Any, *, path: str) -> CliproxyError:
    """Convierte una respuesta fallida en la excepción que le corresponde.

    Se mira primero el cuerpo y después el status: un `auth_not_found` llega con
    503, y tratarlo como "el servidor está caído" haría que el routing reintente
    para siempre un modelo que ninguna credencial cubre.
    """
    message = _error_message(payload)
    lowered = message.lower()
    detail = f"{path} → {status_code}" + (f": {message}" if message else "")
    body = payload if isinstance(payload, dict) else {"raw": payload}

    if any(marker in lowered for marker in _NO_CREDENTIAL_MARKERS):
        return CliproxyNoCredentialError(detail, status_code=status_code, payload=body)

    if any(marker in lowered for marker in _RETRYABLE_MARKERS):
        return CliproxyRetryableError(detail, status_code=status_code, payload=body)

    if status_code in _RETRYABLE_STATUS:
        return CliproxyRetryableError(detail, status_code=status_code, payload=body)

    return CliproxyRequestError(detail, status_code=status_code, payload=body)


def payload_has_error(payload: Any) -> bool:
    """True si el cuerpo trae un error pese a un HTTP 200.

    Pasa de verdad: la generación de imagen de Gemini devuelve 200 con
    `error.code: 429` dentro. Sin esto, el fallo se cuela como éxito vacío.
    """
    return isinstance(payload, dict) and bool(payload.get("error"))
