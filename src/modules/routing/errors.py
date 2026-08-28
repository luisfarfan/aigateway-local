"""
Un fallo, un nombre.

El mismo fallo tiene que producir el mismo código HTTP hacia el cliente, la
misma decisión de fallback y la misma etiqueta en las métricas. Cuando cada capa
clasifica por su cuenta, terminan discrepando: el cliente ve un 503 que invita a
reintentar mientras el routing ya decidió que el modelo no sirve. Acá se
clasifica una vez.
"""

from __future__ import annotations

from src.modules.backends.base import BackendCapabilityError
from src.modules.providers.cliproxy.errors import (
    CliproxyNoCredentialError,
    CliproxyRequestError,
    CliproxyRetryableError,
    CliproxyTransportError,
)
from src.modules.structured.guard import InvalidStructuredOutput

NO_CREDENTIAL = "no_credential"
INVALID_REQUEST = "invalid_request"
UPSTREAM_UNAVAILABLE = "upstream_unavailable"
UPSTREAM_TIMEOUT = "upstream_timeout"
INVALID_OUTPUT = "invalid_output"
UPSTREAM_ERROR = "upstream_error"
# El backend no puede hacer esa operación con ese modelo (Ollama y websearch,
# por ejemplo). No es culpa de la petición ni del upstream: es un candidato mal
# elegido, y la respuesta correcta es probar el siguiente de la cadena.
UNSUPPORTED_CAPABILITY = "unsupported_capability"

# Orden importante: de lo más específico a lo más general.
_KIND_BY_TYPE: tuple[tuple[type[Exception], str], ...] = (
    # Antes que los demás: hereda de CliproxyError y si no, lo capturaría otro.
    (BackendCapabilityError, UNSUPPORTED_CAPABILITY),
    (CliproxyNoCredentialError, NO_CREDENTIAL),
    (CliproxyRequestError, INVALID_REQUEST),
    (CliproxyRetryableError, UPSTREAM_UNAVAILABLE),
    (CliproxyTransportError, UPSTREAM_TIMEOUT),
    (InvalidStructuredOutput, INVALID_OUTPUT),
)

HTTP_STATUS_BY_KIND: dict[str, int] = {
    # 502 y no 503: que ninguna credencial cubra el modelo no se arregla
    # esperando, así que el cliente no debe reintentar igual.
    NO_CREDENTIAL: 502,
    INVALID_REQUEST: 400,
    UPSTREAM_UNAVAILABLE: 503,
    UPSTREAM_TIMEOUT: 504,
    # 422: el upstream respondió, lo que no cumple es el contenido.
    INVALID_OUTPUT: 422,
    # 501: la petición es válida, este gateway no la puede servir con lo que hay.
    UNSUPPORTED_CAPABILITY: 501,
    UPSTREAM_ERROR: 502,
}

# Qué le decimos al cliente sobre reintentar tal cual, sin cambiar nada.
RETRYABLE_KINDS = frozenset({UPSTREAM_UNAVAILABLE, UPSTREAM_TIMEOUT})


def kind_of(exc: Exception) -> str:
    for error_type, kind in _KIND_BY_TYPE:
        if isinstance(exc, error_type):
            return kind
    return UPSTREAM_ERROR


def http_status_of(kind: str) -> int:
    return HTTP_STATUS_BY_KIND.get(kind, 502)


def is_retryable(kind: str) -> bool:
    return kind in RETRYABLE_KINDS


def outcome_of(kind: str) -> str:
    """Cómo se registra en `llm_requests.outcome`."""
    if kind == UPSTREAM_TIMEOUT:
        return "timeout"
    if kind == INVALID_OUTPUT:
        return "invalid_output"
    if kind == INVALID_REQUEST:
        return "invalid_request"
    if kind == UNSUPPORTED_CAPABILITY:
        return "unsupported_capability"
    return "upstream_error"
