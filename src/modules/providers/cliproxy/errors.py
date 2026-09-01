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

import re
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
        retry_after_s: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}
        # Cuánto pide esperar el upstream, cuando lo dice. El routing lo usa para
        # abrir el circuito por el tiempo REAL en vez de por un default fijo:
        # reintentar cada 2 minutos algo que el propio Google dice que vuelve en
        # 50 horas es gastar latencia para llegar al mismo 429.
        self.retry_after_s = retry_after_s


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


# `1h2m3.5s`, `419364.77s`, `90s` — la forma que usan tanto `RetryInfo.retryDelay`
# como `quotaResetDelay`, y también el texto del mensaje ("reset after 50h58m28s").
_DURATION_RE = re.compile(
    r"(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?", re.IGNORECASE
)

# Tope de seguridad. Un valor absurdo —por un parseo malo o un upstream que
# miente— dejaría un modelo fuera de la cadena para siempre, y eso se nota
# demasiado tarde. Un día es más de lo que dura cualquier cuota real que hayamos
# medido (la peor fue ~116 h, pero ahí conviene volver a probar antes).
MAX_RETRY_AFTER_S = 24 * 3600


def _duration_to_seconds(text: str) -> int | None:
    """Segundos de una duración estilo Google, o `None` si no se entiende."""
    match = _DURATION_RE.fullmatch(text.strip())
    if not match or not any(match.groupdict().values()):
        return None
    h, m, sec = match.group("h"), match.group("m"), match.group("s")
    total = int(h or 0) * 3600 + int(m or 0) * 60 + float(sec or 0)
    return int(total) if total > 0 else None


def parse_retry_after(payload: Any) -> int | None:
    """Cuánto pide esperar el upstream, si lo dice en algún sitio conocido.

    Google lo repite en tres lugares distintos del mismo 429 y ninguno está
    garantizado, así que se prueban los tres en orden de fiabilidad: el
    `RetryInfo` estructurado primero, después el `quotaResetDelay` de los
    metadatos, y por último el texto del mensaje — que es el menos estable pero
    a veces es el único que viene.

    Devuelve `None` cuando no hay nada legible: ahí el breaker usa su default.
    """
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None

    candidatos: list[str] = []
    for detail in error.get("details") or []:
        if not isinstance(detail, dict):
            continue
        if delay := detail.get("retryDelay"):
            candidatos.append(str(delay))
        metadata = detail.get("metadata")
        if isinstance(metadata, dict) and (delay := metadata.get("quotaResetDelay")):
            candidatos.append(str(delay))

    if match := re.search(
        r"reset (?:after|in)\s+([0-9hms.]+)", str(error.get("message") or ""), re.IGNORECASE
    ):
        candidatos.append(match.group(1))

    for texto in candidatos:
        if (seconds := _duration_to_seconds(texto)) is not None:
            return min(seconds, MAX_RETRY_AFTER_S)
    return None


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
    retry_after = parse_retry_after(payload)

    if any(marker in lowered for marker in _NO_CREDENTIAL_MARKERS):
        return CliproxyNoCredentialError(
            detail, status_code=status_code, payload=body, retry_after_s=retry_after
        )

    if any(marker in lowered for marker in _RETRYABLE_MARKERS):
        return CliproxyRetryableError(
            detail, status_code=status_code, payload=body, retry_after_s=retry_after
        )

    if status_code in _RETRYABLE_STATUS:
        return CliproxyRetryableError(
            detail, status_code=status_code, payload=body, retry_after_s=retry_after
        )

    return CliproxyRequestError(
        detail, status_code=status_code, payload=body, retry_after_s=retry_after
    )


def payload_has_error(payload: Any) -> bool:
    """True si el cuerpo trae un error pese a un HTTP 200.

    Pasa de verdad: la generación de imagen de Gemini devuelve 200 con
    `error.code: 429` dentro. Sin esto, el fallo se cuela como éxito vacío.
    """
    return isinstance(payload, dict) and bool(payload.get("error"))
