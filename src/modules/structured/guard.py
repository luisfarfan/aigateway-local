"""
Guard de salida estructurada: la única puerta para pedir JSON tipado.

Un `response_format` no alcanza. Medido: los modelos de OpenAI rechazan el
schema crudo de Pydantic con HTTP 400, y Gemini y Claude descartan el campo
entero y contestan prosa. El guard cierra esa brecha con tres intentos acotados:

  1. **Contrato por partida doble** — `response_format` estricto *y* el schema
     dentro de la conversación, con los valores de cada enum escritos. El
     proveedor que respeta el primero no se entera del segundo; el que no, ahora
     tiene el contrato donde sí lo lee.
  2. **Reparación** — se cita el error exacto de parseo o validación y se
     reenuncian los campos. A un modelo que ignoró el schema no se le repite la
     misma petición opaca.
  3. **Reintento limpio** — se descarta la conversación y queda sólo el schema.
     Cuando el contexto es lo que descarriló al modelo, insistir sobre él no
     ayuda.

Agotados los tres, levanta. Nunca devuelve un objeto a medias ni inventa
defaults: un fallo silencioso acá se convierte en datos corruptos aguas abajo.

Un detalle deliberado: se **manda** el schema estricto pero se **valida** contra
el original. Lo estricto es una exigencia del transporte de OpenAI
(`additionalProperties: false`, todo en `required`), no el contrato de quien
llama. Validar contra la versión estricta rechazaría respuestas correctas que
omiten un campo opcional.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError

from src.core.exceptions import GatewayError
from src.modules.providers.cliproxy.errors import CliproxyError
from src.modules.structured.schema import (
    repair_instruction,
    schema_instruction,
    to_strict_json_schema,
)

log = structlog.get_logger(__name__)

MAX_ATTEMPTS = 3

_CODE_FENCE = re.compile(r"\A\s*```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*?)\n?\s*```\s*\Z", re.DOTALL)

# Cuánto texto del modelo se cita en un error. Suficiente para diagnosticar,
# poco para no volcar una respuesta entera al log.
_QUOTE_LIMIT = 300


class InvalidStructuredOutput(GatewayError):
    """Los tres intentos fallaron. Trae el detalle de cada uno."""

    def __init__(self, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


class ChatFn(Protocol):
    """Lo mínimo que el guard necesita: mandar mensajes y recibir texto.

    Se pide un protocolo y no el cliente entero para que el guard sirva igual
    sobre modelos cloud o locales, y para poder probarlo sin red.
    """

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]: ...


@dataclass
class Attempt:
    """Un intento físico. F3 persiste esto como fila en `llm_attempts`."""

    number: int
    outcome: str  # ok | not_json | schema_invalid | upstream_error
    duration_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


@dataclass
class GuardResult:
    value: dict[str, Any]
    model: str
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def repairs(self) -> int:
        """Intentos que hubo que gastar de más. 0 = salió al primer tiro."""
        return max(0, len(self.attempts) - 1)

    @property
    def prompt_tokens(self) -> int:
        return sum(a.prompt_tokens for a in self.attempts)

    @property
    def completion_tokens(self) -> int:
        return sum(a.completion_tokens for a in self.attempts)


async def call_with_guard(
    chat: ChatFn,
    *,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    name: str = "Respuesta",
    model: str,
    max_tokens: int = 4096,
) -> GuardResult:
    """Devuelve un objeto que cumple `schema`, o levanta `InvalidStructuredOutput`."""
    strict = to_strict_json_schema(schema)
    validator = Draft202012Validator(schema)
    contract = schema_instruction(strict, name=name)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": strict, "strict": True},
    }

    attempts: list[Attempt] = []
    conversation = [*messages, {"role": "system", "content": contract}]
    last_error = "sin intentos"

    for number in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            content, prompt_tokens, completion_tokens = await chat(
                conversation,
                model=model,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except CliproxyError as exc:
            # Un fallo de transporte no es culpa del modelo: no se gasta un
            # intento de reparación reformulando algo que nunca llegó.
            attempts.append(
                Attempt(number, "upstream_error", time.monotonic() - started, error=exc.message)
            )
            raise InvalidStructuredOutput(
                f"El upstream falló en el intento {number}: {exc.message}", attempts
            ) from exc

        elapsed = time.monotonic() - started
        parsed, failure = _parse(content, validator)

        if failure is None:
            attempts.append(Attempt(number, "ok", elapsed, prompt_tokens, completion_tokens))
            if number > 1:
                log.info("guard.repaired", model=model, attempts=number)
            return GuardResult(value=parsed, model=model, attempts=attempts)

        outcome, last_error = failure
        attempts.append(
            Attempt(number, outcome, elapsed, prompt_tokens, completion_tokens, last_error)
        )
        conversation = _next_conversation(
            number, messages, contract, strict, name, last_error, content
        )

    log.warning("guard.exhausted", model=model, attempts=len(attempts), error=last_error)
    raise InvalidStructuredOutput(
        f"El modelo {model} no produjo un '{name}' válido en {MAX_ATTEMPTS} intentos. "
        f"Último error: {last_error}",
        attempts,
    )


def _next_conversation(
    number: int,
    messages: list[dict[str, Any]],
    contract: str,
    strict: dict[str, Any],
    name: str,
    error: str,
    bad_content: str,
) -> list[dict[str, Any]]:
    """Qué se manda en el intento siguiente.

    Reparación (2): se conserva el contexto y se añade la corrección, con la
    respuesta fallida incluida para que el modelo vea qué escribió.

    Reintento limpio (3): se tira el contexto original. Si el modelo se
    descarriló por el contenido de la conversación, insistir sobre ella lo
    vuelve a descarrilar.
    """
    if number == 1:
        return [
            *messages,
            {"role": "system", "content": contract},
            {"role": "assistant", "content": bad_content[:_QUOTE_LIMIT]},
            {"role": "user", "content": repair_instruction(error, strict, name=name)},
        ]
    return [
        {"role": "system", "content": contract},
        {"role": "user", "content": repair_instruction(error, strict, name=name)},
    ]


def _parse(
    content: str, validator: Draft202012Validator
) -> tuple[dict[str, Any], tuple[str, str] | None]:
    """`(objeto, None)` si sirve; `({}, (motivo, detalle))` si no."""
    candidate = extract_json(content)
    if candidate is None:
        return {}, ("not_json", f"la respuesta no contiene JSON: {content[:_QUOTE_LIMIT]!r}")

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {}, ("not_json", f"JSON inválido ({exc}): {candidate[:_QUOTE_LIMIT]!r}")

    if not isinstance(parsed, dict):
        return {}, ("schema_invalid", f"se esperaba un objeto y llegó {type(parsed).__name__}")

    try:
        validator.validate(parsed)
    except JsonSchemaValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "(raíz)"
        return {}, ("schema_invalid", f"en {path}: {exc.message}")

    return parsed, None


def extract_json(content: str) -> str | None:
    """Saca el JSON de una respuesta que puede venir envuelta.

    Tres capas, de la más limpia a la más sucia: tal cual, sin el bloque de
    código markdown, o recortando desde la primera llave hasta la última. La
    tercera existe porque un modelo que ignoró el `response_format` suele
    devolver el objeto correcto rodeado de prosa, y tirar esa respuesta sería
    gastar un intento de reparación de gusto.
    """
    stripped = content.strip()
    if not stripped:
        return None

    if match := _CODE_FENCE.match(stripped):
        stripped = match.group(1).strip()

    if stripped.startswith("{"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return None
