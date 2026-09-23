"""
El contrato de evaluación, y la traducción entre sus dos dialectos.

Un modelo de evaluación (Jev, Kev) no genera texto: recibe un `state` y un
conjunto de preguntas tipadas y devuelve una probabilidad por pregunta. Hay dos
formas de pedirlo, y el gateway acepta las dos porque cada una tiene clientes
que ya existen:

  * **Nativa de Vercel** (`POST /v1/evaluate`): la pregunta de sí/no se llama
    `boolean` y responde `probability`; el uso va en camelCase.
  * **TypeSafe** (`POST /typesafe/v1/systemone`): la misma pregunta se llama
    `noul` y responde `noul`; el uso va en snake_case. Es la que habla el SDK
    oficial de TypeSafe y la que sirve Kev en local.

Por dentro se usa **una sola**: la nativa. Los backends reciben y devuelven esa
forma, y la de TypeSafe se traduce en el borde, a la entrada y a la salida. Así
un backend nuevo implementa un dialecto, no dos, y la traducción vive en un
único lugar donde se puede probar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Límites que documenta Vercel para Jev. Se validan acá y no se delegan al
# upstream por una razón concreta: una petición inválida que llega a la cadena
# gasta un intento por candidato para terminar en el mismo 400. Rechazarla antes
# es gratis y el mensaje es mejor.
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class Question(BaseModel):
    """Una pregunta en la forma nativa.

    `extra="allow"`: si el upstream agrega un campo a las preguntas, un cliente
    que ya lo manda no tiene que esperar a que este gateway lo conozca.
    """

    model_config = {"extra": "allow"}

    type: Literal["boolean", "choice", "score"]
    instructions: str = Field(min_length=1)
    criteria: Any = None

    @model_validator(mode="after")
    def _criteria_por_tipo(self) -> Question:
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not self.criteria:
                raise ValueError("una pregunta `choice` necesita `criteria`: {opción: descripción}")
            if len(self.criteria) > MAX_CHOICE_OPTIONS:
                raise ValueError(f"`choice` admite hasta {MAX_CHOICE_OPTIONS} opciones")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not (
                MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS
            ):
                raise ValueError(
                    f"una pregunta `score` necesita `criteria`: una lista de "
                    f"{MIN_SCORE_LEVELS} a {MAX_SCORE_LEVELS} niveles, de menor a mayor"
                )
        elif self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError("en `boolean`, `criteria` es opcional: {true: ..., false: ...}")
        return self


class EvaluateRequest(BaseModel):
    """Cuerpo de `POST /v1/evaluate`."""

    model_config = {"extra": "allow"}

    model: str | None = None
    # Texto, objeto o lista: el upstream acepta las tres y serializarlo acá
    # cambiaría lo que el modelo ve.
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)

    def questions_payload(self) -> dict[str, Any]:
        return {k: q.model_dump(exclude_none=True) for k, q in self.questions.items()}


@dataclass
class EvaluationResult:
    """Lo que devuelve cualquier backend de evaluación, ya en forma nativa."""

    model: str
    answers: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    # Lo que el upstream dice que costó. `None` = no lo dijo, y el costo se
    # calcula con `pricing.yaml`. Vercel sí lo informa: `cost` es lo cobrado y
    # `marketCost` el precio de lista (difieren en promociones y con BYOK).
    cost_usd: float | None = None
    market_cost_usd: float | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


# ─── Dialecto TypeSafe ────────────────────────────────────────────────────────

_TYPESAFE_TO_NATIVE = {"noul": "boolean"}


class TypeSafeTranslationError(ValueError):
    """La petición TypeSafe no se puede traducir. Es un 400."""


def typesafe_to_native(body: dict[str, Any]) -> EvaluateRequest:
    """Petición TypeSafe → nativa. Lo único que cambia es `noul` → `boolean`."""
    if not isinstance(body, dict):
        raise TypeSafeTranslationError("el cuerpo tiene que ser un objeto JSON")
    questions = body.get("questions")
    if not isinstance(questions, dict):
        raise TypeSafeTranslationError("falta `questions`")

    translated: dict[str, Any] = {}
    for key, question in questions.items():
        if not isinstance(question, dict):
            raise TypeSafeTranslationError(f"questions.{key}: tiene que ser un objeto")
        kind = question.get("type")
        if kind == "boolean":
            # En TypeSafe `boolean` no existe. Aceptarlo en silencio haría que el
            # mismo cliente funcione acá y falle contra TypeSafe directo.
            raise TypeSafeTranslationError(
                f"questions.{key}.type: expected one of 'noul', 'choice', 'score'"
            )
        translated[key] = {**question, "type": _TYPESAFE_TO_NATIVE.get(kind, kind)}

    return EvaluateRequest.model_validate({**body, "questions": translated})


def native_to_typesafe(payload: dict[str, Any], questions: dict[str, Question]) -> dict[str, Any]:
    """Respuesta nativa → TypeSafe.

    `score` recupera su `legend` (índice → etiqueta) desde los criterios de la
    pregunta, que es lo que devuelve TypeSafe y el nativo omite.
    """
    answers: dict[str, Any] = {}
    for key, answer in (payload.get("answers") or {}).items():
        if not isinstance(answer, dict):
            answers[key] = answer
            continue
        kind = answer.get("type")
        if kind == "boolean":
            answers[key] = {"type": "noul", "noul": answer.get("probability")}
        elif kind == "score":
            translated = dict(answer)
            question = questions.get(key)
            if question is not None and isinstance(question.criteria, list):
                translated.setdefault(
                    "legend", {str(i): label for i, label in enumerate(question.criteria)}
                )
            answers[key] = translated
        else:
            answers[key] = answer

    usage = payload.get("usage") or {}
    out: dict[str, Any] = {
        "model": payload.get("model"),
        "answers": answers,
        "usage": {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        },
    }
    if "providerMetadata" in payload:
        out["provider_metadata"] = payload["providerMetadata"]
    if "proxima" in payload:
        out["proxima"] = payload["proxima"]
    return out


# Para backends que hablan TypeSafe (OpenRouter, Kev): el gateway les manda la
# forma nativa traducida, y traduce de vuelta lo que responden. Es la misma
# regla que en el borde, en sentido inverso.


def native_questions_to_typesafe(questions: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {**q, "type": "noul"} if q.get("type") == "boolean" else dict(q)
        for key, q in questions.items()
    }


def typesafe_answers_to_native(answers: dict[str, Any]) -> dict[str, Any]:
    """`noul` → `boolean`; `score` pierde `legend`, que la forma nativa no trae
    (el borde TypeSafe la reconstruye desde las preguntas)."""
    out: dict[str, Any] = {}
    for key, answer in answers.items():
        if not isinstance(answer, dict):
            out[key] = answer
        elif answer.get("type") == "noul":
            out[key] = {"type": "boolean", "probability": answer.get("noul")}
        elif answer.get("type") == "score":
            out[key] = {k: v for k, v in answer.items() if k != "legend"}
        else:
            out[key] = answer
    return out
