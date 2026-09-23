"""Lo que devuelve el SDK. Objetos planos, sin dependencias."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Source:
    """Fuente citada por una búsqueda web."""

    uri: str
    title: str = ""


@dataclass(frozen=True)
class Image:
    """Imagen generada, siempre como data URI."""

    url: str


@dataclass
class Completion:
    """Respuesta del gateway, ya desenvuelta.

    Se expone `model` además de `requested_model` porque el gateway puede haber
    caído a otro: quien llama tiene derecho a saber quién le respondió sin leer
    los logs del servidor.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sources: list[Source] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    searched: bool = False
    cache: str = "disabled"
    fell_back_from: str | None = None
    # El objeto ya validado, cuando se pidió salida estructurada.
    parsed: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def fell_back(self) -> bool:
        return self.fell_back_from is not None


@dataclass
class Embeddings:
    """Vectores, en el mismo orden que los textos que se mandaron."""

    vectors: list[list[float]]
    model: str
    prompt_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0

    def __len__(self) -> int:
        return len(self.vectors)


@dataclass
class Evaluation:
    """Respuestas de un modelo de evaluación (Jev), una por pregunta.

    `answers` es la forma nativa tal cual. Los atajos existen porque cada tipo
    guarda su valor en un campo distinto y eso se olvida: `boolean` en
    `probability`, `choice` en `choice`, `score` en `score`.
    """

    answers: dict[str, Any]
    model: str
    input_tokens: int = 0
    fell_back_from: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def probability(self, key: str) -> float:
        """Probabilidad de `True` en una pregunta `boolean`."""
        return float(self.answers[key]["probability"])

    def choice(self, key: str) -> str:
        """Opción elegida en una pregunta `choice`."""
        return str(self.answers[key]["choice"])

    def score(self, key: str) -> float:
        """Nivel esperado (0 = el más bajo) en una pregunta `score`."""
        return float(self.answers[key]["score"])

    def probabilities(self, key: str) -> dict[str, float]:
        """Distribución completa de una pregunta `choice` o `score`."""
        return dict(self.answers[key].get("probabilities") or {})

    @property
    def fell_back(self) -> bool:
        return self.fell_back_from is not None
