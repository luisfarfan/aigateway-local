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
