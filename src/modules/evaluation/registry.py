"""
Qué backend sirve cada modelo de evaluación.

Mismo criterio que `backends/registry.py`: prefijo explícito, nunca adivinar
por el nombre. `vercel/typesafe-ai/jev` va a Vercel AI Gateway,
`openrouter/typesafe/jev-1.13` a OpenRouter, y el prefijo se quita antes de
llamar.

Registro aparte del de chat, y no un método más en `Backend`, porque evaluar es
otro contrato: un backend de evaluación no sabe hacer chat ni imagen, y obligarlo
a implementar esos métodos para levantar `BackendCapabilityError` en todos sería
ceremonia sin valor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.modules.evaluation.schema import EvaluationResult

VERCEL_PREFIX = "vercel/"
OPENROUTER_PREFIX = "openrouter/"

# Nombres cortos que un cliente puede usar sin saber por dónde entra el modelo.
ALIASES = {
    "jev": f"{VERCEL_PREFIX}typesafe-ai/jev",
}

# Cada agregador nombra a TypeSafe distinto: Vercel `typesafe-ai/…`, OpenRouter
# `typesafe/…` (y el alias `~typesafe/jev-latest`). El namespace alcanza para
# saber a quién pertenece un id sin prefijo — no es adivinar, es su catálogo.
_OPENROUTER_NAMESPACES = ("typesafe/", "~typesafe/")


@runtime_checkable
class Evaluator(Protocol):
    @property
    def name(self) -> str: ...

    async def evaluate(
        self, *, model: str, state: Any, questions: dict[str, Any]
    ) -> EvaluationResult: ...


@dataclass(frozen=True)
class ResolvedEvaluator:
    backend: Evaluator
    model: str  # sin prefijo, tal como lo espera el backend


class EvaluatorRegistry:
    def __init__(
        self, *, vercel: Evaluator | None = None, openrouter: Evaluator | None = None
    ) -> None:
        self._vercel = vercel
        self._openrouter = openrouter

    @property
    def configured(self) -> bool:
        return self._vercel is not None or self._openrouter is not None

    @staticmethod
    def canonical(model: str) -> str:
        """El id con su prefijo de backend.

        Sirve para que el modelo que pide el cliente y el de la cadena de
        `routing.yaml` se reconozcan como el mismo: `typesafe-ai/jev` y
        `vercel/typesafe-ai/jev` son una sola llamada, y sin normalizar la cadena
        los probaría a los dos — dos intentos contra el mismo upstream caído.

        Un id sin prefijo se asigna por su namespace (`typesafe/…` es de
        OpenRouter) y, si no hay pista, a Vercel: igual que CLIProxyAPI es el
        default del registro de chat.
        """
        model = model.strip()
        if model in ALIASES:
            return ALIASES[model]
        if model.startswith((VERCEL_PREFIX, OPENROUTER_PREFIX)):
            return model
        if model.startswith(_OPENROUTER_NAMESPACES):
            return f"{OPENROUTER_PREFIX}{model}"
        return f"{VERCEL_PREFIX}{model}"

    def resolve(self, model: str) -> ResolvedEvaluator | None:
        """`None` si el backend de ese modelo no está configurado."""
        model = self.canonical(model)
        for prefix, backend in (
            (VERCEL_PREFIX, self._vercel),
            (OPENROUTER_PREFIX, self._openrouter),
        ):
            if model.startswith(prefix):
                if backend is None:
                    return None
                return ResolvedEvaluator(backend, model[len(prefix) :])
        return None
