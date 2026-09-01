"""
Qué backend sirve cada modelo.

La regla es un prefijo explícito y no una heurística: `ollama/qwen2.5:7b` va al
backend local, `geminiweb/…` a la app web de Gemini, y cualquier otra cosa a
CLIProxyAPI. Adivinar por el nombre
—"si contiene `qwen` es local"— se rompe el día que un proveedor cloud sirva un
Qwen, que ya pasa hoy con `gpt-oss` en antigravity.

El prefijo se quita antes de llamar: el backend recibe el id que su API entiende.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.modules.backends.base import Backend

LOCAL_PREFIX = "ollama/"
# Último recurso de la ruta de imagen: la app web de Gemini por cookie. Prefijo
# propio y no un id suelto porque no es un modelo más de CLIProxyAPI — es otro
# backend, con otra credencial y otro contrato (ver `backends/gemini_web.py`).
GEMINI_WEB_PREFIX = "geminiweb/"


@dataclass(frozen=True)
class Resolved:
    backend: Backend
    model: str  # el id ya sin prefijo, tal como lo espera el backend


class BackendRegistry:
    """Resuelve modelo → backend. Local es opcional: sin Ollama configurado,
    un candidato `ollama/…` no se puede servir y el routing pasa al siguiente."""

    def __init__(
        self,
        *,
        cloud: Backend,
        local: Backend | None = None,
        gemini_web: Backend | None = None,
    ) -> None:
        self._cloud = cloud
        self._local = local
        self._gemini_web = gemini_web

    @property
    def local_available(self) -> bool:
        return self._local is not None

    @property
    def gemini_web_available(self) -> bool:
        return self._gemini_web is not None

    def resolve(self, model: str) -> Resolved | None:
        """`None` si el modelo pide un backend que no está configurado."""
        if model.startswith(LOCAL_PREFIX):
            if self._local is None:
                return None
            return Resolved(self._local, model[len(LOCAL_PREFIX) :])
        if model.startswith(GEMINI_WEB_PREFIX):
            if self._gemini_web is None:
                return None
            return Resolved(self._gemini_web, model[len(GEMINI_WEB_PREFIX) :])
        return Resolved(self._cloud, model)

    def is_local(self, model: str) -> bool:
        return model.startswith(LOCAL_PREFIX)
