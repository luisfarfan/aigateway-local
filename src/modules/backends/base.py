"""
El contrato que cumple cualquier cosa que sirva un modelo.

Existe para que la cadena de routing pueda mezclar cloud y local sin que el
endpoint sepa cuál le tocó. La consecuencia práctica es la que se quería desde
el principio: se agota la cuota de Gemini y la petición la termina sirviendo un
modelo en la GPU de la máquina, en vez de fallar.

No todos los backends pueden todo. Ollama no hace búsqueda web ni genera
imágenes, y decirlo con un error propio —en vez de con un 500 genérico— permite
que el routing lo trate como lo que es: motivo para probar el siguiente
candidato, no para abortar la petición.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.modules.providers.cliproxy.errors import CliproxyError
from src.modules.providers.cliproxy.translate import EmbeddingResult, LLMResult, Message

# Alias con el nombre que corresponde al uso actual: estas clases nacieron para
# CLIProxyAPI pero hoy son el vocabulario de errores de cualquier upstream.
UpstreamError = CliproxyError


class BackendCapabilityError(CliproxyError):
    """El backend no puede hacer esa operación con ese modelo.

    No es un fallo del upstream ni de la petición: es un candidato mal elegido.
    El routing lo trata como motivo para pasar al siguiente de la cadena.
    """


@runtime_checkable
class Backend(Protocol):
    """Lo que el plano síncrono necesita de un backend."""

    @property
    def name(self) -> str: ...

    async def family_of(self, model: str) -> Any: ...

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> LLMResult: ...

    async def search(
        self, messages: list[Message], *, model: str, max_tokens: int = 4096
    ) -> LLMResult: ...

    async def image(
        self,
        prompt: str,
        *,
        model: str,
        size: str | None = None,
        quality: str | None = None,
    ) -> LLMResult: ...

    async def embed(self, texts: list[str], *, model: str) -> EmbeddingResult: ...

    def stream_chat(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> Any:
        """Context manager asíncrono que cede los bytes SSE del upstream."""
        ...
