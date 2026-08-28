"""
Backend de Ollama — modelos que corren en la GPU de esta máquina.

Habla la superficie OpenAI-compatible que Ollama expone en `/v1/chat/completions`,
así que reutiliza la misma traducción y el mismo parseo que el camino cloud. Un
segundo formato de mensajes sólo habría sido una segunda cosa que mantener.

Su papel en las cadenas es el de **último recurso**: es más lento y más limitado
que los modelos cloud, pero no depende de ninguna cuota ni de que haya internet.
Cuando se agota todo lo demás, esto sigue respondiendo.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from src.modules.backends.base import BackendCapabilityError
from src.modules.providers.cliproxy import translate
from src.modules.providers.cliproxy.errors import (
    CliproxyTransportError,
    classify,
    payload_has_error,
)
from src.modules.providers.cliproxy.families import Family
from src.modules.providers.cliproxy.translate import EmbeddingResult, LLMResult, Message

log = structlog.get_logger(__name__)

# Los modelos locales son bastante más lentos que los cloud: un 7B en GPU
# doméstica tarda decenas de segundos en respuestas largas. Un timeout corto acá
# convertiría el último recurso en un fallo garantizado.
DEFAULT_TIMEOUT_SECONDS = 300.0


class OllamaBackend:
    """Cliente mínimo sobre la API OpenAI-compatible de Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def family_of(self, model: str) -> Family:
        """Todo lo local se agrupa aparte: no comparte tarifa ni cuota con nada
        de lo cloud, y mezclarlos ensuciaría el cache y los costos."""
        return Family.LOCAL

    async def models(self) -> list[dict[str, Any]]:
        """Modelos instalados, en forma OpenAI para el inventario del gateway."""
        payload = await self._get("/api/tags")
        return [
            {"id": f"ollama/{m['name']}", "object": "model", "owned_by": "ollama"}
            for m in payload.get("models") or []
            if m.get("name")
        ]

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResult:
        request = translate.chat_request(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        payload = await self._post(request.path, request.body)
        return translate.parse_chat(payload, model=model)

    async def search(
        self, messages: list[Message], *, model: str, max_tokens: int = 4096
    ) -> LLMResult:
        raise BackendCapabilityError(
            f"Ollama no hace búsqueda web (modelo {model!r}); el routing debe probar otro candidato"
        )

    async def image(
        self,
        prompt: str,
        *,
        model: str,
        size: str | None = None,
        quality: str | None = None,
    ) -> LLMResult:
        raise BackendCapabilityError(
            f"Ollama no genera imágenes (modelo {model!r}); el routing debe probar otro candidato"
        )

    async def embed(self, texts: list[str], *, model: str) -> EmbeddingResult:
        """Embeddings locales. Es donde tienen más sentido: no cuestan cuota, no
        salen de la máquina, y para un RAG se calculan por millares."""
        request = translate.embeddings_request(model=model, texts=texts)
        payload = await self._post(request.path, request.body)
        return translate.parse_embeddings(payload, model=model)

    # ── Transporte ────────────────────────────────────────────────────────────

    async def _get(self, path: str) -> dict[str, Any]:
        return await self._request("GET", path)

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", path, body)

    async def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            response = (
                await self._client.get(path)
                if method == "GET"
                else await self._client.post(path, json=body)
            )
        except httpx.TimeoutException as exc:
            raise CliproxyTransportError(f"timeout en ollama {path}") from exc
        except httpx.HTTPError as exc:
            raise CliproxyTransportError(f"ollama inalcanzable en {path}: {exc}") from exc

        try:
            payload: Any = response.json()
        except ValueError:
            payload = {"error": {"message": response.text[:500]}}

        if response.status_code >= 400 or payload_has_error(payload):
            raise classify(response.status_code, payload, path=f"ollama{path}")
        return payload if isinstance(payload, dict) else {"data": payload}
