"""
Las dos superficies del cliente.

`Gateway` es async y `SyncGateway` es síncrono de verdad —usa `httpx.Client`, no
un `asyncio.run` envolviendo al async—. Envolver rompería dentro de un bucle ya
corriendo, que es exactamente donde más molesta.

Las dos comparten `_protocol`, así que la única diferencia posible entre ellas es
el transporte.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import httpx

from proxima_llm import _protocol as p
from proxima_llm.types import Completion, Embeddings

# Una imagen de entrada: ruta en disco, bytes crudos, o `(nombre, bytes)` cuando
# quien llama quiere controlar el nombre —y con él el tipo que se declara—.
ImageInput = str | Path | bytes | tuple[str, bytes]

DEFAULT_TIMEOUT_SECONDS = 300.0


class _Base:
    """Cabeceras y URLs, común a las dos superficies."""

    def __init__(
        self,
        base_url: str,
        *,
        project: str = "default",
        api_key: str | None = None,
        client_id: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._project = project
        self._timeout = timeout_seconds
        # `X-Proxima-Project` es lo que separa la contabilidad de costos entre
        # consumidores. Sin él, todo el gasto cae en un mismo balde.
        # El `Content-Type` NO se fija acá: httpx ya pone `application/json`
        # cuando se manda `json=`, y fijarlo a nivel de cliente pisaba el
        # `multipart/form-data; boundary=...` de `image_edit()` — el gateway
        # recibía un multipart etiquetado como JSON y contestaba 422.
        headers = {"X-Proxima-Project": project}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if client_id:
            headers["X-Proxima-Client"] = client_id
        self._headers = headers

    @property
    def project(self) -> str:
        return self._project

    def _headers_for(self, *, no_cache: bool) -> dict[str, str]:
        if not no_cache:
            return self._headers
        return {**self._headers, "X-Proxima-No-Cache": "1"}


class Gateway(_Base):
    """Cliente asíncrono."""

    def __init__(self, base_url: str, *, transport: Any = None, **kwargs: Any) -> None:
        super().__init__(base_url, **kwargs)
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Gateway:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def models(self) -> list[dict[str, Any]]:
        """Inventario. Ojo: lista lo configurado, no lo que responde ahora.

        Incluye los tiers (`owned_by: proxima-tier`) además de los modelos
        concretos: son valores válidos de `model`.
        """
        return (await self._request("GET", "/v1/models"))["data"]

    async def capabilities(self) -> dict[str, Any]:
        """Qué sabe hacer cada modelo y a qué resuelve cada tier, para elegir por
        programa sin leer el README."""
        return await self._request("GET", "/v1/capabilities")

    async def chat(
        self,
        prompt: str | list[p.Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        no_cache: bool = False,
    ) -> Completion:
        body = p.chat_body(p.as_messages(prompt), model=model, max_tokens=max_tokens)
        return p.read_completion(
            await self._request("POST", "/v1/chat/completions", body, no_cache=no_cache)
        )

    async def search(
        self,
        prompt: str | list[p.Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        no_cache: bool = False,
    ) -> Completion:
        """Chat con búsqueda web. Las fuentes vienen en `.sources`."""
        body = p.chat_body(
            p.as_messages(prompt), model=model, max_tokens=max_tokens, websearch=True
        )
        return p.read_completion(
            await self._request("POST", "/v1/chat/completions", body, no_cache=no_cache)
        )

    async def structured(
        self,
        prompt: str | list[p.Message],
        *,
        schema: dict[str, Any],
        name: str = "Respuesta",
        model: str | None = None,
        max_tokens: int = 4096,
        no_cache: bool = False,
    ) -> Completion:
        """JSON que cumple `schema`. El objeto validado llega en `.parsed`.

        Si el modelo no logra producirlo, el gateway levanta un error con el
        detalle de cada intento en `ProximaError.attempts` — no un objeto a
        medias ni defaults inventados.
        """
        body = p.chat_body(
            p.as_messages(prompt),
            model=model,
            max_tokens=max_tokens,
            schema=schema,
            schema_name=name,
        )
        return p.read_completion(
            await self._request("POST", "/v1/chat/completions", body, no_cache=no_cache)
        )

    async def image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
    ) -> Completion:
        body = p.image_body(prompt, model=model, size=size, quality=quality)
        return p.read_images(await self._request("POST", "/v1/images/generations", body))

    async def image_edit(
        self,
        prompt: str,
        images: ImageInput | list[ImageInput],
        *,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
    ) -> Completion:
        """Recontextualiza una FOTO real en vez de generar desde cero.

        Es lo que preserva la identidad del producto: para un catálogo, una
        imagen inventada que se le parece no sirve. `images` acepta varias
        referencias (producto + fondo de marca, p. ej.).

        Cada elemento puede ser una ruta, unos bytes, o `(nombre, bytes)`.
        """
        files = _as_files(images)
        form = p.image_edit_form(prompt, model=model, size=size, quality=quality)
        headers = self._headers_for(no_cache=False)
        try:
            response = await self._http.post(
                "/v1/images/edits", data=form, files=files, headers=headers
            )
        except httpx.HTTPError as exc:
            from proxima_llm.errors import ProximaError

            raise ProximaError(
                f"gateway inalcanzable: {exc}", kind="unreachable", retryable=True
            ) from exc

        payload = _json_or_empty(response)
        p.raise_for_error(response.status_code, payload)
        return p.read_images(payload)

    async def embed(self, texts: str | list[str], *, model: str | None = None) -> Embeddings:
        """Vectores para búsqueda semántica. Los sirve un modelo local, así que
        no consumen cuota de ninguna suscripción."""
        items = [texts] if isinstance(texts, str) else list(texts)
        return p.read_embeddings(
            await self._request("POST", "/v1/embeddings", p.embeddings_body(items, model=model))
        )

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        no_cache: bool = False,
    ) -> dict[str, Any]:
        headers = self._headers_for(no_cache=no_cache)
        try:
            response = (
                await self._http.get(path, headers=headers)
                if method == "GET"
                else await self._http.post(path, json=body, headers=headers)
            )
        except httpx.HTTPError as exc:
            # El gateway no contestó. Reintentar sí puede servir: casi siempre es
            # el gateway reiniciándose o la red.
            from proxima_llm.errors import ProximaError

            raise ProximaError(
                f"gateway inalcanzable: {exc}", kind="unreachable", retryable=True
            ) from exc

        payload = _json_or_empty(response)
        p.raise_for_error(response.status_code, payload)
        return payload


class SyncGateway(_Base):
    """Cliente síncrono. Misma API, sin `await`."""

    def __init__(self, base_url: str, *, transport: Any = None, **kwargs: Any) -> None:
        super().__init__(base_url, **kwargs)
        self._http = httpx.Client(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SyncGateway:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def models(self) -> list[dict[str, Any]]:
        """Inventario, incluidos los tiers (`owned_by: proxima-tier`)."""
        return self._request("GET", "/v1/models")["data"]

    def capabilities(self) -> dict[str, Any]:
        """Qué sabe hacer cada modelo y a qué resuelve cada tier."""
        return self._request("GET", "/v1/capabilities")

    def chat(
        self,
        prompt: str | list[p.Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        no_cache: bool = False,
    ) -> Completion:
        body = p.chat_body(p.as_messages(prompt), model=model, max_tokens=max_tokens)
        return p.read_completion(
            self._request("POST", "/v1/chat/completions", body, no_cache=no_cache)
        )

    def search(
        self,
        prompt: str | list[p.Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        no_cache: bool = False,
    ) -> Completion:
        body = p.chat_body(
            p.as_messages(prompt), model=model, max_tokens=max_tokens, websearch=True
        )
        return p.read_completion(
            self._request("POST", "/v1/chat/completions", body, no_cache=no_cache)
        )

    def structured(
        self,
        prompt: str | list[p.Message],
        *,
        schema: dict[str, Any],
        name: str = "Respuesta",
        model: str | None = None,
        max_tokens: int = 4096,
        no_cache: bool = False,
    ) -> Completion:
        body = p.chat_body(
            p.as_messages(prompt),
            model=model,
            max_tokens=max_tokens,
            schema=schema,
            schema_name=name,
        )
        return p.read_completion(
            self._request("POST", "/v1/chat/completions", body, no_cache=no_cache)
        )

    def image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
    ) -> Completion:
        body = p.image_body(prompt, model=model, size=size, quality=quality)
        return p.read_images(self._request("POST", "/v1/images/generations", body))

    def image_edit(
        self,
        prompt: str,
        images: ImageInput | list[ImageInput],
        *,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
    ) -> Completion:
        """Recontextualiza una FOTO real en vez de generar desde cero.

        Es lo que preserva la identidad del producto: para un catálogo, una
        imagen inventada que se le parece no sirve. `images` acepta varias
        referencias (producto + fondo de marca, p. ej.).

        Cada elemento puede ser una ruta, unos bytes, o `(nombre, bytes)`.
        """
        files = _as_files(images)
        form = p.image_edit_form(prompt, model=model, size=size, quality=quality)
        headers = self._headers_for(no_cache=False)
        try:
            response = self._http.post("/v1/images/edits", data=form, files=files, headers=headers)
        except httpx.HTTPError as exc:
            from proxima_llm.errors import ProximaError

            raise ProximaError(
                f"gateway inalcanzable: {exc}", kind="unreachable", retryable=True
            ) from exc

        payload = _json_or_empty(response)
        p.raise_for_error(response.status_code, payload)
        return p.read_images(payload)

    def embed(self, texts: str | list[str], *, model: str | None = None) -> Embeddings:
        items = [texts] if isinstance(texts, str) else list(texts)
        return p.read_embeddings(
            self._request("POST", "/v1/embeddings", p.embeddings_body(items, model=model))
        )

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        no_cache: bool = False,
    ) -> dict[str, Any]:
        headers = self._headers_for(no_cache=no_cache)
        try:
            response = (
                self._http.get(path, headers=headers)
                if method == "GET"
                else self._http.post(path, json=body, headers=headers)
            )
        except httpx.HTTPError as exc:
            from proxima_llm.errors import ProximaError

            raise ProximaError(
                f"gateway inalcanzable: {exc}", kind="unreachable", retryable=True
            ) from exc

        payload = _json_or_empty(response)
        p.raise_for_error(response.status_code, payload)
        return payload


def _as_files(
    images: ImageInput | list[ImageInput],
) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Normaliza las tres formas de pasar una imagen a lo que espera httpx.

    Se aceptan las tres porque el llamador rara vez tiene la misma: un script
    tiene una ruta, un worker tiene bytes que acaba de descargar, y quien
    genera al vuelo quiere ponerle nombre. Obligar a una sola sería trasladarle
    la conversión a cada consumidor.
    """
    items = images if isinstance(images, list) else [images]
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for index, item in enumerate(items):
        if isinstance(item, tuple):
            name, content = item
        elif isinstance(item, bytes | bytearray):
            name, content = f"image{index}.png", bytes(item)
        else:
            path = Path(item)
            name, content = path.name, path.read_bytes()
        files.append(("image", (name, content, _content_type_of(name))))
    return files


def _content_type_of(filename: str) -> str:
    """El upstream mira el tipo declarado, no los bytes: un JPEG anunciado como
    PNG lo rechaza. `mimetypes` acierta por extensión y PNG es el default sano
    para un nombre sin extensión."""
    guessed, _ = mimetypes.guess_type(filename)
    return guessed if guessed and guessed.startswith("image/") else "image/png"


def _json_or_empty(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {"error": {"message": response.text[:500], "type": "invalid_response"}}
    return payload if isinstance(payload, dict) else {"data": payload}
