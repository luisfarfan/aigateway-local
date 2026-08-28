"""
Cliente HTTP de CLIProxyAPI.

Une las tres piezas del módulo: resuelve la familia del modelo contra el
catálogo (`families`), traduce el request a la superficie que corresponde
(`translate`), y convierte cualquier fallo en una excepción que el routing pueda
usar para decidir (`errors`).

Dos detalles que parecen menores y no lo son:

  * **`base_url` con y sin `/v1`.** Los consumidores existentes lo escriben de
    las dos formas. Un 404 por una barra de más es un fallo tonto y difícil de
    ver, así que se normaliza al construir.
  * **HTTP 200 con error dentro.** La generación de imagen de Gemini responde
    200 con `error.code: 429` en el cuerpo. Sin mirar el cuerpo, ese fallo se
    cuela como un éxito vacío.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from src.modules.providers.cliproxy import translate
from src.modules.providers.cliproxy.errors import (
    CliproxyTransportError,
    classify,
    payload_has_error,
)
from src.modules.providers.cliproxy.families import (
    Family,
    family_from_model_id,
    family_from_owned_by,
)
from src.modules.providers.cliproxy.translate import LLMResult

log = structlog.get_logger(__name__)


def normalize_base_url(base_url: str) -> str:
    """Quita la barra final y un `/v1` sobrante.

    Las rutas se construyen siempre con su prefijo completo (`/v1/...`,
    `/v1beta/...`), así que la base no debe traerlo.
    """
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[: -len("/v1")]
    return cleaned


class ModelCatalog:
    """Catálogo de modelos con su `owned_by`, cacheado con TTL.

    Existe para no pagar un `GET /v1/models` por request, y porque `owned_by` es
    lo único que distingue dos caminos con el mismo id de modelo.

    Ojo — y está medido: **listar no es estar vivo**. Una instancia puede
    anunciar 62 modelos de los que 32 responden `token revoked` o `403`. Este
    catálogo sirve para saber *de quién* es un modelo, nunca para saber si
    responde. Eso lo decide el watchdog probando de verdad (F4).
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._owners: dict[str, str] = {}
        self._fetched_at: float = 0.0

    def is_stale(self, now: float) -> bool:
        return not self._owners or (now - self._fetched_at) > self._ttl

    def update(self, models: list[dict[str, Any]], *, now: float) -> None:
        self._owners = {
            model["id"]: model.get("owned_by", "")
            for model in models
            if isinstance(model, dict) and model.get("id")
        }
        self._fetched_at = now

    def owner_of(self, model: str) -> str | None:
        return self._owners.get(model)

    @property
    def model_ids(self) -> list[str]:
        return list(self._owners)


class CliproxyClient:
    """Cliente asíncrono. Una instancia por proceso; reutiliza conexiones."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 120.0,
        image_timeout_seconds: float = 300.0,
        catalog_ttl_seconds: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = normalize_base_url(base_url)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
            transport=transport,
        )
        # La generación de imagen es de otro orden de magnitud: medido, la
        # misma llamada tardó 30 s un rato y 148 s otro. Con el timeout de chat
        # se cancelaría a mitad, gastando la cuota sin traer nada.
        self._image_timeout = image_timeout_seconds
        self._catalog = ModelCatalog(ttl_seconds=catalog_ttl_seconds)

    @property
    def name(self) -> str:
        """Identidad del backend, para el protocolo `Backend` y las trazas."""
        return "cliproxy"

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> CliproxyClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # ── Catálogo ──────────────────────────────────────────────────────────────

    async def models(self, *, force: bool = False) -> list[dict[str, Any]]:
        """`GET /v1/models`, con el catálogo refrescado de paso."""
        payload = await self._request("GET", "/v1/models")
        self._catalog.update(payload.get("data") or [], now=time.monotonic())
        return payload.get("data") or []

    async def family_of(self, model: str) -> Family:
        """Familia del modelo: catálogo primero, prefijo como respaldo."""
        now = time.monotonic()
        if self._catalog.is_stale(now):
            try:
                await self.models()
            except Exception as exc:
                # Quedarse sin catálogo degrada la precisión, no la
                # disponibilidad: el respaldo por prefijo sigue respondiendo.
                log.warning("cliproxy.catalog_refresh_failed", error=str(exc))

        family = family_from_owned_by(self._catalog.owner_of(model))
        if family is Family.UNKNOWN:
            family = family_from_model_id(model)
        return family

    # ── Capacidades ───────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[translate.Message],
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
        payload = await self._request("POST", request.path, request.body)
        return translate.parse_chat(payload, model=model)

    async def search(
        self,
        messages: list[translate.Message],
        *,
        model: str,
        max_tokens: int = 4096,
    ) -> LLMResult:
        family = await self.family_of(model)
        request = translate.websearch_request(
            model=model, family=family, messages=messages, max_tokens=max_tokens
        )
        payload = await self._request("POST", request.path, request.body)
        return translate.parse_websearch(payload, model=model, family=family)

    async def image(
        self,
        prompt: str,
        *,
        model: str,
        size: str | None = None,
        quality: str | None = None,
    ) -> LLMResult:
        request = translate.image_request(model=model, prompt=prompt, size=size, quality=quality)
        payload = await self._request("POST", request.path, request.body)
        return translate.parse_image(payload, model=model)

    async def embed(self, texts: list[str], *, model: str):
        """CLIProxyAPI no expone embeddings — comprobado contra la instancia:
        `/v1/embeddings` devuelve vacío. Se declara con un error propio para que
        el routing pruebe el siguiente candidato, que será uno local."""
        from src.modules.backends.base import BackendCapabilityError

        raise BackendCapabilityError(
            f"CLIProxyAPI no expone embeddings (modelo {model!r}); usar un modelo local"
        )

    # ── Transporte ────────────────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        timeout = timeout_s if timeout_s is not None else httpx.USE_CLIENT_DEFAULT
        try:
            if method == "GET":
                response = await self._client.get(path, timeout=timeout)
            else:
                response = await self._client.post(path, json=body, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise CliproxyTransportError(f"timeout en {path}") from exc
        except httpx.HTTPError as exc:
            raise CliproxyTransportError(f"fallo de transporte en {path}: {exc}") from exc

        try:
            payload: Any = response.json()
        except ValueError:
            payload = {"error": {"message": response.text[:500]}}

        if response.status_code >= 400:
            raise classify(response.status_code, payload, path=path)
        if payload_has_error(payload):
            # 200 con error adentro: se clasifica con el código que venga en el
            # cuerpo, que es el que describe el fallo real.
            inner = payload.get("error")
            status = inner.get("code") if isinstance(inner, dict) else None
            raise classify(
                status if isinstance(status, int) else response.status_code,
                payload,
                path=path,
            )

        return payload if isinstance(payload, dict) else {"data": payload}
