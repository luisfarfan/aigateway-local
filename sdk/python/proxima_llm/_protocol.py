"""
Construcción de peticiones y lectura de respuestas, compartido por las dos
superficies.

Sync y async se diferencian sólo en cómo mandan el HTTP. Todo lo demás —qué
cuerpo se arma, cómo se lee la respuesta, cómo se traduce un error— vive acá,
para que las dos no puedan divergir. Una diferencia de comportamiento entre
`Gateway` y `SyncGateway` sería un bug imposible de justificar.
"""

from __future__ import annotations

from typing import Any

from proxima_llm.errors import ProximaError
from proxima_llm.types import Completion, Embeddings, Image, Source

Message = dict[str, Any]

# Forma del bloque de tool que activa la búsqueda web. El gateway acepta las
# tres variantes de proveedor y traduce a la que corresponda al modelo que
# resuelva, así que desde acá basta con una.
WEBSEARCH_TOOL = {"type": "web_search"}


def as_messages(prompt_or_messages: str | list[Message]) -> list[Message]:
    """Acepta un string suelto o la lista completa.

    El atajo existe porque la mayoría de las llamadas son un solo turno de
    usuario, y obligar a escribir el dict entero para eso es ruido.
    """
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]
    return prompt_or_messages


def chat_body(
    messages: list[Message],
    *,
    model: str | None,
    max_tokens: int,
    websearch: bool = False,
    schema: dict[str, Any] | None = None,
    schema_name: str = "Respuesta",
) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
    if model:
        body["model"] = model
    if websearch:
        body["tools"] = [dict(WEBSEARCH_TOOL)]
    if schema is not None:
        # Forma estándar de OpenAI. El gateway la hace cumplir también en los
        # proveedores que descartan este campo, que son la mayoría.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }
    return body


def image_body(
    prompt: str, *, model: str | None, size: str | None, quality: str | None
) -> dict[str, Any]:
    body: dict[str, Any] = {"prompt": prompt}
    for key, value in (("model", model), ("size", size), ("quality", quality)):
        if value:
            body[key] = value
    return body


def raise_for_error(status: int, payload: Any) -> None:
    """Convierte la respuesta de error del gateway en `ProximaError`."""
    if status < 400:
        return

    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        raise ProximaError(f"HTTP {status}", kind="unknown", status=status)

    raise ProximaError(
        str(error.get("message") or f"HTTP {status}"),
        kind=str(error.get("type") or "unknown"),
        status=status,
        # Lo dice el gateway, no se deduce del status.
        retryable=bool(error.get("retryable", False)),
        attempts=error.get("attempts") or [],
    )


def read_completion(payload: dict[str, Any]) -> Completion:
    choices = payload.get("choices") or []
    message: dict[str, Any] = choices[0].get("message", {}) if choices else {}
    usage = payload.get("usage") or {}
    extra = payload.get("proxima") or {}

    return Completion(
        text=message.get("content") or "",
        model=payload.get("model") or "",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        sources=[
            Source(uri=s.get("uri", ""), title=s.get("title", ""))
            for s in extra.get("sources") or []
        ],
        images=[
            Image(url=item["image_url"]["url"])
            for item in message.get("images") or []
            if isinstance(item, dict) and item.get("image_url", {}).get("url")
        ],
        searched=bool(extra.get("searched", False)),
        cache=str(extra.get("cache", "disabled")),
        fell_back_from=extra.get("fell_back_from"),
        parsed=extra.get("parsed"),
        raw=payload,
    )


def read_images(payload: dict[str, Any]) -> Completion:
    usage = payload.get("usage") or {}
    return Completion(
        text="",
        model=payload.get("model") or "",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        images=[Image(url=item["url"]) for item in payload.get("data") or [] if item.get("url")],
        raw=payload,
    )


def embeddings_body(texts: list[str], *, model: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"input": texts}
    if model:
        body["model"] = model
    return body


def read_embeddings(payload: dict[str, Any]) -> Embeddings:
    """Ordena por `index` antes de devolver.

    La API permite entregarlos desordenados, y un RAG que asocie el vector
    equivocado al texto equivocado falla en silencio.
    """
    items = sorted(payload.get("data") or [], key=lambda item: int(item.get("index", 0)))
    usage = payload.get("usage") or {}
    return Embeddings(
        vectors=[item.get("embedding") or [] for item in items],
        model=payload.get("model") or "",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        raw=payload,
    )
