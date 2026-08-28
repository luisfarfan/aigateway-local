"""
Traducción per-proveedor sobre CLIProxyAPI vanilla.

Esto es lo que en el fork de CLIProxyAPI son parches en Go. Vive acá, en Python,
por dos razones medidas (`docs/F0-VANILLA-CAPABILITIES.md`): el binario vanilla
ya expone todas las superficies que hacen falta, y mantener un fork de Go
sincronizado con upstream cuesta más que traducir desde este lado.

Las tres asimetrías que justifican el módulo:

  * **Websearch de Google** — el bloque `{"type": "web_search"}` sobre
    `/v1/chat/completions` lo descarta vanilla: el modelo responde "no tengo
    acceso a información en tiempo real". Hay que ir por la superficie nativa,
    `/v1beta/models/{model}:generateContent` con `tools: [{"googleSearch": {}}]`.
    De regalo llega `groundingMetadata`, que es de donde salen las fuentes — el
    path OpenAI-compatible las tira.
  * **Websearch de OpenAI** — endpoint distinto (`/v1/responses`), forma de
    respuesta distinta (`output[]` con `web_search_call` + `message`).
  * **Imagen de Gemini** — no sale por `/v1/images/generations` sino dentro del
    mensaje de chat, en `images[].image_url.url`, como data URI.

Hacia afuera el gateway sigue hablando OpenAI (`to_openai_chat_completion`); la
asimetría es interna.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.modules.providers.cliproxy.families import Family, is_gemini_model

Message = dict[str, Any]


@dataclass(frozen=True)
class Source:
    """Una fuente citada por el websearch."""

    uri: str
    title: str


@dataclass
class LLMResult:
    """Resultado normalizado, venga de la superficie que venga.

    `sources` e `images` existen acá arriba a propósito: son la parte que cada
    proveedor entrega en un sitio distinto, y el resto del gateway no debería
    tener que saber cuál.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sources: list[Source] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    searched: bool = False
    # Function calling. Un agente decide su siguiente paso con esto: perderlo
    # deja al cliente con un mensaje vacío y un bucle que no avanza.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Request:
    """Un request ya traducido, listo para mandar."""

    path: str
    body: dict[str, Any]


# ─── Chat ─────────────────────────────────────────────────────────────────────


def chat_request(
    *,
    model: str,
    messages: list[Message],
    max_tokens: int = 4096,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    stream: bool = False,
) -> Request:
    """Petición de chat. `tools` viaja tal cual: es function calling del cliente.

    No confundir con la búsqueda web, que también llega como un bloque de tool
    pero necesita traducción per-proveedor (`websearch_request`). Las
    herramientas de función se pasan verbatim porque su forma ya es la de
    OpenAI y el upstream la entiende: reescribirlas sólo podría estropearlas.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream:
        # Sin esto el último chunk no trae `usage` y la petición queda sin
        # contabilizar: streaming sería un agujero en el reporte de costos.
        body["stream_options"] = {"include_usage": True}
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if response_format is not None:
        # Los proveedores que no lo entienden lo descartan sin quejarse, así que
        # mandarlo siempre es gratis. El contrato de verdad viaja además dentro
        # de la conversación — ver `structured/guard.py`.
        body["response_format"] = response_format
    return Request(path="/v1/chat/completions", body=body)


def parse_chat(payload: dict[str, Any], *, model: str) -> LLMResult:
    """Lee la forma OpenAI, incluida la variante con bloques de contenido."""
    choices = payload.get("choices") or []
    message: dict[str, Any] = choices[0].get("message", {}) if choices else {}
    usage = payload.get("usage") or {}

    return LLMResult(
        text=_content_text(message.get("content")),
        model=payload.get("model") or model,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        images=_chat_images(message),
        tool_calls=[tc for tc in message.get("tool_calls") or [] if isinstance(tc, dict)],
    )


def _content_text(content: Any) -> str:
    """`content` es str en la forma clásica y lista de bloques en la nueva."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _chat_images(message: dict[str, Any]) -> list[str]:
    """Data URIs de `images[].image_url.url`, la vía de Gemini para imagen."""
    images: list[str] = []
    for item in message.get("images") or []:
        url = (item.get("image_url") or {}).get("url")
        if isinstance(url, str) and url:
            images.append(url)
    return images


# ─── Websearch ────────────────────────────────────────────────────────────────


def websearch_request(
    *, model: str, family: Family, messages: list[Message], max_tokens: int = 4096
) -> Request:
    """Enruta al websearch del proveedor. Tres formas incompatibles entre sí."""
    if family is Family.OPENAI:
        return _openai_websearch_request(model=model, messages=messages)
    if family is Family.GOOGLE or (family is Family.ANTIGRAVITY and is_gemini_model(model)):
        return _google_websearch_request(model=model, messages=messages)
    if family is Family.ANTHROPIC:
        return _anthropic_websearch_request(model=model, messages=messages, max_tokens=max_tokens)
    raise ValueError(f"No hay ruta de websearch conocida para el modelo {model!r} ({family})")


def _openai_websearch_request(*, model: str, messages: list[Message]) -> Request:
    """`/v1/responses`: el `system` va como `instructions`, el resto como `input`."""
    instructions: str | None = None
    turns: list[Message] = []
    for message in messages:
        if message.get("role") == "system":
            instructions = _content_text(message.get("content"))
        else:
            turns.append(message)

    # Un único turno de usuario se manda como string plano: la forma en array
    # también se acepta, pero el string es la que documenta OpenAI.
    single_turn = len(turns) == 1 and isinstance(turns[0].get("content"), str)
    body: dict[str, Any] = {
        "model": model,
        "input": turns[0]["content"] if single_turn else turns,
        "tools": [{"type": "web_search_preview"}],
    }
    if instructions:
        body["instructions"] = instructions
    return Request(path="/v1/responses", body=body)


def _google_websearch_request(*, model: str, messages: list[Message]) -> Request:
    """Superficie Gemini nativa. Es la única que hace grounding en vanilla."""
    contents: list[dict[str, Any]] = []
    system_parts: list[str] = []
    for message in messages:
        text = _content_text(message.get("content"))
        if message.get("role") == "system":
            system_parts.append(text)
            continue
        # Gemini sólo conoce los roles `user` y `model`.
        role = "model" if message.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": text}]})

    body: dict[str, Any] = {"contents": contents, "tools": [{"googleSearch": {}}]}
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    return Request(path=f"/v1beta/models/{model}:generateContent", body=body)


def _anthropic_websearch_request(
    *, model: str, messages: list[Message], max_tokens: int
) -> Request:
    """Bloque de tool de Anthropic sobre la superficie OpenAI-compatible.

    NO VERIFICADO: no hay cuenta Anthropic conectada, así que no hay fixture que
    fije esta forma. Los modelos Claude que sirve antigravity no valen para
    comprobarlo — llegan por Vertex (`req_vrtx_…`) y no ejercitan el header
    `anthropic-beta`. Tratar como provisional hasta que exista el fixture.
    """
    return Request(
        path="/v1/chat/completions",
        body={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        },
    )


def parse_websearch(payload: dict[str, Any], *, model: str, family: Family) -> LLMResult:
    if family is Family.OPENAI:
        return _parse_openai_responses(payload, model=model)
    if family is Family.GOOGLE or (family is Family.ANTIGRAVITY and is_gemini_model(model)):
        return _parse_google_generate_content(payload, model=model)
    return parse_chat(payload, model=model)


def _parse_openai_responses(payload: dict[str, Any], *, model: str) -> LLMResult:
    """La prueba de que buscó es el item `web_search_call`, no que el texto
    parezca actualizado.

    Las fuentes viven en `annotations` de cada bloque de texto, como
    `url_citation`. Pueden venir vacías aunque la búsqueda haya ocurrido: el
    modelo a veces cita en prosa sin emitir la anotación. Por eso `searched` y
    `sources` son señales distintas y no se deducen una de la otra.
    """
    output = payload.get("output") or []
    blocks = [
        block
        for item in output
        if item.get("type") == "message"
        for block in item.get("content") or []
        if block.get("type") == "output_text"
    ]

    text = "".join(block.get("text", "") for block in blocks)
    sources = [
        Source(uri=annotation["url"], title=annotation.get("title", ""))
        for block in blocks
        for annotation in block.get("annotations") or []
        if annotation.get("type") == "url_citation" and annotation.get("url")
    ]

    usage = payload.get("usage") or {}
    return LLMResult(
        text=text,
        model=payload.get("model") or model,
        prompt_tokens=int(usage.get("input_tokens") or 0),
        completion_tokens=int(usage.get("output_tokens") or 0),
        sources=_dedupe(sources),
        searched=any(item.get("type") == "web_search_call" for item in output),
    )


def _dedupe(sources: list[Source]) -> list[Source]:
    """Sin repetidos, conservando el orden en que el modelo las citó."""
    seen: set[str] = set()
    unique: list[Source] = []
    for source in sources:
        if source.uri not in seen:
            seen.add(source.uri)
            unique.append(source)
    return unique


def _parse_google_generate_content(payload: dict[str, Any], *, model: str) -> LLMResult:
    """`groundingChunks` es de donde salen las fuentes que el path
    OpenAI-compatible descarta."""
    candidates = payload.get("candidates") or []
    candidate: dict[str, Any] = candidates[0] if candidates else {}
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))

    grounding = candidate.get("groundingMetadata") or {}
    sources = [
        Source(uri=web["uri"], title=web.get("title", ""))
        for chunk in grounding.get("groundingChunks") or []
        if (web := chunk.get("web")) and web.get("uri")
    ]

    usage = payload.get("usageMetadata") or {}
    return LLMResult(
        text=text,
        model=payload.get("modelVersion") or model,
        prompt_tokens=int(usage.get("promptTokenCount") or 0),
        completion_tokens=int(usage.get("candidatesTokenCount") or 0),
        sources=sources,
        searched=bool(grounding.get("webSearchQueries")),
    )


# ─── Embeddings ───────────────────────────────────────────────────────────────


@dataclass
class EmbeddingResult:
    """Vectores y su uso. Separado de `LLMResult` porque no hay texto ni fuentes
    ni imágenes: forzarlo en la misma clase dejaría media docena de campos
    vacíos que alguien acabaría interpretando mal."""

    vectors: list[list[float]]
    model: str
    prompt_tokens: int = 0

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


def embeddings_request(*, model: str, texts: list[str]) -> Request:
    return Request(path="/v1/embeddings", body={"model": model, "input": texts})


def parse_embeddings(payload: dict[str, Any], *, model: str) -> EmbeddingResult:
    """Los vectores salen ordenados por `index`, no por orden de llegada.

    La API permite devolverlos desordenados, y un RAG que asocie el vector
    equivocado al texto equivocado falla de forma silenciosa y difícil de ver.
    """
    items = sorted(
        (item for item in payload.get("data") or [] if isinstance(item, dict)),
        key=lambda item: int(item.get("index", 0)),
    )
    usage = payload.get("usage") or {}
    return EmbeddingResult(
        vectors=[item.get("embedding") or [] for item in items],
        model=payload.get("model") or model,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
    )


# ─── Imagen ───────────────────────────────────────────────────────────────────


def image_request(
    *,
    model: str,
    prompt: str,
    size: str | None = None,
    quality: str | None = None,
) -> Request:
    """Gemini entrega la imagen por chat; el resto por `/v1/images/generations`."""
    if is_gemini_model(model):
        return Request(
            path="/v1/chat/completions",
            body={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
            },
        )

    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "response_format": "b64_json",
    }
    if size:
        body["size"] = size
    if quality:
        body["quality"] = quality
    return Request(path="/v1/images/generations", body=body)


def parse_image(payload: dict[str, Any], *, model: str) -> LLMResult:
    """Normaliza las dos vías a `images`, siempre como data URI.

    `/v1/images/generations` devuelve base64 pelado sin cabecera; se le antepone
    el prefijo para que quien consuma no tenga que preguntar de dónde vino.
    """
    if is_gemini_model(model):
        return parse_chat(payload, model=model)

    images: list[str] = []
    for item in payload.get("data") or []:
        if b64 := item.get("b64_json"):
            images.append(f"data:image/png;base64,{b64}")
        elif url := item.get("url"):
            images.append(url)

    usage = payload.get("usage") or {}
    return LLMResult(
        text="",
        model=payload.get("model") or model,
        prompt_tokens=int(usage.get("input_tokens") or 0),
        completion_tokens=int(usage.get("output_tokens") or 0),
        images=images,
    )


# ─── Salida ───────────────────────────────────────────────────────────────────


def to_openai_chat_completion(result: LLMResult) -> dict[str, Any]:
    """Forma OpenAI para el contrato externo del gateway.

    Las fuentes van en `proxima.sources`, fuera del objeto `message`: cualquier
    cliente que hable OpenAI ignora la clave extra, y el que sí la conoce no
    tiene que reparsear el texto para recuperarlas.
    """
    message: dict[str, Any] = {"role": "assistant", "content": result.text or None}
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
    if result.images:
        message["images"] = [
            {"type": "image_url", "image_url": {"url": url}, "index": i}
            for i, url in enumerate(result.images)
        ]

    payload: dict[str, Any] = {
        "object": "chat.completion",
        "model": result.model,
        # `finish_reason` no es decorativo: un cliente agéntico decide si
        # ejecutar herramientas o terminar mirando este campo.
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if result.tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
    }
    if result.sources or result.searched:
        payload["proxima"] = {
            "searched": result.searched,
            "sources": [{"uri": s.uri, "title": s.title} for s in result.sources],
        }
    return payload
