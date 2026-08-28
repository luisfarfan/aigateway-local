"""
Plano síncrono `/v1/*` — el contrato que hablan los consumidores.

Es deliberadamente OpenAI-compatible y vive en la raíz, no bajo `/api/v1`:
migrar a este gateway tiene que ser cambiar una `base_url` y nada más. El plano
asíncrono de jobs (`/api/v1/jobs`) sigue existiendo sin tocar; son dos puertas a
los mismos proveedores, no dos sistemas.

El valor de pasar por acá en vez de ir directo a CLIProxyAPI es que el cliente
manda **una sola forma** de pedir websearch y el gateway la traduce a la que
cada proveedor entiende — que son tres, incompatibles entre sí, y una de ellas
ni siquiera funciona por la superficie OpenAI (ver `translate.py`).
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.modules.auth.middleware import get_current_client_id
from src.modules.backends.base import BackendCapabilityError
from src.modules.backends.registry import BackendRegistry
from src.modules.cache import llm_cache
from src.modules.observability.recorder import AttemptRecord, Observation, observe
from src.modules.providers.cliproxy.errors import (
    CliproxyError,
)
from src.modules.providers.cliproxy.translate import to_openai_chat_completion
from src.modules.routing import errors as routing_errors
from src.modules.routing.breaker import CircuitBreaker
from src.modules.routing.config import RoutingTable, load_routing
from src.modules.routing.executor import NoCandidatesError, RouteResult, run_with_fallback
from src.modules.structured.guard import InvalidStructuredOutput, call_with_guard

log = structlog.get_logger(__name__)


async def _authenticate(request: Request, client_id: str = Depends(get_current_client_id)) -> str:
    """Valida la clave y deja el `client_id` donde los endpoints lo encuentren.

    Envuelve al middleware compartido en vez de modificarlo: el plano de jobs
    depende de él y no tiene por qué cambiar para que este lo reutilice.
    """
    request.state.client_id = client_id
    return client_id


# La autenticación se declara en el router y no ruta por ruta: así una ruta
# nueva nace protegida en vez de depender de que alguien se acuerde de añadirla.
# Con `API_KEYS` vacío el middleware deja pasar todo (modo abierto de
# desarrollo); en cuanto hay claves configuradas, exige Bearer.
router = APIRouter(tags=["OpenAI-compatible"], dependencies=[Depends(_authenticate)])

# Tipos de tool que un cliente puede mandar para pedir búsqueda web. Se aceptan
# las tres formas de proveedor porque los consumidores ya las escriben así; el
# gateway se encarga de mandar la correcta según el modelo que resuelva.
_WEBSEARCH_TOOL_TYPES = frozenset({"web_search", "web_search_preview", "web_search_20250305"})


class ChatCompletionRequest(BaseModel):
    """Subconjunto de la petición de OpenAI que el gateway entiende hoy.

    `extra` queda permitido a propósito: un cliente que mande campos que todavía
    no soportamos recibe una respuesta útil en vez de un 422, y el campo queda
    visible en los logs para decidir si merece implementarse.
    """

    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int = 4096
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    stream: bool = False
    # Forma estándar de OpenAI. Mandarla activa el guard: el gateway se encarga
    # de que el schema se cumpla también en los proveedores que descartan este
    # campo (Gemini, Claude), que es la mayoría.
    response_format: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class EmbeddingsRequest(BaseModel):
    """Forma de OpenAI. `input` acepta un texto o una lista."""

    model: str | None = None
    input: str | list[str]
    encoding_format: str = "float"

    model_config = {"extra": "allow"}


class ImageRequest(BaseModel):
    model: str | None = None
    prompt: str
    size: str | None = None
    quality: str | None = None
    response_format: str = "b64_json"

    model_config = {"extra": "allow"}


def _backends(request: Request) -> BackendRegistry:
    return request.app.state.backends


def _resolve(registry: BackendRegistry, model: str):
    """Backend que sirve el modelo, o error tratable como fallback.

    Un candidato cuyo backend no está configurado no es un fallo de la petición:
    es un candidato inservible, y el routing debe pasar al siguiente.
    """
    resolved = registry.resolve(model)
    if resolved is None:
        raise BackendCapabilityError(f"No hay backend configurado para {model!r}")
    return resolved


def _wants_websearch(tools: list[dict[str, Any]] | None) -> bool:
    """True sólo si TODOS los tools son de búsqueda web.

    Si el cliente además manda herramientas de función, la petición es de
    function calling y va por el camino de chat: la superficie nativa de
    búsqueda de Gemini no acepta funciones, así que enrutar ahí las perdería
    en silencio — que es exactamente el fallo que este método arregla.
    """
    entries = [tool or {} for tool in tools or []]
    if not entries:
        return False
    return all(entry.get("type") in _WEBSEARCH_TOOL_TYPES for entry in entries)


def _function_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Las herramientas que no son de búsqueda, para reenviarlas tal cual."""
    functions = [
        tool for tool in tools or [] if (tool or {}).get("type") not in _WEBSEARCH_TOOL_TYPES
    ]
    return functions or None


# Cómo se traduce cada fallo a HTTP. El código importa porque es lo que un
# cliente usa para decidir si reintentar, y confundir "ninguna credencial cubre
# este modelo" con "el upstream está caído" hace que reintente para siempre.
def _status_and_code(exc: Exception) -> tuple[int, str]:
    """El código HTTP y la etiqueta del fallo, desde la clasificación única.

    Vive en `routing/errors.py` para que lo que ve el cliente, lo que decide el
    fallback y lo que queda en las métricas no puedan divergir.
    """
    kind = routing_errors.kind_of(exc)
    return routing_errors.http_status_of(kind), kind


def _failure_fields(exc: Exception) -> dict[str, Any]:
    """Traduce la excepción a lo que el histórico necesita."""
    kind = routing_errors.kind_of(exc)
    return {
        "kind": kind,
        "message": getattr(exc, "message", str(exc)),
        "retryable": routing_errors.is_retryable(kind),
        "outcome": routing_errors.outcome_of(kind),
    }


def _error_response(exc: Exception) -> JSONResponse:
    status, code = _status_and_code(exc)

    message = getattr(exc, "message", str(exc))
    log.warning("cliproxy.request_failed", code=code, error=message)
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": code,
                # Un cliente sabe si vale la pena reintentar sin parsear el texto.
                "retryable": routing_errors.is_retryable(code),
            }
        },
    )


@router.get("/v1/models", summary="Modelos disponibles")
async def list_models(request: Request) -> Any:
    """Inventario de todos los backends, cloud y local.

    Aviso que vale repetir: esto lista lo que hay configurado, no lo que
    responde. Un modelo con la credencial revocada aparece igual. Quien sabe
    cuáles están vivos es el watchdog, que prueba.
    """
    registry = _backends(request)
    data: list[dict[str, Any]] = []
    try:
        data.extend(await registry.resolve("_").backend.models())
    except CliproxyError as exc:
        return _error_response(exc)

    if registry.local_available:
        try:
            local = registry.resolve("ollama/_")
            data.extend(await local.backend.models())
        except CliproxyError as exc:
            # Que el backend local no conteste no puede vaciar el inventario
            # cloud: se devuelve lo que hay y se registra el hueco.
            log.warning("backends.local_models_failed", error=exc.message)

    return {"object": "list", "data": data}


def _json_schema_of(response_format: dict[str, Any] | None) -> tuple[dict[str, Any], str] | None:
    """`(schema, nombre)` si la petición pide salida estructurada."""
    if not response_format or response_format.get("type") != "json_schema":
        return None
    spec = response_format.get("json_schema") or {}
    schema = spec.get("schema")
    if not isinstance(schema, dict):
        return None
    return schema, str(spec.get("name") or "Respuesta")


def _routing(request: Request) -> tuple[RoutingTable, CircuitBreaker]:
    """Tabla de rutas y breaker, respetando `X-Proxima-No-Fallback`.

    Para qué la cabecera: la cadena de `chat` termina en un modelo local
    pequeño, que como red de seguridad está bien para una respuesta suelta pero
    es mal candidato para un bucle agéntico — produciría muchos turnos de ruido
    caro en vez de fallar rápido. Un cliente que prefiere el error explícito lo
    pide con esta cabecera.

    Se implementa vaciando `fallback_on`, no la cadena: los circuitos abiertos
    se siguen respetando, que es protección y no elección de modelo.
    """
    table = load_routing()
    if request.headers.get("X-Proxima-No-Fallback"):
        table = replace(table, fallback_on=frozenset())
    return table, CircuitBreaker(table.breaker)


def _requested_model(
    request: Request, table: RoutingTable, route: str, model: str | None
) -> str | None:
    """Qué modelo encabeza la cadena.

    Si el cliente nombró uno, ese. Si no y la ruta tiene cadena, la cadena
    manda. Si tampoco hay cadena, el default de settings — así una ruta sin
    configurar sigue respondiendo en vez de fallar por config faltante.
    """
    if model:
        return model
    if table.candidates(route):
        return None
    return request.app.state.settings.cliproxy_default_model


def _annotate_routing(obs: Observation, result: RouteResult[Any]) -> None:
    """Deja el rastro del fallback en la observación.

    `requested_model` se queda con lo que se pidió y `response_model` con lo que
    contestó: si difieren, hubo fallback, y eso es justo lo que uno quiere poder
    consultar después.
    """
    obs.meta["routing"] = [{"model": a.model, "outcome": a.outcome} for a in result.attempts]
    if result.fell_back:
        obs.meta["fell_back_from"] = result.first_choice


def _client_id_of(request: Request) -> str | None:
    """Quién llama, derivado de la clave que usó.

    La cabecera `X-Proxima-Client` la escribe cualquiera, así que sirve para
    etiquetar pero no para atribuir. El middleware deja el id derivado de la API
    key en el contexto de la petición; ese es el que vale para el histórico.
    """
    return getattr(request.state, "client_id", None) or request.headers.get("X-Proxima-Client")


def _project_of(request: Request) -> str:
    return (
        request.headers.get("X-Proxima-Project") or request.app.state.settings.llm_default_project
    )


async def _structured(
    request: Request,
    body: ChatCompletionRequest,
    model: str,
    schema: dict[str, Any],
    name: str,
    obs: Observation,
) -> Any:
    """Salida estructurada con cache y guard.

    El cache va **antes** del guard a propósito: un acierto ahorra la llamada y,
    con ella, las reparaciones que esa llamada podría haber necesitado.
    """
    resolved = _resolve(_backends(request), model)
    settings = request.app.state.settings
    project = _project_of(request)

    use_cache = settings.llm_cache_enabled and not request.headers.get("X-Proxima-No-Cache")
    key = None
    family = await resolved.backend.family_of(resolved.model)
    obs.family = str(family)
    obs.meta["backend"] = resolved.backend.name
    if use_cache:
        key = llm_cache.derive_key(
            project=project,
            route="structured",
            family=str(family),
            payload={"messages": body.messages, "schema": schema, "max_tokens": body.max_tokens},
        )
        if (cached := await llm_cache.get(key)) is not None:
            cached["proxima"] = {**cached.get("proxima", {}), "cache": llm_cache.HIT}
            # Un acierto de cache no consumió tokens ni costó nada: se registra
            # con contadores en cero para que no infle el gasto del proyecto.
            obs.cache = llm_cache.HIT
            obs.succeeded(model=cached.get("model"))
            return cached

    async def chat(messages, *, model, max_tokens, response_format=None):
        result = await resolved.backend.chat(
            messages,
            model=resolved.model,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return result.text, result.prompt_tokens, result.completion_tokens

    try:
        guarded = await call_with_guard(
            chat,
            messages=body.messages,
            schema=schema,
            name=name,
            model=model,
            max_tokens=body.max_tokens,
        )
    except InvalidStructuredOutput as exc:
        obs.attempts = [
            AttemptRecord(
                a.number, a.outcome, a.duration_s, a.prompt_tokens, a.completion_tokens, a.error
            )
            for a in exc.attempts
        ]
        obs.prompt_tokens = sum(a.prompt_tokens for a in exc.attempts)
        obs.completion_tokens = sum(a.completion_tokens for a in exc.attempts)
        obs.failed(
            kind="invalid_structured_output",
            message=exc.message,
            retryable=False,
            outcome="invalid_output",
        )
        # 422 y no 502: el upstream respondió, lo que no cumple es el contenido.
        # Reintentar igual no cambia nada; hay que cambiar el prompt o el schema.
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": exc.message,
                    "type": "invalid_structured_output",
                    "retryable": False,
                    "attempts": [
                        {"number": a.number, "outcome": a.outcome, "error": a.error}
                        for a in exc.attempts
                    ],
                }
            },
        )

    obs.attempts = [
        AttemptRecord(
            a.number, a.outcome, a.duration_s, a.prompt_tokens, a.completion_tokens, a.error
        )
        for a in guarded.attempts
    ]
    obs.prompt_tokens = guarded.prompt_tokens
    obs.completion_tokens = guarded.completion_tokens
    obs.cache = llm_cache.MISS if use_cache else llm_cache.DISABLED
    obs.succeeded(model=guarded.model)

    payload = {
        "object": "chat.completion",
        "model": guarded.model,
        "choices": [
            {
                "index": 0,
                # Contenido como texto JSON, que es lo que hace OpenAI. El objeto
                # ya validado va aparte para no obligar a re-parsearlo.
                "message": {
                    "role": "assistant",
                    "content": json.dumps(guarded.value, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": guarded.prompt_tokens,
            "completion_tokens": guarded.completion_tokens,
            "total_tokens": guarded.prompt_tokens + guarded.completion_tokens,
        },
        "proxima": {
            "parsed": guarded.value,
            "repairs": guarded.repairs,
            "cache": llm_cache.MISS if use_cache else llm_cache.DISABLED,
        },
    }

    if key is not None:
        await llm_cache.set(key, payload, settings.llm_cache_ttl_s)
    return payload


def _usage_from_chunk(raw: bytes, state: dict[str, Any]) -> None:
    """Mira los chunks al pasar para contabilizar, sin tocarlos.

    El `usage` llega en el último chunk gracias a `stream_options.include_usage`.
    Sin esto, una petición en streaming no aparecería en el reporte de costos —
    un agujero que crece justo con los clientes que más consumen.
    """
    for line in raw.split(b"\n"):
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if model := chunk.get("model"):
            state["model"] = model
        if usage := chunk.get("usage"):
            state["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
            state["completion_tokens"] = int(usage.get("completion_tokens") or 0)


async def _stream_guarded(
    request: Request,
    body: ChatCompletionRequest,
    obs: Observation,
    structured: tuple[dict[str, Any], str] | None,
    websearch: bool,
) -> Any:
    """Rechaza lo que no se puede servir por SSE, y delega el resto.

    Las dos negativas se responden **antes** de abrir nada, así que aquí sí se
    puede usar `observe` de la forma normal.
    """
    if structured:
        # Pedir las dos cosas es contradictorio: no se puede validar contra un
        # schema lo que aún no terminó de llegar. Mejor un 400 claro que
        # ignorar una de las dos en silencio.
        async with observe(obs):
            obs.failed(
                kind="invalid_request",
                message="`stream` y `response_format` son incompatibles",
                retryable=False,
                outcome="invalid_request",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": (
                            "`stream` y `response_format` son incompatibles: la salida "
                            "estructurada se valida entera, no por trozos"
                        ),
                        "type": "invalid_request",
                        "retryable": False,
                    }
                },
            )

    if websearch:
        async with observe(obs):
            obs.failed(
                kind="unsupported_capability",
                message="la búsqueda web no se sirve en streaming",
                retryable=False,
                outcome="unsupported_capability",
            )
            return JSONResponse(
                status_code=501,
                content={
                    "error": {
                        "message": (
                            "la búsqueda web usa superficies que no emiten SSE; pídela sin `stream`"
                        ),
                        "type": "unsupported_capability",
                        "retryable": False,
                    }
                },
            )

    return await _stream_completions(request, body, obs)


async def _stream_completions(
    request: Request, body: ChatCompletionRequest, obs: Observation
) -> Any:
    """Streaming SSE, con fallback sólo hasta el primer byte.

    Tres cosas no se pueden hacer sobre un stream, y por eso este camino es más
    estrecho que el normal:

    - **Fallback a mitad no existe.** Una vez enviada la cabecera 200 y el
      primer chunk, cambiar de modelo produciría una respuesta cosida de dos
      modelos distintos. Se prueba la cadena mientras se abre el stream; a
      partir del primer byte, lo que salga sale.
    - **El guard no aplica.** No se puede validar contra un schema lo que aún
      no ha terminado de llegar; pedir las dos cosas se rechaza con 400 en vez
      de ignorar una en silencio.
    - **El cache no participa.** Guardar exigiría acumular la respuesta entera,
      que es lo contrario de lo que pidió el cliente.
    """
    registry = _backends(request)
    table, breaker = _routing(request)
    requested = _requested_model(request, table, "chat", body.model)
    candidates = table.candidates("chat", requested)
    tools = _function_tools(body.tools)

    # El `observe` entra en ESTE stack, no en el del endpoint: si se cerrara al
    # devolver la respuesta —que es lo que pasa con un `async with` en el
    # endpoint— se registraría la petición antes de que fluyera un solo byte, y
    # todo streaming quedaría con cero tokens y cero costo.
    stack = AsyncExitStack()
    await stack.enter_async_context(observe(obs))
    chunks = None
    served_by = None
    last_error: Exception | None = None

    for model in candidates:
        if await breaker.is_open(model):
            continue
        try:
            resolved = _resolve(registry, model)
            obs.family = str(await resolved.backend.family_of(resolved.model))
            obs.meta["backend"] = resolved.backend.name
            chunks = await stack.enter_async_context(
                resolved.backend.stream_chat(
                    body.messages,
                    model=resolved.model,
                    max_tokens=body.max_tokens,
                    tools=tools,
                    tool_choice=body.tool_choice,
                )
            )
            served_by = model
            break
        except Exception as exc:  # noqa: BLE001 — se clasifica abajo
            last_error = exc
            await breaker.record_failure(model)
            if not table.should_fallback(routing_errors.kind_of(exc)):
                break

    if chunks is None:
        await stack.aclose()
        if last_error is not None:
            obs.failed(**_failure_fields(last_error))
            return _error_response(last_error)
        obs.failed(
            kind="no_candidates",
            message="ningún modelo disponible para streaming",
            retryable=True,
            outcome="upstream_error",
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "ningún modelo disponible",
                    "type": "no_candidates",
                    "retryable": True,
                }
            },
        )

    await breaker.record_success(served_by)
    obs.meta["streaming"] = True
    if served_by != candidates[0]:
        obs.meta["fell_back_from"] = candidates[0]

    state: dict[str, Any] = {"model": served_by, "prompt_tokens": 0, "completion_tokens": 0}

    async def relay():
        try:
            async for raw in chunks:
                _usage_from_chunk(raw, state)
                yield raw
        finally:
            # El ORDEN importa: primero se llena la observación, después se
            # cierra el stack — que es quien la publica. Al revés se registra
            # una petición vacía y todo streaming aparece con cero tokens.
            #
            # Corre también si el cliente corta a mitad: un agente que abandona
            # el turno igual consumió tokens, y deben quedar contabilizados.
            obs.prompt_tokens = state["prompt_tokens"]
            obs.completion_tokens = state["completion_tokens"]
            obs.succeeded(model=state["model"])
            await stack.aclose()

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Proxima-Served-By": served_by,
        },
    )


@router.post("/v1/chat/completions", summary="Chat (con o sin búsqueda web)")
async def chat_completions(request: Request, body: ChatCompletionRequest) -> Any:
    registry = _backends(request)
    table, breaker = _routing(request)
    structured = _json_schema_of(body.response_format)
    websearch = _wants_websearch(body.tools)
    route = "structured" if structured else ("websearch" if websearch else "chat")
    requested = _requested_model(request, table, route, body.model)

    obs = Observation(
        project=_project_of(request),
        route=route,
        requested_model=requested or (table.candidates(route) or ["?"])[0],
        client_id=_client_id_of(request),
    )

    if body.stream:
        return await _stream_guarded(request, body, obs, structured, websearch)

    async with observe(obs):
        if structured:
            schema, name = structured

            async def attempt_structured(model: str) -> Any:
                return await _structured(request, body, model, schema, name, obs)

            call = attempt_structured
        else:

            async def attempt_chat(model: str) -> Any:
                resolved = _resolve(registry, model)
                obs.family = str(await resolved.backend.family_of(resolved.model))
                obs.meta["backend"] = resolved.backend.name
                if websearch:
                    return await resolved.backend.search(
                        body.messages, model=resolved.model, max_tokens=body.max_tokens
                    )
                return await resolved.backend.chat(
                    body.messages,
                    model=resolved.model,
                    max_tokens=body.max_tokens,
                    tools=_function_tools(body.tools),
                    tool_choice=body.tool_choice,
                )

            call = attempt_chat

        try:
            routed = await run_with_fallback(
                call, route=route, table=table, breaker=breaker, requested_model=requested
            )
        except NoCandidatesError as exc:
            obs.failed(
                kind="no_candidates",
                message=str(exc),
                retryable=True,
                outcome="upstream_error",
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {"message": str(exc), "type": "no_candidates", "retryable": True}
                },
            )
        except ValueError as exc:
            # `websearch_request` no conoce una ruta para ese modelo. Es un error
            # de la petición, no del upstream: reintentar igual no arregla nada.
            obs.failed(
                kind="invalid_request",
                message=str(exc),
                retryable=False,
                outcome="invalid_request",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": {"message": str(exc), "type": "invalid_request", "retryable": False}
                },
            )
        except Exception as exc:  # noqa: BLE001 — toda la cadena falló
            obs.failed(**_failure_fields(exc))
            # Deja constancia de que se agotó la cadena, no un modelo suelto.
            obs.meta["routing_exhausted"] = True
            return _error_response(exc)

        _annotate_routing(obs, routed)

        if structured:
            # `_structured` ya llenó la observación y devolvió el cuerpo listo.
            return routed.value

        result = routed.value
        obs.prompt_tokens = result.prompt_tokens
        obs.completion_tokens = result.completion_tokens
        obs.searched = result.searched
        obs.succeeded(model=result.model)

        payload = to_openai_chat_completion(result)
        if routed.fell_back:
            # El fallback nunca es silencioso: quien llamó pidió un modelo y le
            # respondió otro, y tiene derecho a enterarse sin leer los logs.
            payload["proxima"] = {
                **payload.get("proxima", {}),
                "fell_back_from": routed.first_choice,
                "served_by": routed.model,
            }
        return payload


@router.post("/v1/images/generations", summary="Generación de imagen")
async def images_generations(request: Request, body: ImageRequest) -> Any:
    """Devuelve siempre data URIs en `data[].url`.

    Las dos vías de arriba entregan cosas distintas — base64 pelado en un caso,
    data URI dentro del mensaje de chat en el otro — y normalizar acá evita que
    cada consumidor tenga que saber cuál le tocó.
    """
    registry = _backends(request)
    table, breaker = _routing(request)
    requested = _requested_model(request, table, "image", body.model)
    obs = Observation(
        project=_project_of(request),
        route="image",
        requested_model=requested or (table.candidates("image") or ["?"])[0],
        client_id=_client_id_of(request),
    )

    async with observe(obs):

        async def attempt(model: str) -> Any:
            resolved = _resolve(registry, model)
            obs.family = str(await resolved.backend.family_of(resolved.model))
            obs.meta["backend"] = resolved.backend.name
            return await resolved.backend.image(
                body.prompt, model=resolved.model, size=body.size, quality=body.quality
            )

        try:
            routed = await run_with_fallback(
                attempt, route="image", table=table, breaker=breaker, requested_model=requested
            )
        except NoCandidatesError as exc:
            obs.failed(
                kind="no_candidates", message=str(exc), retryable=True, outcome="upstream_error"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {"message": str(exc), "type": "no_candidates", "retryable": True}
                },
            )
        except Exception as exc:  # noqa: BLE001 — toda la cadena falló
            obs.failed(**_failure_fields(exc))
            obs.meta["routing_exhausted"] = True
            return _error_response(exc)

        _annotate_routing(obs, routed)
        result = routed.value

        obs.prompt_tokens = result.prompt_tokens
        obs.completion_tokens = result.completion_tokens
        # La imagen se tarifa por unidad, no por token.
        obs.image_count = len(result.images)
        obs.succeeded(model=result.model)

        return {
            "created": 0,
            "model": result.model,
            "data": [{"url": url} for url in result.images],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            },
        }


@router.post("/v1/embeddings", summary="Embeddings")
async def embeddings(request: Request, body: EmbeddingsRequest) -> Any:
    """Vectores para búsqueda semántica y RAG.

    Los sirve el backend local: CLIProxyAPI no expone esta superficie, y calcular
    embeddings en la máquina no consume cuota de ninguna suscripción — que
    importa cuando un solo reindexado son decenas de miles de llamadas.
    """
    registry = _backends(request)
    table, breaker = _routing(request)
    requested = _requested_model(request, table, "embeddings", body.model)
    texts = [body.input] if isinstance(body.input, str) else list(body.input)

    obs = Observation(
        project=_project_of(request),
        route="embeddings",
        requested_model=requested or (table.candidates("embeddings") or ["?"])[0],
        client_id=_client_id_of(request),
    )

    async with observe(obs):
        if not texts:
            obs.failed(
                kind="invalid_request",
                message="`input` no puede estar vacío",
                retryable=False,
                outcome="invalid_request",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "`input` no puede estar vacío",
                        "type": "invalid_request",
                        "retryable": False,
                    }
                },
            )

        async def attempt(model: str) -> Any:
            resolved = _resolve(registry, model)
            obs.family = str(await resolved.backend.family_of(resolved.model))
            obs.meta["backend"] = resolved.backend.name
            return await resolved.backend.embed(texts, model=resolved.model)

        try:
            routed = await run_with_fallback(
                attempt,
                route="embeddings",
                table=table,
                breaker=breaker,
                requested_model=requested,
            )
        except NoCandidatesError as exc:
            obs.failed(
                kind="no_candidates", message=str(exc), retryable=True, outcome="upstream_error"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {"message": str(exc), "type": "no_candidates", "retryable": True}
                },
            )
        except Exception as exc:  # noqa: BLE001 — toda la cadena falló
            obs.failed(**_failure_fields(exc))
            obs.meta["routing_exhausted"] = True
            return _error_response(exc)

        _annotate_routing(obs, routed)
        result = routed.value
        obs.prompt_tokens = result.prompt_tokens
        obs.succeeded(model=result.model)

        return {
            "object": "list",
            "model": result.model,
            "data": [
                {"object": "embedding", "index": i, "embedding": vector}
                for i, vector in enumerate(result.vectors)
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "total_tokens": result.prompt_tokens,
            },
        }
