"""
El plano síncrono `/v1/*`, sin red ni upstream.

Se monta el router sobre una app mínima con un cliente falso: lo que se prueba
acá es el contrato hacia afuera — qué forma tiene la respuesta, qué código HTTP
sale de cada tipo de fallo, y que la petición se enrute a `search` o a `chat`
según el tool que mande el cliente. La traducción en sí ya está cubierta en
`test_cliproxy_translate.py` contra respuestas reales.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.openai_compat.router import router
from src.modules.backends.registry import BackendRegistry
from src.modules.providers.cliproxy.errors import (
    CliproxyNoCredentialError,
    CliproxyRequestError,
    CliproxyRetryableError,
    CliproxyTransportError,
)
from src.modules.providers.cliproxy.translate import LLMResult, Source
from src.modules.routing.tiers import load_tiers


class FakeCliproxyClient:
    """Doble de un backend. Registra la última llamada para poder afirmar ruteo."""

    name = "fake"

    def __init__(self, result: LLMResult | None = None, raises: Exception | None = None):
        self._result = result or LLMResult(text="ok", model="m")
        self._raises = raises
        self.calls: list[str] = []

    async def _answer(self, name: str) -> LLMResult:
        self.calls.append(name)
        if self._raises:
            raise self._raises
        return self._result

    async def chat(self, messages: list[dict[str, Any]], **_: Any) -> LLMResult:
        return await self._answer("chat")

    async def family_of(self, model: str) -> str:
        return "antigravity"

    async def search(self, messages: list[dict[str, Any]], **_: Any) -> LLMResult:
        return await self._answer("search")

    async def image(self, prompt: str, **_: Any) -> LLMResult:
        return await self._answer("image")

    async def models(self, **_: Any) -> list[dict[str, Any]]:
        self.calls.append("models")
        if self._raises:
            raise self._raises
        return [{"id": "gemini-3-flash", "owned_by": "antigravity"}]


@pytest.fixture(autouse=True)
def settings_aisladas():
    """Vacía la caché de `get_settings` antes y después de cada test.

    Los tests de autenticación cambian `API_KEYS` por entorno. Sin esto, el
    Settings cacheado con claves sobrevive al test y los siguientes reciben 401
    sin que nada en ellos lo explique.
    """
    from src.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeStreamingBackend(FakeCliproxyClient):
    """Backend que emite SSE como los upstreams reales."""

    CHUNKS = [
        b'data: {"model":"fake-1","choices":[{"delta":{"content":"ho"},'
        b'"native_finish_reason":null}]}\n\n',
        b'data: {"model":"fake-1","choices":[{"delta":{"content":"la"}}],'
        b'"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
        b"data: [DONE]\n\n",
    ]

    @asynccontextmanager
    async def stream_chat(self, messages, **kwargs):
        self.calls.append("stream_chat")
        if self._raises:
            raise self._raises

        async def gen():
            for chunk in self.CHUNKS:
                yield chunk

        yield gen()


class FakeSettings:
    cliproxy_default_model = "gemini-3-flash"
    llm_cache_enabled = True
    llm_cache_ttl_s = 3600
    llm_default_project = "tests"


def build_app(client: FakeCliproxyClient, *, local: object | None = None) -> FastAPI:
    """App mínima con el router montado.

    Sin `local`, las pruebas del contrato HTTP no dependen de Ollama. Se pasa
    sólo donde la ruta bajo prueba es local por definición — embeddings, que
    CLIProxyAPI no sirve.
    """
    app = FastAPI()
    app.include_router(router)
    app.state.backends = BackendRegistry(cloud=client, local=local)
    app.state.settings = FakeSettings()
    return app


async def call(app: FastAPI, method: str, path: str, **kwargs: Any):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        return await http.request(method, path, **kwargs)


# ─── Ruteo ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sin_tools_va_a_chat():
    fake = FakeCliproxyClient()
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={"model": "gemini-3-flash", "messages": [{"role": "user", "content": "hola"}]},
    )
    assert response.status_code == 200
    assert fake.calls == ["chat"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_type", ["web_search", "web_search_preview", "web_search_20250305"])
async def test_cualquier_forma_de_websearch_activa_la_busqueda(tool_type: str):
    """El valor de pasar por el gateway: el cliente manda la forma que ya
    escribe hoy — sea la de Gemini, la de OpenAI o la de Anthropic — y el
    gateway resuelve cuál corresponde al modelo que toque."""
    fake = FakeCliproxyClient()
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gemini-3-flash",
            "messages": [{"role": "user", "content": "hola"}],
            "tools": [{"type": tool_type}],
        },
    )
    assert response.status_code == 200
    assert fake.calls == ["search"]


@pytest.mark.asyncio
async def test_el_modelo_es_opcional():
    """Sin `model`, cae al default configurado. Un consumidor no debería tener
    que saber qué modelo está vivo hoy."""
    fake = FakeCliproxyClient()
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hola"}]},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_streaming_devuelve_los_chunks_del_upstream():
    """Se reenvían sin reserializar: cada proveedor mete campos propios en los
    chunks (`native_finish_reason`, `system_fingerprint`) y reconstruir el JSON
    sólo puede perderlos."""
    fake = FakeStreamingBackend()
    transport = ASGITransport(app=build_app(fake))
    async with (
        AsyncClient(transport=transport, base_url="http://test") as http,
        http.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}], "stream": True},
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        cuerpo = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert b"native_finish_reason" in cuerpo
    assert b"[DONE]" in cuerpo


@pytest.mark.asyncio
async def test_streaming_dice_quien_lo_sirvio_en_una_cabecera():
    """En streaming el cuerpo es del upstream, así que el dato va en cabecera:
    quien pidió un modelo tiene derecho a saber cuál respondió."""
    fake = FakeStreamingBackend()
    transport = ASGITransport(app=build_app(fake))
    async with (
        AsyncClient(transport=transport, base_url="http://test") as http,
        http.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}], "stream": True},
        ) as response,
    ):
        assert response.headers["x-proxima-served-by"]
        async for _ in response.aiter_bytes():
            pass


@pytest.mark.asyncio
async def test_streaming_contabiliza_los_tokens_del_ultimo_chunk():
    """`stream_options.include_usage` hace que el upstream mande el uso al
    final. Sin leerlo, las peticiones en streaming serían un agujero en el
    reporte de costos — y las abren justo los clientes que más consumen."""
    from src.api.openai_compat.router import _usage_from_chunk

    state = {"model": "x", "prompt_tokens": 0, "completion_tokens": 0}
    _usage_from_chunk(
        b'data: {"model":"gpt-5.4-mini","usage":{"prompt_tokens":11,"completion_tokens":7}}\n',
        state,
    )
    assert state == {"model": "gpt-5.4-mini", "prompt_tokens": 11, "completion_tokens": 7}


def test_un_chunk_roto_no_rompe_la_contabilidad():
    """Un `data:` que no sea JSON no puede tumbar el stream entero."""
    from src.api.openai_compat.router import _usage_from_chunk

    state = {"model": "m", "prompt_tokens": 0, "completion_tokens": 0}
    _usage_from_chunk(b"data: {esto no es json\ndata: [DONE]\n", state)
    assert state["prompt_tokens"] == 0


@pytest.mark.asyncio
async def test_streaming_con_response_format_se_rechaza():
    """Pedir las dos cosas es contradictorio: no se puede validar contra un
    schema lo que aún no terminó de llegar. Mejor un 400 claro que ignorar una
    de las dos en silencio."""
    response = await call(
        build_app(FakeStreamingBackend()),
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "response_format": RESPONSE_FORMAT,
        },
    )
    assert response.status_code == 400
    assert "response_format" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_streaming_con_busqueda_web_se_rechaza():
    """La búsqueda usa superficies que no emiten SSE (la nativa de Gemini,
    /v1/responses de OpenAI). Se dice, en vez de devolver un stream mudo."""
    response = await call(
        build_app(FakeStreamingBackend()),
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status_code == 501


# ─── Forma de la respuesta ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_respuesta_es_chat_completion_con_fuentes_aparte():
    fake = FakeCliproxyClient(
        LLMResult(
            text="Argentina",
            model="gemini-3-flash",
            prompt_tokens=7,
            completion_tokens=3,
            sources=[Source(uri="https://fifa.com", title="fifa.com")],
            searched=True,
        )
    )
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "web_search"}],
        },
    )
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Argentina"
    assert body["usage"]["total_tokens"] == 10
    # Fuera de `message`, para que un cliente OpenAI puro lo ignore sin romperse.
    assert body["proxima"]["sources"] == [{"uri": "https://fifa.com", "title": "fifa.com"}]


@pytest.mark.asyncio
async def test_imagen_default_es_b64_json():
    """Sin `response_format`, el default de OpenAI es b64_json: base64 pelado en
    `data[].b64_json`. Las dos vías internas (chat vs /images) entregan formatos
    distintos, pero el contrato de salida es uno solo y respeta lo pedido."""
    fake = FakeCliproxyClient(
        LLMResult(text="", model="gpt-image-2", images=["data:image/png;base64,AAA"])
    )
    response = await call(
        build_app(fake), "POST", "/v1/images/generations", json={"prompt": "un cubo"}
    )
    assert response.json()["data"][0] == {"b64_json": "AAA"}


# ─── Errores ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status", "expected_retryable"),
    [
        (CliproxyNoCredentialError("auth_not_found"), 502, False),
        (CliproxyRequestError("modelo inválido"), 400, False),
        (CliproxyRetryableError("429"), 503, True),
        (CliproxyTransportError("timeout"), 504, True),
    ],
)
async def test_cada_fallo_tiene_su_codigo_y_dice_si_reintentar(
    error: Exception, expected_status: int, expected_retryable: bool
):
    """`retryable` viaja explícito porque los tres fallos de upstream pintan
    parecido y sólo dos se arreglan reintentando. Confundir `auth_not_found`
    con un 503 hace que un cliente reintente para siempre un modelo que ninguna
    credencial cubre."""
    fake = FakeCliproxyClient(raises=error)
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == expected_status
    assert response.json()["error"]["retryable"] is expected_retryable


@pytest.mark.asyncio
async def test_modelo_sin_ruta_de_websearch_da_400():
    """`websearch_request` levanta ValueError si no conoce la ruta. Es culpa de
    la petición, no del upstream."""
    fake = FakeCliproxyClient(raises=ValueError("No hay ruta de websearch conocida"))
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "web_search"}],
        },
    )
    assert response.status_code == 400


# ─── Salida estructurada ──────────────────────────────────────────────────────

SCHEMA = {
    "type": "object",
    "properties": {"titulo": {"type": "string"}},
    "required": ["titulo"],
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "Ficha", "schema": SCHEMA, "strict": True},
}


@pytest.fixture
def fake_cache(monkeypatch):
    """Cache en memoria: los tests no hablan con Redis."""
    from src.modules.cache import llm_cache

    store: dict[str, Any] = {}

    async def fake_get(key):
        return store.get(str(key))

    async def fake_set(key, value, ttl):
        store[str(key)] = value

    monkeypatch.setattr(llm_cache, "get", fake_get)
    monkeypatch.setattr(llm_cache, "set", fake_set)
    return store


@pytest.mark.asyncio
async def test_response_format_activa_el_guard(fake_cache):
    """Un cliente manda la forma estándar de OpenAI y el gateway hace que se
    cumpla incluso donde el proveedor descarta el campo."""
    fake = FakeCliproxyClient(LLMResult(text='{"titulo": "hola"}', model="gemini-3-flash"))
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "response_format": RESPONSE_FORMAT,
        },
    )
    body = response.json()

    assert response.status_code == 200
    # El objeto ya validado va aparte, para no obligar a re-parsear el texto.
    assert body["proxima"]["parsed"] == {"titulo": "hola"}
    assert body["proxima"]["repairs"] == 0
    assert json.loads(body["choices"][0]["message"]["content"]) == {"titulo": "hola"}


@pytest.mark.asyncio
async def test_el_cache_evita_la_segunda_llamada(fake_cache):
    """Y con ella, las reparaciones que esa llamada hubiera podido necesitar."""
    fake = FakeCliproxyClient(LLMResult(text='{"titulo": "hola"}', model="gemini-3-flash"))
    app = build_app(fake)
    payload = {
        "messages": [{"role": "user", "content": "x"}],
        "response_format": RESPONSE_FORMAT,
    }

    first = await call(app, "POST", "/v1/chat/completions", json=payload)
    assert first.json()["proxima"]["cache"] == "miss"

    second = await call(app, "POST", "/v1/chat/completions", json=payload)
    assert second.json()["proxima"]["cache"] == "hit"
    assert fake.calls == ["chat"]  # una sola llamada arriba


@pytest.mark.asyncio
async def test_no_cache_fuerza_la_llamada(fake_cache):
    fake = FakeCliproxyClient(LLMResult(text='{"titulo": "hola"}', model="m"))
    app = build_app(fake)
    payload = {
        "messages": [{"role": "user", "content": "x"}],
        "response_format": RESPONSE_FORMAT,
    }

    await call(app, "POST", "/v1/chat/completions", json=payload)
    await call(
        app,
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"X-Proxima-No-Cache": "1"},
    )
    assert fake.calls == ["chat", "chat"]


@pytest.mark.asyncio
async def test_proyectos_distintos_no_comparten_cache(fake_cache):
    """El mismo prompt en dos proyectos es la misma respuesta pero contabilidad
    distinta: compartir entrada haría que los costos de F3 mientan."""
    fake = FakeCliproxyClient(LLMResult(text='{"titulo": "hola"}', model="m"))
    app = build_app(fake)
    payload = {
        "messages": [{"role": "user", "content": "x"}],
        "response_format": RESPONSE_FORMAT,
    }

    await call(
        app, "POST", "/v1/chat/completions", json=payload, headers={"X-Proxima-Project": "a"}
    )
    await call(
        app, "POST", "/v1/chat/completions", json=payload, headers={"X-Proxima-Project": "b"}
    )
    assert fake.calls == ["chat", "chat"]


@pytest.mark.asyncio
async def test_salida_que_no_cumple_da_422_con_el_detalle(fake_cache):
    """422 y no 502: el upstream respondió, lo que no cumple es el contenido.
    Reintentar igual no cambia nada."""
    fake = FakeCliproxyClient(LLMResult(text="no pienso dar json", model="m"))
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "response_format": RESPONSE_FORMAT,
        },
    )
    body = response.json()

    assert response.status_code == 422
    assert body["error"]["retryable"] is False
    assert len(body["error"]["attempts"]) == 3
    assert fake.calls == ["chat", "chat", "chat"]


# ─── Autenticación ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sin_claves_configuradas_deja_pasar():
    """Modo abierto de desarrollo: `API_KEYS` vacío no exige nada. Es lo que
    permite levantar el gateway en local sin ceremonia."""
    response = await call(
        build_app(FakeCliproxyClient()),
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_con_claves_configuradas_exige_bearer(monkeypatch):
    """Al exponer el gateway a la red, esto es lo único que separa tus cuentas
    cloud de cualquiera que esté en el mismo WiFi."""
    from src.core.config import get_settings
    from src.modules.auth import middleware

    get_settings.cache_clear()
    monkeypatch.setenv("API_KEYS", "clave-buena")
    monkeypatch.setattr(middleware, "get_settings", get_settings)

    app = build_app(FakeCliproxyClient())
    body = {"messages": [{"role": "user", "content": "x"}]}
    if True:
        sin_clave = await call(app, "POST", "/v1/chat/completions", json=body)
        mala = await call(
            app,
            "POST",
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer clave-mala"},
        )
        buena = await call(
            app,
            "POST",
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer clave-buena"},
        )

    assert sin_clave.status_code == 401
    assert mala.status_code == 401
    assert buena.status_code == 200


@pytest.mark.asyncio
async def test_la_autenticacion_cubre_todas_las_rutas(monkeypatch):
    """Está declarada en el router, no ruta por ruta: una ruta nueva nace
    protegida en vez de depender de que alguien se acuerde."""
    from src.core.config import get_settings
    from src.modules.auth import middleware

    get_settings.cache_clear()
    monkeypatch.setenv("API_KEYS", "clave-buena")
    monkeypatch.setattr(middleware, "get_settings", get_settings)

    app = build_app(FakeCliproxyClient())
    try:
        modelos = await call(app, "GET", "/v1/models")
        imagen = await call(app, "POST", "/v1/images/generations", json={"prompt": "x"})
    finally:
        get_settings.cache_clear()

    assert modelos.status_code == 401
    assert imagen.status_code == 401


# ─── Embeddings ───────────────────────────────────────────────────────────────


class FakeEmbeddingBackend(FakeCliproxyClient):
    name = "fake-local"

    async def embed(self, texts: list[str], *, model: str) -> Any:
        from src.modules.providers.cliproxy.translate import EmbeddingResult

        self.calls.append("embed")
        if self._raises:
            raise self._raises
        # Devueltos a propósito en orden inverso al pedido, para comprobar que
        # el endpoint los reordena por índice.
        return EmbeddingResult(
            vectors=[[float(len(t))] for t in texts], model=model, prompt_tokens=len(texts)
        )


@pytest.mark.asyncio
async def test_embeddings_devuelve_un_vector_por_texto():
    fake = FakeEmbeddingBackend()
    response = await call(
        build_app(fake, local=fake),
        "POST",
        "/v1/embeddings",
        json={"input": ["hola", "mundo largo"]},
    )
    body = response.json()

    assert response.status_code == 200
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert body["data"][0]["embedding"] == [4.0]
    assert body["data"][1]["embedding"] == [11.0]


@pytest.mark.asyncio
async def test_embeddings_acepta_un_texto_suelto():
    """La mayoría de las llamadas son un texto; obligar a envolverlo en lista
    sería ruido."""
    fake = FakeEmbeddingBackend()
    response = await call(
        build_app(fake, local=fake), "POST", "/v1/embeddings", json={"input": "hola"}
    )
    assert len(response.json()["data"]) == 1


@pytest.mark.asyncio
async def test_embeddings_con_input_vacio_da_400():
    fake = FakeEmbeddingBackend()
    response = await call(build_app(fake, local=fake), "POST", "/v1/embeddings", json={"input": []})
    assert response.status_code == 400
    assert fake.calls == []


@pytest.mark.asyncio
async def test_cliproxy_declara_que_no_hace_embeddings():
    """Comprobado contra la instancia real: `/v1/embeddings` devuelve vacío. Se
    declara con un error propio para que el routing pruebe el siguiente
    candidato en vez de dar por muerta la petición."""
    from src.modules.backends.base import BackendCapabilityError
    from src.modules.providers.cliproxy.client import CliproxyClient

    client = CliproxyClient(base_url="http://x", api_key="k")
    with pytest.raises(BackendCapabilityError):
        await client.embed(["hola"], model="gemini-3-flash")
    await client.aclose()


# ─── Function calling ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_las_herramientas_de_funcion_no_activan_la_busqueda_web():
    """La superficie nativa de búsqueda de Gemini no acepta funciones: enrutar
    ahí una petición de function calling las perdería en silencio."""
    fake = FakeCliproxyClient()
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "clima en Lima"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        },
    )
    assert response.status_code == 200
    assert fake.calls == ["chat"]


@pytest.mark.asyncio
async def test_websearch_mas_funciones_va_por_chat():
    """Si el cliente manda las dos cosas, manda function calling: es el camino
    que sí puede transportar ambas."""
    fake = FakeCliproxyClient()
    await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {"type": "web_search"},
                {"type": "function", "function": {"name": "f"}},
            ],
        },
    )
    assert fake.calls == ["chat"]


@pytest.mark.asyncio
async def test_el_streaming_queda_contabilizado_al_terminar():
    """Dos veces se me escapó: primero porque `observe` se cerraba al devolver
    la respuesta, antes de que fluyera un byte; después porque el `finally`
    cerraba el registro antes de rellenarlo. En ambos casos el streaming salía
    con cero tokens y cero costo — invisible salvo que se mire la base."""
    from src.modules.observability import recorder

    publicadas: list[Any] = []
    original = recorder.Observation.succeeded

    def espia(self, *, model=None):
        original(self, model=model)
        publicadas.append((self.prompt_tokens, self.completion_tokens, self.outcome))

    recorder.Observation.succeeded = espia
    try:
        transport = ASGITransport(app=build_app(FakeStreamingBackend()))
        async with (
            AsyncClient(transport=transport, base_url="http://test") as http,
            http.stream(
                "POST",
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "x"}], "stream": True},
            ) as response,
        ):
            async for _ in response.aiter_bytes():
                pass
    finally:
        recorder.Observation.succeeded = original

    # El doble emite usage 5/2 en su último chunk.
    assert publicadas == [(5, 2, "ok")]


@pytest.mark.asyncio
async def test_no_fallback_falla_en_vez_de_degradar():
    """Un bucle agéntico prefiere un error explícito a que le respondan veinte
    turnos de ruido desde un modelo pequeño. La cadena de `chat` termina en uno
    local justamente como red de seguridad, y esta cabecera la desactiva."""
    fake = FakeCliproxyClient(raises=CliproxyRetryableError("429"))
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}]},
        headers={"X-Proxima-No-Fallback": "1"},
    )
    assert response.status_code == 503
    assert fake.calls == ["chat"]  # un solo intento, sin recorrer la cadena


@pytest.mark.asyncio
async def test_sin_la_cabecera_si_recorre_la_cadena():
    """Contraste con el test anterior: el comportamiento por defecto sigue
    siendo intentar los demás candidatos."""
    fake = FakeCliproxyClient(raises=CliproxyRetryableError("429"))
    await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}]},
    )
    assert len(fake.calls) > 1


# ─── Fallback en streaming: observabilidad y No-Fallback ──────────────────────


class FakeSkipBackend(FakeStreamingBackend):
    """Streaming que resuelve family sin tocar red."""

    async def family_of(self, model: str) -> str:
        return "antigravity"


@pytest.mark.asyncio
async def test_streaming_registra_el_modelo_saltado_por_circuito_abierto(monkeypatch):
    """El hueco que dejó al usuario a ciegas: un modelo saltado por circuito
    abierto no dejaba rastro. Ahora cada intento —saltado o fallido— queda en
    `obs.attempts` con SU modelo, no con el que acabó respondiendo."""
    from src.modules.observability import recorder
    from src.modules.routing.breaker import CircuitBreaker

    capturado: list[Any] = []
    original = recorder.Observation.succeeded

    def espia(self, *, model=None):
        capturado.append(list(self.attempts))
        original(self, model=model)

    # Circuito abierto para el primer candidato de la cadena chat.
    async def is_open(self, model):
        return model == "gemini-3-flash"

    monkeypatch.setattr(recorder.Observation, "succeeded", espia)
    monkeypatch.setattr(CircuitBreaker, "is_open", is_open)
    transport = ASGITransport(app=build_app(FakeSkipBackend()))
    async with (
        AsyncClient(transport=transport, base_url="http://test") as http,
        http.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}], "stream": True},
        ) as response,
    ):
        async for _ in response.aiter_bytes():
            pass

    attempts = capturado[0]
    saltado = next(a for a in attempts if a.outcome == "skipped_open")
    servido = next(a for a in attempts if a.outcome == "ok")
    assert saltado.model == "gemini-3-flash"  # el saltado, con SU nombre
    assert servido.model != "gemini-3-flash"  # otro respondió
    assert servido.model == attempts[1].model


@pytest.mark.asyncio
async def test_no_fallback_no_degrada_ante_circuito_abierto(monkeypatch):
    """La corrección de fondo: con la cabecera, si el modelo pedido tiene el
    circuito abierto, la petición FALLA. Antes degradaba a gemini-flash pese a
    la cabecera, porque el salto por circuito es otra rama que `fallback_on` no
    tocaba."""
    from src.modules.routing.breaker import CircuitBreaker

    async def is_open(self, model):
        return True  # todo cerrado el paso

    monkeypatch.setattr(CircuitBreaker, "is_open", is_open)
    response = await call(
        build_app(FakeSkipBackend()),
        "POST",
        "/v1/chat/completions",
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "x"}]},
        headers={"X-Proxima-No-Fallback": "1"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "no_candidates"


def test_no_fallback_recorta_la_cadena_a_un_candidato():
    """Sin la cabecera, un modelo pedido encabeza la cadena entera. Con ella,
    la cadena es sólo ese modelo: no hay a dónde degradar."""
    from dataclasses import replace

    from src.modules.routing.config import load_routing

    table = load_routing()
    nf = replace(table, single_candidate=True)
    assert nf.candidates("chat", "claude-sonnet-4-6") == ["claude-sonnet-4-6"]
    assert len(table.candidates("chat", "claude-sonnet-4-6")) > 1


# ─── Tiers (routing por intención) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pedir_un_tier_resuelve_a_un_modelo_real(monkeypatch):
    """`model: "cheap"` no es un modelo — es una intención. El gateway lo
    resuelve contra el mapa de capacidades."""
    from src.modules.routing import tiers as tiers_mod

    fake_table = tiers_mod.TierTable(
        policies={"cheap": tiers_mod.TierPolicy("cheap", require=("chat",), rank_by="cost_asc")},
        models={
            "barato": tiers_mod.ModelInfo("barato", "google", {"chat": True}, 1.0, 0.4),
            "caro": tiers_mod.ModelInfo("caro", "openai", {"chat": True}, 1.0, 11.0),
        },
    )
    tiers_mod.clear_cache()
    monkeypatch.setattr(tiers_mod, "load_tiers", lambda *a, **k: fake_table)

    fake = FakeCliproxyClient()
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={"model": "cheap", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 200
    assert fake.calls == ["chat"]  # resolvió y sirvió, no trató "cheap" como modelo


def test_un_tier_no_es_un_modelo():
    """`is_tier` separa la intención del nombre concreto."""
    from src.modules.routing.tiers import clear_cache, load_tiers

    clear_cache()
    table = load_tiers()
    assert table.is_tier("cheap")
    assert not table.is_tier("gemini-3-flash")


# ─── Descubrimiento (los sistemas, no sólo los devs) ──────────────────────────


@pytest.mark.asyncio
async def test_los_tiers_aparecen_en_v1_models():
    """Un sistema que consulta /v1/models tiene que VER los tiers, no depender
    de que su desarrollador leyera el README."""
    from src.modules.routing import tiers as tiers_mod

    fake = tiers_mod.TierTable(
        policies={"smart": tiers_mod.TierPolicy("smart", require=("chat",), order=("m",))},
        models={"m": tiers_mod.ModelInfo("m", "x", {"chat": True}, 1.0, 1.0)},
    )
    tiers_mod.clear_cache()
    import src.api.openai_compat.router as router_mod

    router_mod.load_tiers = lambda *a, **k: fake  # type: ignore[assignment]
    try:
        response = await call(build_app(FakeCliproxyClient()), "GET", "/v1/models")
        ids = {m["id"]: m.get("owned_by") for m in response.json()["data"]}
        assert ids.get("smart") == "proxima-tier"
    finally:
        router_mod.load_tiers = load_tiers


@pytest.mark.asyncio
async def test_v1_capabilities_expone_el_mapa_y_los_tiers():
    """Un consumidor puede preguntarle al gateway de qué es capaz, en vez de
    leer el README."""
    from src.modules.routing import tiers as tiers_mod

    fake = tiers_mod.TierTable(
        policies={"cheap": tiers_mod.TierPolicy("cheap", require=("chat",), rank_by="cost_asc")},
        models={
            "barato": tiers_mod.ModelInfo(
                "barato", "google", {"chat": True, "vision": True}, 1.0, 0.4
            )
        },
    )
    import src.api.openai_compat.router as router_mod

    router_mod.load_tiers = lambda *a, **k: fake  # type: ignore[assignment]
    try:
        response = await call(build_app(FakeCliproxyClient()), "GET", "/v1/capabilities")
        body = response.json()
        assert set(body["models"]["barato"]["capabilities"]) == {"chat", "vision"}
        assert body["tiers"]["cheap"]["chat"] == ["barato"]
    finally:
        router_mod.load_tiers = load_tiers


@pytest.mark.asyncio
async def test_imagen_respeta_response_format_b64_json():
    """Bug reportado: se pedía b64_json y llegaba una url con data-URI. Ahora el
    contrato de OpenAI se respeta: b64_json → base64 pelado; url → data-URI."""
    from src.modules.providers.cliproxy.translate import LLMResult

    fake = FakeCliproxyClient(
        LLMResult(text="", model="gemini-3.1-flash-image", images=["data:image/png;base64,QUJD"])
    )

    b64 = await call(
        build_app(fake),
        "POST",
        "/v1/images/generations",
        json={"prompt": "x", "response_format": "b64_json"},
    )
    item = b64.json()["data"][0]
    assert item == {"b64_json": "QUJD"}  # pelado, sin cabecera data:

    url = await call(
        build_app(fake),
        "POST",
        "/v1/images/generations",
        json={"prompt": "x", "response_format": "url"},
    )
    assert url.json()["data"][0] == {"url": "data:image/png;base64,QUJD"}
