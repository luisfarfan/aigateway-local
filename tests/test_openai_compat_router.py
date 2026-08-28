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
async def test_streaming_se_rechaza_explicitamente():
    """Mejor un 400 claro que un 200 que no hace streaming."""
    fake = FakeCliproxyClient()
    response = await call(
        build_app(fake),
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}], "stream": True},
    )
    assert response.status_code == 400
    assert response.json()["error"]["retryable"] is False
    assert fake.calls == []


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
async def test_imagen_sale_siempre_como_data_uri():
    """Las dos vías de arriba entregan formatos distintos; el contrato de
    salida es uno solo."""
    fake = FakeCliproxyClient(
        LLMResult(text="", model="gpt-image-2", images=["data:image/png;base64,AAA"])
    )
    response = await call(
        build_app(fake), "POST", "/v1/images/generations", json={"prompt": "un cubo"}
    )
    body = response.json()
    assert body["data"][0]["url"].startswith("data:image/")


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
