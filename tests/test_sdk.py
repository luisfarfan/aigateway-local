"""
El SDK, contra el gateway real montado en proceso.

No se mockea el SDK: se lo apunta al router de verdad con un backend falso
detrás. Lo que se verifica es que **las dos superficies dan lo mismo** — un
comportamiento distinto entre `Gateway` y `SyncGateway` sería un bug imposible
de justificar — y que el contrato HTTP y el del SDK no se despegan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from src.api.openai_compat.router import router
from src.modules.backends.registry import BackendRegistry
from src.modules.providers.cliproxy.errors import CliproxyRetryableError
from src.modules.providers.cliproxy.translate import LLMResult, Source

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk" / "python"))

from proxima_llm import Gateway, ProximaError  # noqa: E402
from proxima_llm import _protocol as protocol  # noqa: E402


class FakeBackend:
    name = "fake"

    def __init__(self, result: LLMResult | None = None, raises: Exception | None = None):
        self._result = result or LLMResult(text="hola", model="gemini-3-flash")
        self._raises = raises
        self.bodies: list[dict[str, Any]] = []

    async def _answer(self) -> LLMResult:
        if self._raises:
            raise self._raises
        return self._result

    async def chat(self, messages: Any, **kwargs: Any) -> LLMResult:
        self.bodies.append({"messages": messages, **kwargs})
        return await self._answer()

    async def search(self, messages: Any, **kwargs: Any) -> LLMResult:
        self.bodies.append({"messages": messages, "websearch": True, **kwargs})
        return await self._answer()

    async def image(self, prompt: str, **kwargs: Any) -> LLMResult:
        self.bodies.append({"prompt": prompt, **kwargs})
        return await self._answer()

    async def image_edit(self, prompt: str, *, images: Any, **kwargs: Any) -> LLMResult:
        self.bodies.append({"prompt": prompt, "images": images, **kwargs})
        return await self._answer()

    async def family_of(self, model: str) -> str:
        return "google"

    async def models(self) -> list[dict[str, Any]]:
        return [{"id": "gemini-3-flash", "owned_by": "antigravity"}]


class FakeSettings:
    cliproxy_default_model = "gemini-3-flash"
    llm_cache_enabled = False
    llm_cache_ttl_s = 60
    llm_default_project = "default"


def build_gateway(backend: FakeBackend, **kwargs: Any) -> Gateway:
    app = FastAPI()
    app.include_router(router)
    app.state.backends = BackendRegistry(cloud=backend)
    app.state.settings = FakeSettings()
    return Gateway("http://gw", transport=ASGITransport(app=app), project="tienda", **kwargs)


# ─── Construcción de peticiones ───────────────────────────────────────────────


def test_un_string_se_vuelve_un_turno_de_usuario():
    """La mayoría de las llamadas son un solo turno; obligar a escribir el dict
    entero para eso es ruido."""
    assert protocol.as_messages("hola") == [{"role": "user", "content": "hola"}]


def test_la_lista_completa_pasa_tal_cual():
    messages = [{"role": "system", "content": "sé breve"}, {"role": "user", "content": "x"}]
    assert protocol.as_messages(messages) is messages


def test_el_sdk_manda_una_sola_forma_de_websearch():
    """El gateway acepta las tres variantes de proveedor y traduce; desde el SDK
    basta con una."""
    body = protocol.chat_body([], model=None, max_tokens=10, websearch=True)
    assert body["tools"] == [{"type": "web_search"}]


def test_el_schema_viaja_como_response_format_de_openai():
    body = protocol.chat_body(
        [], model=None, max_tokens=10, schema={"type": "object"}, schema_name="Ficha"
    )
    assert body["response_format"]["json_schema"]["name"] == "Ficha"
    assert body["response_format"]["json_schema"]["strict"] is True


# ─── Contra el gateway real ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_devuelve_texto_y_uso():
    backend = FakeBackend(
        LLMResult(text="PONG", model="gemini-3-flash", prompt_tokens=4, completion_tokens=2)
    )
    async with build_gateway(backend) as gw:
        result = await gw.chat("Di PONG")

    assert result.text == "PONG"
    assert result.total_tokens == 6
    assert result.model == "gemini-3-flash"


@pytest.mark.asyncio
async def test_el_proyecto_viaja_en_la_cabecera():
    """Es lo que separa la contabilidad de costos entre consumidores."""
    backend = FakeBackend()
    async with build_gateway(backend) as gw:
        assert gw.project == "tienda"
        await gw.chat("x")
    # Llegó al router, que lo usa para etiquetar métricas e histórico.
    assert backend.bodies


@pytest.mark.asyncio
async def test_search_expone_las_fuentes_como_objetos():
    backend = FakeBackend(
        LLMResult(
            text="Argentina",
            model="gemini-3-flash",
            sources=[Source(uri="https://fifa.com", title="fifa.com")],
            searched=True,
        )
    )
    async with build_gateway(backend) as gw:
        result = await gw.search("¿quién ganó?")

    assert result.searched is True
    assert result.sources[0].uri == "https://fifa.com"


@pytest.mark.asyncio
async def test_structured_devuelve_el_objeto_ya_validado():
    """Sin re-parsear el texto: el gateway ya lo validó contra el schema."""
    backend = FakeBackend(LLMResult(text='{"titulo": "hola"}', model="gemini-3-flash"))
    async with build_gateway(backend) as gw:
        result = await gw.structured(
            "clasifica",
            schema={"type": "object", "properties": {"titulo": {"type": "string"}}},
        )

    assert result.parsed == {"titulo": "hola"}


@pytest.mark.asyncio
async def test_el_fallback_llega_al_llamante():
    """Quien llamó pidió un modelo y le respondió otro: tiene derecho a saberlo
    sin leer los logs del servidor."""
    backend = FakeBackend()
    async with build_gateway(backend) as gw:
        result = await gw.chat("x", model="modelo-que-no-existe-en-ninguna-cadena")

    # El backend falso responde a cualquier modelo, así que no hubo fallback.
    assert result.fell_back is False
    assert result.fell_back_from is None


# ─── Errores ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_el_error_trae_la_decision_de_reintentar_del_gateway():
    """No se deduce del status: sólo el gateway sabe si ya agotó la cadena."""
    backend = FakeBackend(raises=CliproxyRetryableError("429 en todos"))
    async with build_gateway(backend) as gw:
        with pytest.raises(ProximaError) as exc:
            await gw.chat("x")

    assert exc.value.kind == "upstream_unavailable"
    assert exc.value.retryable is True
    assert exc.value.status == 503


@pytest.mark.asyncio
async def test_una_salida_invalida_trae_el_detalle_de_cada_intento():
    backend = FakeBackend(LLMResult(text="no pienso dar json", model="m"))
    async with build_gateway(backend) as gw:
        with pytest.raises(ProximaError) as exc:
            await gw.structured("x", schema={"type": "object", "required": ["a"]})

    assert exc.value.kind == "invalid_structured_output"
    assert exc.value.retryable is False
    assert len(exc.value.attempts) == 3


@pytest.mark.asyncio
async def test_el_gateway_caido_se_reporta_como_reintentable():
    """Casi siempre es el gateway reiniciándose o la red."""
    gw = Gateway("http://127.0.0.1:9", timeout_seconds=2)
    with pytest.raises(ProximaError) as exc:
        await gw.chat("x")
    await gw.aclose()

    assert exc.value.kind == "unreachable"
    assert exc.value.retryable is True


# ─── Las dos superficies coinciden ────────────────────────────────────────────


def test_sync_y_async_exponen_la_misma_api():
    """Una diferencia entre las dos sería un bug imposible de justificar."""
    from proxima_llm import SyncGateway

    publicos = lambda cls: {  # noqa: E731
        n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n))
    }
    solo_async = {"aclose"}
    solo_sync = {"close"}
    assert publicos(Gateway) - solo_async == publicos(SyncGateway) - solo_sync


def test_sync_es_sincrono_de_verdad():
    """Usa httpx.Client, no un asyncio.run envolviendo al async: envolver
    rompería dentro de un bucle ya corriendo, que es donde más molesta."""
    import inspect

    from proxima_llm import SyncGateway

    assert not inspect.iscoroutinefunction(SyncGateway.chat)
    assert inspect.iscoroutinefunction(Gateway.chat)


def test_sync_habla_con_el_gateway_de_verdad():
    """La superficie síncrona contra un servidor real, no un transporte de
    mentira. Sin esto, `SyncGateway` sería un adorno que nunca se ejercitó — y es
    la que van a usar los consumidores síncronos."""
    import socket
    import threading
    import time

    import uvicorn
    from proxima_llm import SyncGateway

    backend = FakeBackend(LLMResult(text="PONG", model="gemini-3-flash"))
    app = FastAPI()
    app.include_router(router)
    app.state.backends = BackendRegistry(cloud=backend)
    app.state.settings = FakeSettings()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "el servidor de prueba no arrancó"

    try:
        with SyncGateway(f"http://127.0.0.1:{port}", project="tienda") as gw:
            result = gw.chat("Di PONG")
        assert result.text == "PONG"
        assert result.model == "gemini-3-flash"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ─── Imagen ───────────────────────────────────────────────────────────────────


def test_una_imagen_en_b64_llega_al_llamante():
    """Regresión: `data[].b64_json` es el DEFAULT del gateway, y el SDK sólo
    leía `data[].url`. La llamada costaba una imagen, devolvía 200, y
    `completion.images` venía vacío sin un solo error que lo explicara.

    `Image.url` promete un data URI siempre, así que el base64 pelado se
    envuelve acá y no en cada consumidor."""
    completion = protocol.read_images({"model": "gpt-image-2", "data": [{"b64_json": "QUJD"}]})
    assert [img.url for img in completion.images] == ["data:image/png;base64,QUJD"]


def test_url_explicito_se_respeta_tal_cual():
    """Si el cliente pidió `url`, el gateway ya manda un data URI entero: envolverlo
    otra vez lo rompería."""
    completion = protocol.read_images(
        {"model": "gpt-image-2", "data": [{"url": "data:image/png;base64,QUJD"}]}
    )
    assert [img.url for img in completion.images] == ["data:image/png;base64,QUJD"]


@pytest.mark.asyncio
async def test_image_edit_sube_la_foto_al_backend(tmp_path: Path):
    """La foto tiene que llegar al backend en bytes, no perderse por el camino.

    Se prueba con una ruta porque es la forma que usa un script; `_as_files`
    acepta además bytes y `(nombre, bytes)`.
    """
    backend = FakeBackend(
        LLMResult(text="", model="gpt-image-2", images=["data:image/png;base64,QUJD"])
    )
    foto = tmp_path / "producto.png"
    foto.write_bytes(b"\x89PNG-falso")

    async with build_gateway(backend) as gw:
        completion = await gw.image_edit("sobre madera", foto)

    assert completion.model == "gpt-image-2"
    assert [img.url for img in completion.images] == ["data:image/png;base64,QUJD"]
    enviadas = backend.bodies[-1]["images"]
    assert [(img.filename, img.content) for img in enviadas] == [("producto.png", b"\x89PNG-falso")]
