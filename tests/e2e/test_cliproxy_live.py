"""
End-to-end contra una instancia real de CLIProxyAPI.

Se salta solo si no hay `CLIPROXY_API_KEY` en el entorno, así que la suite normal
sigue corriendo sin red. Correrlo cuando se toque la traducción o al cambiar de
versión del binario de arriba:

    CLIPROXY_API_KEY=... pytest tests/e2e -v

Lo que verifica y los tests offline no pueden: que las superficies que elegimos
siguen existiendo y comportándose igual del otro lado.
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.openai_compat.router import router
from src.modules.providers.cliproxy.client import CliproxyClient

API_KEY = os.environ.get("CLIPROXY_API_KEY", "")
BASE_URL = os.environ.get("CLIPROXY_BASE_URL", "http://127.0.0.1:8417")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not API_KEY, reason="CLIPROXY_API_KEY no está definida"),
]


class _Settings:
    cliproxy_default_model = os.environ.get("CLIPROXY_DEFAULT_MODEL", "gemini-3-flash")


@pytest.fixture
async def gateway():
    # La base se pasa CON `/v1` a propósito: los consumidores la escriben de las
    # dos formas y una barra de más no puede costar un 404.
    client = CliproxyClient(base_url=f"{BASE_URL}/v1", api_key=API_KEY)
    app = FastAPI()
    app.include_router(router)
    app.state.cliproxy_client = client
    app.state.settings = _Settings()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://gw", timeout=120) as http:
        yield http
    await client.aclose()


@pytest.mark.asyncio
async def test_base_url_tolera_el_sufijo_v1():
    client = CliproxyClient(base_url=f"{BASE_URL}/v1", api_key=API_KEY)
    assert client.base_url == BASE_URL.rstrip("/")
    await client.aclose()


@pytest.mark.asyncio
async def test_lista_modelos(gateway):
    response = await gateway.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"]


@pytest.mark.asyncio
async def test_chat_responde_con_uso(gateway):
    response = await gateway.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Responde solo con la palabra PONG."}],
            "max_tokens": 20,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "PONG" in body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_websearch_devuelve_fuentes(gateway):
    """El cliente manda el tool block de Gemini; el gateway decide la superficie.

    Que vuelvan fuentes es la prueba de que fue por la nativa: por el path
    OpenAI-compatible el tool se descarta y la respuesta llega sin grounding.
    """
    response = await gateway.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Precio del Bitcoin hoy. Cita la URL."}],
            "tools": [{"type": "web_search"}],
            "max_tokens": 250,
        },
    )
    assert response.status_code == 200
    proxima = response.json()["proxima"]
    assert proxima["searched"] is True
    assert proxima["sources"], "la superficie nativa tiene que devolver fuentes"
