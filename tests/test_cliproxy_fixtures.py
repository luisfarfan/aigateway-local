"""
Contrato de CLIProxyAPI, fijado contra respuestas reales grabadas (F0).

Estos tests no llaman a la red: releen `tests/fixtures/cliproxy/*.json`, grabados
por `scripts/record_fixtures.py` contra una instancia autenticada. Cada aserción
es un campo del que `translate.py` va a depender, así que si un re-grabado trae
una forma distinta, esto falla en vez de romper el provider en runtime.

Los dos casos negativos son parte del contrato: documentan que vanilla NO
soporta el websearch de Gemini por la superficie OpenAI-compatible, y la forma
exacta del error cuando ninguna credencial cubre un modelo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cliproxy"


def load(name: str) -> dict[str, Any]:
    path = FIXTURE_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"fixture {name} no grabado — correr scripts/record_fixtures.py")
    return json.loads(path.read_text())


def body_of(name: str) -> Any:
    return load(name)["response"]["body"]


# ─── Inventario ───────────────────────────────────────────────────────────────


def test_models_list_expone_id_y_owned_by():
    """El routing agrupa por familia de proveedor, y `owned_by` es el único
    campo que la distingue: un `claude-sonnet-4-6` con owned_by=antigravity no
    es el mismo camino que uno con owned_by=anthropic."""
    data = body_of("models_list")["data"]
    assert data, "el inventario no puede venir vacío"
    for model in data:
        assert model["id"]
        assert model["owned_by"]


# ─── Superficie base ──────────────────────────────────────────────────────────


def test_chat_plain_forma_openai():
    body = body_of("chat_plain")
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert "PONG" in message["content"]
    # `usage` alimenta el cálculo de costo (F3). Sin él no hay nada que medir.
    assert body["usage"]["prompt_tokens"] >= 0
    assert body["usage"]["completion_tokens"] >= 0


# ─── Websearch ────────────────────────────────────────────────────────────────


def test_websearch_codex_devuelve_web_search_call():
    """Codex funciona en vanilla. La prueba de que buscó de verdad es el item
    `web_search_call` en `output`, no que el texto parezca actualizado."""
    body = body_of("websearch_codex_responses")
    types = [item["type"] for item in body["output"]]
    assert "web_search_call" in types
    assert "message" in types

    text = "".join(
        block.get("text", "")
        for item in body["output"]
        if item["type"] == "message"
        for block in item["content"]
    )
    assert text.strip()


def test_websearch_gemini_por_openai_compat_no_hace_grounding():
    """NEGATIVO: vanilla acepta el request (HTTP 200) pero descarta el tool
    block, así que la respuesta no trae grounding. Es la razón de existir de la
    ruta nativa — si algún día esto empieza a funcionar, este test falla y
    revisamos si `translate.py` puede simplificarse."""
    fixture = load("websearch_gemini_openai_compat_UNSUPPORTED")
    assert fixture["response"]["status"] == 200
    body = fixture["response"]["body"]
    assert "groundingMetadata" not in json.dumps(body)


def test_websearch_gemini_nativo_trae_fuentes():
    """La superficie nativa sí busca, y `groundingChunks` es de donde salen las
    fuentes: uri + title por cada una."""
    candidate = body_of("websearch_gemini_native")["candidates"][0]
    grounding = candidate["groundingMetadata"]
    assert grounding["webSearchQueries"]

    chunks = grounding["groundingChunks"]
    assert chunks
    for chunk in chunks:
        assert chunk["web"]["uri"]
        assert chunk["web"]["title"]


# ─── Imagen ───────────────────────────────────────────────────────────────────


def test_imagen_gemini_llega_por_chat_como_data_uri():
    """Gemini no usa /v1/images/generations: la imagen viene dentro del mensaje
    de chat, en `images[].image_url.url`, como data URI."""
    message = body_of("image_gemini_chat")["choices"][0]["message"]
    image = message["images"][0]["image_url"]["url"]
    # El grabador sustituye blobs por su huella para no versionar 874 KB.
    assert image["__blob__"] is True
    assert image["prefix"].startswith("data:image/")
    assert image["length"] > 1000


def test_modelo_sin_credencial_da_auth_not_found():
    """NEGATIVO: el routing (F4) tiene que distinguir 'ninguna credencial cubre
    este modelo' de un fallo de red o una cuota agotada — los tres pintan como
    5xx pero sólo uno se arregla reintentando en otro modelo."""
    fixture = load("image_openai_generations_NOAUTH")
    assert fixture["response"]["status"] >= 500
    error = fixture["response"]["body"]["error"]
    assert error["message"].startswith("auth_not_found")
    assert "no auth available" in error["message"]
