"""
El prober de capacidades, sin red.

Lo que se fija aquí es lo que hace fiable al mapa: que una capacidad sólo se
marca `true` cuando se **verifica**, no cuando la llamada no lanzó. Un modelo que
ignora la imagen y responde igual NO tiene visión, y el prober tiene que decirlo.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.modules.providers.cliproxy.translate import EmbeddingResult, LLMResult
from src.modules.routing.prober import ModelCard, detect_drift, probe_model, to_serializable


class FakeBackend:
    """Backend controlable: se le dice qué devuelve en cada capacidad."""

    name = "fake"

    def __init__(
        self,
        *,
        chat_text: str = "PONG",
        tool_calls: list | None = None,
        vision_text: str = "rojo",
        embed_ok: bool = False,
        chat_raises: Exception | None = None,
    ):
        self._chat_text = chat_text
        self._tool_calls = tool_calls or []
        self._vision_text = vision_text
        self._embed_ok = embed_ok
        self._chat_raises = chat_raises

    async def chat(
        self, messages: Any, *, model: str, max_tokens: int, tools=None, tool_choice=None
    ):
        if self._chat_raises:
            raise self._chat_raises
        # visión: los mensajes traen un bloque de imagen
        is_vision = isinstance(messages[0].get("content"), list)
        if tools:
            return LLMResult(text="", model=model, tool_calls=self._tool_calls)
        if is_vision:
            return LLMResult(text=self._vision_text, model=model)
        return LLMResult(text=self._chat_text, model=model)

    async def embed(self, texts: list[str], *, model: str):
        if not self._embed_ok:
            from src.modules.backends.base import BackendCapabilityError

            raise BackendCapabilityError("no embeddings")
        return EmbeddingResult(vectors=[[0.1]], model=model)


@pytest.mark.asyncio
async def test_vision_solo_true_si_acierta_el_color():
    """El falso positivo que encontré corriéndolo: mistral/qwen marcaban visión
    porque respondían, aunque no vieran la imagen. Ahora hay que nombrar el
    color."""
    ve = FakeBackend(vision_text="es de color rojo intenso")
    card = await probe_model(ve, "m", owned_by="google", hosted="cloud")
    assert card.capabilities["vision"] is True

    ciego = FakeBackend(vision_text="no puedo ver imágenes")
    card = await probe_model(ciego, "m", owned_by="google", hosted="cloud")
    assert card.capabilities["vision"] is False
    assert "vision" in card.errors  # queda el porqué


@pytest.mark.asyncio
async def test_tools_solo_true_si_emite_tool_call():
    """Responder no basta: tiene que devolver el tool_call."""
    con = FakeBackend(tool_calls=[{"id": "c1", "type": "function"}])
    card = await probe_model(con, "m", owned_by="openai", hosted="cloud")
    assert card.capabilities["tools"] is True

    sin = FakeBackend(tool_calls=[])
    card = await probe_model(sin, "m", owned_by="openai", hosted="cloud")
    assert card.capabilities["tools"] is False


@pytest.mark.asyncio
async def test_un_modelo_que_no_chatea_no_arrastra_las_demas_sondas_de_texto():
    """Los modelos de sólo-embedding fallan chat; tools/visión no aplican."""
    from src.modules.providers.cliproxy.errors import CliproxyRequestError

    emb = FakeBackend(chat_raises=CliproxyRequestError("no es chat"), embed_ok=True)
    card = await probe_model(emb, "bge", owned_by="ollama", hosted="local")
    assert card.capabilities["chat"] is False
    assert card.capabilities["embeddings"] is True
    assert "tools" not in card.capabilities  # ni se intentó
    assert card.can == ["embeddings"]


def test_la_alarma_de_deriva_señala_lo_vivo_sin_mapear():
    """El seguro contra que sonnet-6 se cuele en silencio."""
    drift = detect_drift(
        live_ids={"gemini-3-flash", "sonnet-6-nuevo"},
        mapped_ids={"gemini-3-flash", "modelo-retirado"},
    )
    assert drift["live_unmapped"] == ["sonnet-6-nuevo"]
    assert drift["mapped_dead"] == ["modelo-retirado"]


def test_el_mapa_serializado_es_estable_para_diff():
    """Ordenado por id: un re-barrido sin cambios da el mismo YAML, y un cambio
    real salta en el diff."""
    cards = [
        ModelCard(id="zeta", owned_by="x", family="f", hosted="cloud"),
        ModelCard(id="alpha", owned_by="x", family="f", hosted="cloud"),
    ]
    out = to_serializable(cards)
    assert list(out["models"]) == ["alpha", "zeta"]
