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


def test_un_barrido_barato_no_borra_lo_que_no_probó():
    """Bug encontrado con el prober programado: corre sin --images y sobrescribía
    el mapa, borrando la capacidad de imagen que fijó un barrido completo. Ahora
    lo no probado se hereda del mapa anterior."""
    from src.modules.routing.prober import carry_forward

    # barrido barato: NO probó image ni websearch
    card = ModelCard(
        id="gemini-3-flash",
        owned_by="google",
        family="google",
        hosted="cloud",
        capabilities={"chat": True, "tools": True, "vision": True, "embeddings": False},
    )
    previous = {
        "gemini-3-flash": {
            "capabilities": {"chat": True, "image": True, "websearch": True},
        }
    }
    carry_forward([card], previous, probed=frozenset({"chat", "tools", "vision", "embeddings"}))

    assert card.capabilities["image"] is True  # heredado, no borrado
    assert card.capabilities["websearch"] is True  # heredado
    assert card.capabilities["chat"] is True  # el de este barrido


def test_lo_probado_este_barrido_manda_sobre_lo_heredado():
    """Si el barrido SÍ probó una capacidad, su resultado gana — no se pisa con
    el viejo."""
    from src.modules.routing.prober import carry_forward

    card = ModelCard(
        id="m",
        owned_by="x",
        family="x",
        hosted="cloud",
        capabilities={"chat": True, "image": False},  # este barrido probó image=false
    )
    previous = {"m": {"capabilities": {"image": True}}}  # antes era true
    carry_forward(
        [card], previous, probed=frozenset({"chat", "vision", "tools", "embeddings", "image"})
    )

    assert card.capabilities["image"] is False  # el nuevo manda; no se heredó el viejo


@pytest.mark.asyncio
async def test_image_solo_true_si_vuelven_bytes():
    """El bug reportado desde intel-v2: flash-lite (sólo chat) salía image=true
    porque la llamada no lanzaba, aunque respondiera texto sin imagen. Ahora se
    exige que VUELVA una imagen."""
    from src.modules.providers.cliproxy.translate import LLMResult

    class ImgBackend(FakeBackend):
        def __init__(self, images):
            super().__init__()
            self._images = images

        async def image(self, prompt, *, model, size=None, quality=None):
            return LLMResult(text="", model=model, images=self._images)

    genera = await probe_model(
        ImgBackend(["data:image/png;base64,AAA"]),
        "flash-image",
        owned_by="google",
        hosted="cloud",
        include_expensive=frozenset({"image"}),
    )
    assert genera.capabilities["image"] is True

    no_genera = await probe_model(
        ImgBackend([]),
        "flash-lite",
        owned_by="google",
        hosted="cloud",
        include_expensive=frozenset({"image"}),
    )
    assert no_genera.capabilities["image"] is False


@pytest.mark.asyncio
async def test_un_cooldown_de_imagen_conserva_la_etiqueta_previa():
    """La generación se rate-limitea. Un cooldown NO significa que el modelo no
    sepa generar: se conserva lo último conocido en vez de sacarlo de los tiers."""
    from src.modules.providers.cliproxy.errors import CliproxyRetryableError

    class Cooldown(FakeBackend):
        async def image(self, prompt, *, model, size=None, quality=None):
            raise CliproxyRetryableError("model_cooldown: 429")

    card = await probe_model(
        Cooldown(),
        "flash-image",
        owned_by="google",
        hosted="cloud",
        include_expensive=frozenset({"image"}),
        previous_card={"capabilities": {"image": True}},  # antes SÍ generaba
    )
    assert card.capabilities["image"] is True  # se conservó, no se marcó false


def test_un_barrido_barato_no_borra_la_latencia_de_imagen():
    """REGRESIÓN. El barrido programado corre sin `--images` cada 6 h para no
    gastar cuota. `carry_forward` heredaba el sí/no de la capacidad pero NO la
    latencia medida, así que borraba `image_latency_s` y dejaba a `fast` sin con
    qué ordenar la ruta de imagen: todos los modelos empataban en el default de
    "sin medir".

    Pasó de verdad: un barrido caro midió las latencias y el barato las borró
    seis horas después.
    """
    from src.modules.routing.prober import ALWAYS_PROBED, ModelCard, carry_forward

    previo = {
        "gpt-image-2": {
            "capabilities": {"image": True},
            "image_latency_s": 12.5,
        }
    }
    # Barrido barato: no sondeó imagen, así que la ficha viene vacía de eso.
    card = ModelCard(id="gpt-image-2", owned_by="openai", family="openai", hosted="cloud")

    carry_forward([card], previo, probed=ALWAYS_PROBED)

    assert card.capabilities["image"] is True
    assert card.image_latency_s == 12.5, "sin la latencia, `fast` no puede ordenar imagen"


def test_una_medicion_nueva_le_gana_a_la_heredada():
    """Heredar es para lo que NO se midió hoy. Si el barrido sí midió, ese dato
    manda — si no, una latencia vieja sobreviviría para siempre."""
    from src.modules.routing.prober import ALWAYS_PROBED, ModelCard, carry_forward

    previo = {"m": {"capabilities": {"image": True}, "image_latency_s": 99.0}}
    card = ModelCard(id="m", owned_by="o", family="f", hosted="cloud")
    card.image_latency_s = 3.0   # medido en este barrido

    carry_forward([card], previo, probed=ALWAYS_PROBED)

    assert card.image_latency_s == 3.0
