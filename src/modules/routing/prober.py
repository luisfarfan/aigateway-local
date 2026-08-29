"""
Prober de modelos: descubre qué sabe hacer cada modelo, midiéndolo.

Existe porque los tiers (`smart`/`cheap`/`fast`/…) no pueden cablearse sobre una
lista de nombres: los modelos cambian —sale opus-7, gemini-3.9— y una lista fija
se pudre. La solución es un mapa que se **regenera**, no que se mantiene a mano.

Qué hace: recorre los modelos vivos y prueba cada capacidad con la llamada más
barata posible. El resultado es medición, no opinión:

  - chat            → responde, con latencia y tokens
  - tools           → devuelve `tool_calls` ante una función
  - vision          → describe una imagen
  - embeddings      → devuelve vectores
  - websearch       → devuelve grounding (opcional: gasta una búsqueda)
  - image           → genera imagen (opcional: caro y lento, 30-150 s)

Lo que NO mide, y hay que ser honesto: la **calidad**. "El más listo" o "mejor
programador" son juicios que una máquina no saca de una sonda barata — eso son
evals. El prober da el roster y las capacidades; el ranking de calidad es otra
capa.

Diseñado para correr programado. `detect_drift` compara los modelos vivos con el
último mapa y señala lo que está vivo-pero-sin-clasificar, para que un modelo
nuevo nunca se cuele en silencio.
"""

from __future__ import annotations

import base64
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from src.modules.backends.base import Backend, BackendCapabilityError
from src.modules.backends.registry import BackendRegistry
from src.modules.providers.cliproxy.errors import CliproxyError

log = structlog.get_logger(__name__)

# Sondas mínimas. Pocos tokens: el prober corre sobre TODO el roster y puede
# repetirse a menudo, así que cada llamada tiene que ser barata.
_CHAT = [{"role": "user", "content": "Reply with the single word: PONG"}]
_MAX_TOKENS = 8

_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "responde ping",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    }
]
_TOOL_PROMPT = [{"role": "user", "content": "Llama a la herramienta ping con x='a'."}]


def _red_square_png() -> bytes:
    """PNG 16x16 rojo puro, generado sin dependencias. Un bloque, no un pixel:
    un modelo de visión real lo describe; uno que ignora la imagen, no."""
    import struct
    import zlib

    w = h = 16
    raw = b"".join(b"\x00" + bytes([220, 30, 30]) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


_VISION_URI = "data:image/png;base64," + base64.b64encode(_red_square_png()).decode()
_VISION = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "¿De qué color es la imagen? Una palabra."},
            {"type": "image_url", "image_url": {"url": _VISION_URI}},
        ],
    }
]
# Un modelo que DE VERDAD ve la imagen nombra el rojo. Uno que la ignora y
# responde igual (falso positivo) no acierta el color.
_VISION_EXPECT = ("rojo", "red", "rot", "rouge")

# Capacidades cuya sonda cuesta caro o lento; off por defecto.
EXPENSIVE = frozenset({"image", "websearch"})


@dataclass
class ModelCard:
    """Lo que se sabe de un modelo tras probarlo. Serializable a YAML/JSON."""

    id: str
    owned_by: str
    family: str
    hosted: str  # cloud | local
    capabilities: dict[str, bool] = field(default_factory=dict)
    chat_latency_s: float | None = None
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def can(self) -> list[str]:
        return sorted(c for c, ok in self.capabilities.items() if ok)


async def _timed_ok(coro) -> tuple[bool, float, str | None]:
    """Corre la sonda; devuelve (funcionó, segundos, error o None).

    Un `BackendCapabilityError` es un 'no' legítimo (el backend declara que no
    hace eso), no un fallo: se distingue del error de verdad para no ensuciar el
    registro de errores con negativas esperadas.
    """
    started = time.monotonic()
    try:
        await coro
        return True, time.monotonic() - started, None
    except BackendCapabilityError:
        return False, time.monotonic() - started, None
    except (CliproxyError, Exception) as exc:  # noqa: BLE001
        return False, time.monotonic() - started, str(exc)[:150]


async def probe_model(
    backend: Backend,
    model: str,
    *,
    owned_by: str,
    hosted: str,
    include_expensive: frozenset[str] = frozenset(),
) -> ModelCard:
    """Prueba un modelo capacidad por capacidad. `model` ya sin prefijo de backend."""
    from src.modules.providers.cliproxy.families import Family, family_from_owned_by

    family = family_from_owned_by(owned_by)
    if family is Family.UNKNOWN and hosted == "local":
        family = Family.LOCAL
    card = ModelCard(id=model, owned_by=owned_by, family=str(family), hosted=hosted)

    # chat es la puerta: si no responde, el resto de sondas de texto no aplican.
    ok, secs, err = await _timed_ok(backend.chat(_CHAT, model=model, max_tokens=_MAX_TOKENS))
    card.capabilities["chat"] = ok
    card.chat_latency_s = round(secs, 2)
    if err:
        card.errors["chat"] = err

    if ok:
        # tools: no basta con que responda; tiene que emitir el tool_call.
        try:
            r = await backend.chat(_TOOL_PROMPT, model=model, max_tokens=_MAX_TOKENS, tools=_TOOL)
            card.capabilities["tools"] = bool(r.tool_calls)
        except Exception as exc:  # noqa: BLE001
            card.capabilities["tools"] = False
            card.errors["tools"] = str(exc)[:150]

        # vision: no basta con que responda; tiene que ACERTAR el color. Así se
        # descarta el modelo que ignora la imagen y contesta igual.
        try:
            r = await backend.chat(_VISION, model=model, max_tokens=_MAX_TOKENS)
            saw = any(w in r.text.lower() for w in _VISION_EXPECT)
            card.capabilities["vision"] = saw
            if not saw:
                card.errors["vision"] = f"no acertó el color: {r.text[:60]!r}"
        except Exception as exc:  # noqa: BLE001
            card.capabilities["vision"] = False
            card.errors["vision"] = str(exc)[:150]

    # embeddings: independiente de chat (los modelos de embed no chatean).
    emb_ok, _, emb_err = await _timed_ok(backend.embed(["probe"], model=model))
    card.capabilities["embeddings"] = emb_ok
    if emb_err and not emb_ok:
        card.errors["embeddings"] = emb_err

    if "websearch" in include_expensive and card.capabilities.get("chat"):
        ws_ok, _, ws_err = await _timed_ok(
            backend.search(
                [{"role": "user", "content": "capital de Francia, cita url"}], model=model
            )
        )
        card.capabilities["websearch"] = ws_ok
        if ws_err and not ws_ok:
            card.errors["websearch"] = ws_err

    if "image" in include_expensive:
        img_ok, _, img_err = await _timed_ok(backend.image("un punto rojo", model=model))
        card.capabilities["image"] = img_ok
        if img_err and not img_ok:
            card.errors["image"] = img_err

    log.info("prober.model", model=model, hosted=hosted, can=card.can)
    return card


async def probe_all(
    registry: BackendRegistry,
    *,
    include_expensive: frozenset[str] = frozenset(),
) -> list[ModelCard]:
    """Prueba todos los modelos vivos de todos los backends."""
    cards: list[ModelCard] = []
    seen: set[str] = set()

    for backend in _backends_of(registry):
        try:
            listed = await backend.models()
        except Exception as exc:  # noqa: BLE001
            log.warning("prober.list_failed", backend=backend.name, error=str(exc))
            continue
        hosted = "local" if backend.name == "ollama" else "cloud"
        for entry in listed:
            full_id = entry.get("id", "")
            if not full_id or full_id in seen:
                continue
            seen.add(full_id)
            # el id que el backend entiende no lleva el prefijo `ollama/`
            bare = full_id.split("/", 1)[1] if full_id.startswith("ollama/") else full_id
            cards.append(
                await probe_model(
                    backend,
                    bare,
                    owned_by=entry.get("owned_by", ""),
                    hosted=hosted,
                    include_expensive=include_expensive,
                )
            )
    return cards


def _backends_of(registry: BackendRegistry) -> list[Backend]:
    """Los backends configurados, cloud y (si hay) local."""
    backends: list[Backend] = []
    cloud = registry.resolve("_")
    if cloud:
        backends.append(cloud.backend)
    if registry.local_available:
        local = registry.resolve("ollama/_")
        if local:
            backends.append(local.backend)
    return backends


def detect_drift(live_ids: set[str], mapped_ids: set[str]) -> dict[str, list[str]]:
    """Alarma de deriva: qué está vivo sin mapear, y qué se mapeó sin estar vivo.

    Lo primero es lo importante — un modelo nuevo (sonnet-6) que aparece y nadie
    clasificó. Lo segundo señala un mapa que envejeció (modelos retirados).
    """
    return {
        "live_unmapped": sorted(live_ids - mapped_ids),
        "mapped_dead": sorted(mapped_ids - live_ids),
    }


def to_serializable(cards: list[ModelCard]) -> dict[str, Any]:
    """Mapa listo para volcar a YAML, ordenado para un diff estable."""
    return {
        "version": 1,
        "models": {c.id: _card_dict(c) for c in sorted(cards, key=lambda c: c.id)},
    }


def _card_dict(card: ModelCard) -> dict[str, Any]:
    d = asdict(card)
    d.pop("id")
    return d
