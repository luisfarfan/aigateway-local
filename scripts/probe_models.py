"""
Genera el mapa de capacidades de todos los modelos vivos.

    python scripts/probe_models.py                    # sondas baratas
    python scripts/probe_models.py --websearch        # + búsqueda web (1 por modelo)
    python scripts/probe_models.py --images           # + generación de imagen (LENTO/CARO)
    python scripts/probe_models.py --out config/capabilities.generated.yaml

Sale un YAML con lo que cada modelo sabe hacer, medido, más un reporte de deriva
(vivo-pero-sin-mapear). Pensado para correr programado; aquí es el arranque manual.

Requiere el gateway configurado (lee .env). Cada capacidad es una llamada mínima:
barato salvo --images.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.core.config import get_settings  # noqa: E402
from src.modules.backends.ollama import OllamaBackend  # noqa: E402
from src.modules.backends.registry import BackendRegistry  # noqa: E402
from src.modules.observability.pricing import load_pricing  # noqa: E402
from src.modules.providers.cliproxy.client import CliproxyClient  # noqa: E402
from src.modules.routing.prober import (  # noqa: E402
    ALWAYS_PROBED,
    carry_forward,
    detect_drift,
    probe_all,
    to_serializable,
)


def _cost_note(model: str, family: str) -> str:
    """Anota el costo del mapa, reusando pricing.yaml. Los precios son estimados
    y están marcados como tal en pricing.yaml; aquí sólo se refleja."""
    rate, source = load_pricing().rate_for(model, family)
    if rate is None:
        return "sin-precio"
    return f"in={rate.get('input')} out={rate.get('output')} ({source})"


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--websearch", action="store_true", help="probar búsqueda web")
    parser.add_argument("--images", action="store_true", help="probar imagen (LENTO)")
    parser.add_argument("--out", default="config/capabilities.generated.yaml")
    args = parser.parse_args(argv[1:])

    settings = get_settings()
    cloud = CliproxyClient(
        base_url=settings.cliproxy_base_url,
        api_key=settings.cliproxy_api_key,
        timeout_seconds=settings.cliproxy_timeout_s,
        image_timeout_seconds=settings.cliproxy_image_timeout_s,
    )
    local = (
        OllamaBackend(base_url=settings.ollama_base_url) if settings.enable_backend_ollama else None
    )
    registry = BackendRegistry(cloud=cloud, local=local)

    expensive = set()
    if args.websearch:
        expensive.add("websearch")
    if args.images:
        expensive.add("image")

    print(f"Sondeando modelos vivos (caras: {sorted(expensive) or 'ninguna'})...\n")
    out = REPO / args.out
    previous = {}
    if out.exists():
        previous = (yaml.safe_load(out.read_text()) or {}).get("models") or {}

    cards = await probe_all(registry, include_expensive=frozenset(expensive))

    # Lo no sondeado hoy se hereda del mapa anterior: un barrido barato no debe
    # borrar la capacidad de imagen que fijó uno completo.
    carry_forward(cards, previous, probed=ALWAYS_PROBED | frozenset(expensive))

    for c in sorted(cards, key=lambda c: (c.hosted, c.id)):
        cost = _cost_note(c.id, c.family)
        lat = f"{c.chat_latency_s}s" if c.chat_latency_s is not None else "-"
        print(f"  [{c.hosted:5}] {c.id:30} {lat:>7}  {cost:24}  → {', '.join(c.can) or '(nada)'}")

    payload = to_serializable(cards)
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    print(f"\nMapa escrito en {out}")

    drift = detect_drift(live_ids={c.id for c in cards}, mapped_ids=set(payload["models"]))
    if drift["live_unmapped"]:
        print(f"\n⚠ ALARMA DE DERIVA — vivos sin mapear: {drift['live_unmapped']}")
    else:
        print("\n✓ sin deriva: todos los vivos quedaron mapeados")

    await cloud.aclose()
    if local:
        await local.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
