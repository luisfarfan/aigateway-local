"""
Resuelve un tier (`cheap`/`fast`/`smart`/…) a una cadena de modelos ordenada.

La idea de fondo: un tier es una POLÍTICA sobre el mapa de capacidades, no una
lista de nombres. Así aguanta el cambio de modelos —sale opus-7, entra solo— sin
tocar nada. La política vive en `config/tiers.yaml`; el mapa lo genera el prober.

Dos clases de tier, y la diferencia es honesta:

  - **Medibles** (`cheap`, `fast`): se derivan solos del mapa. `cheap` ordena por
    costo (de `pricing.yaml`), `fast` por latencia (medida). Un modelo nuevo más
    barato entra sin que nadie edite nada.
  - **De calidad** (`smart`): la calidad no sale de una sonda. El orden es curado
    —humano o respaldado por evals— y se marca como tal. Lo único que el sistema
    hace solo es AVISAR cuando aparece un modelo sin clasificar.

La ruta pedida (chat/websearch/image/…) aporta su propia capacidad requerida, así
que los tiers NO la repiten: `fast` significa "el más rápido de lo que esta ruta
necesita", y sirve igual para chat que para imagen sin listas por ruta.

Las rutas de imagen se ordenan con sus propias cifras —latencia de imagen y
precio por unidad—, no con las de chat. Mezclarlas ordena por algo que no aplica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog
import yaml

from src.modules.observability.pricing import load_pricing

log = structlog.get_logger(__name__)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_TIERS_PATH = REPO / "config" / "tiers.yaml"
DEFAULT_MAP_PATH = REPO / "config" / "capabilities.generated.yaml"

# Qué capacidad exige cada ruta. Se suma a la del tier.
ROUTE_CAPABILITY = {
    "chat": "chat",
    "structured": "chat",
    "websearch": "websearch",
    "image": "image",
    # Editar exige la misma capacidad que generar: el prober sondea "image" y no
    # distingue las dos. Sin esta entrada, un tier sobre `image_edit` no exigiría
    # NADA y dejaría entrar modelos de sólo texto a una ruta de imagen.
    "image_edit": "image",
    "embeddings": "embeddings",
}

# Rutas que se tarifan y se miden por imagen, no por token. La distinción existe
# porque ordenar `cheap` con el precio por token de un modelo de imagen da el
# orden equivocado: gpt-image-2 no tiene tarifa por token y heredaría la de su
# familia (11.25), quedando último por una cifra que no aplica.
IMAGE_ROUTES = frozenset({"image", "image_edit"})


@dataclass(frozen=True)
class TierPolicy:
    name: str
    require: tuple[str, ...] = ()
    rank_by: str | None = None
    order: tuple[str, ...] = ()
    exclude_family: frozenset[str] = frozenset()


@dataclass
class ModelInfo:
    """Lo que el resolvedor necesita de un modelo, del mapa + precios."""

    id: str
    family: str
    capabilities: dict[str, bool]
    latency_s: float
    cost: float  # costo combinado in+out para ordenar; float('inf') si sin precio
    image_latency_s: float = 999.0
    image_cost: float = float("inf")  # por imagen, de `pricing.images`

    def has(self, cap: str) -> bool:
        return bool(self.capabilities.get(cap))

    def latency_for(self, route: str) -> float:
        """Una generación de imagen tarda 15-90 s y un chat menos de 2. Ordenar
        `fast` en imagen con la latencia de chat sería ordenar por otra cosa."""
        return self.image_latency_s if route in IMAGE_ROUTES else self.latency_s

    def cost_for(self, route: str) -> float:
        return self.image_cost if route in IMAGE_ROUTES else self.cost


@dataclass
class TierTable:
    policies: dict[str, TierPolicy] = field(default_factory=dict)
    models: dict[str, ModelInfo] = field(default_factory=dict)

    def is_tier(self, name: str) -> bool:
        return name in self.policies

    def resolve(self, tier: str, *, route: str) -> list[str]:
        """Cadena de modelos para `tier` en `route`, ya ordenada.

        Filtra por las capacidades del tier MÁS la que la ruta exige, aplica las
        exclusiones, y ordena según la política. Devuelve `[]` si nada califica —
        el llamador decide si eso es un 503 o un fallback a la ruta normal.
        """
        policy = self.policies.get(tier)
        if policy is None:
            return []

        needed = set(policy.require)
        if cap := ROUTE_CAPABILITY.get(route):
            needed.add(cap)

        eligible = [
            m
            for m in self.models.values()
            if all(m.has(c) for c in needed) and m.family not in policy.exclude_family
        ]

        if policy.order:
            # Curado: respeta el orden dado, quedándose sólo con los elegibles.
            # Un id del orden que ya no existe o no califica se cae solo.
            by_id = {m.id: m for m in eligible}
            return [mid for mid in policy.order if mid in by_id]

        if policy.rank_by == "cost_asc":
            eligible.sort(key=lambda m: (m.cost_for(route), m.latency_for(route)))
        elif policy.rank_by == "latency_asc":
            eligible.sort(key=lambda m: (m.latency_for(route), m.cost_for(route)))

        return [m.id for m in eligible]

    def unclassified(self, tier: str) -> list[str]:
        """Ids del `order` de un tier de calidad que ya no están en el mapa.

        Parte de la alarma de deriva: un `smart` que nombra un modelo retirado
        debe verse, no fallar en silencio.
        """
        policy = self.policies.get(tier)
        if not policy or not policy.order:
            return []
        return [mid for mid in policy.order if mid not in self.models]


# Cache por mtime, NO por proceso: el prober regenera el mapa en un job aparte,
# y el gateway tiene que ver el mapa nuevo sin reiniciar. Se relee sólo cuando el
# archivo cambió; en el caso normal (sin cambios) es una llamada a stat, barato.
_CACHE: dict[str, object] = {"table": None, "stamp": None}


def load_tiers(tiers_path: str | None = None, map_path: str | None = None) -> TierTable:
    """Políticas + mapa + precios. Se recarga solo cuando el mapa o las políticas
    cambian en disco — así un re-barrido del prober surte efecto sin reiniciar."""
    tp = Path(tiers_path) if tiers_path else DEFAULT_TIERS_PATH
    mp = Path(map_path) if map_path else DEFAULT_MAP_PATH
    stamp = (_mtime(tp), _mtime(mp))
    if _CACHE["table"] is None or _CACHE["stamp"] != stamp:
        _CACHE["table"] = TierTable(policies=_load_policies(tp), models=_load_models(mp))
        _CACHE["stamp"] = stamp
        log.info("tiers.loaded", models=len(_CACHE["table"].models))  # type: ignore[union-attr]
    return _CACHE["table"]  # type: ignore[return-value]


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def clear_cache() -> None:
    """Fuerza la recarga en el próximo `load_tiers`. Para tests."""
    _CACHE["table"] = None
    _CACHE["stamp"] = None


def _load_policies(path: Path) -> dict[str, TierPolicy]:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except OSError as exc:
        log.warning("tiers.no_policies", path=str(path), error=str(exc))
        return {}
    out: dict[str, TierPolicy] = {}
    for name, spec in (raw.get("tiers") or {}).items():
        spec = spec or {}
        out[name] = TierPolicy(
            name=name,
            require=tuple(spec.get("require") or ()),
            rank_by=spec.get("rank_by"),
            order=tuple(spec.get("order") or ()),
            exclude_family=frozenset(spec.get("exclude_family") or ()),
        )
    return out


def _load_models(path: Path) -> dict[str, ModelInfo]:
    """Junta el mapa de capacidades con los precios.

    Si el mapa no existe (nunca se corrió el prober), no hay tiers: mejor eso que
    inventar un roster. El costo sale de `pricing.yaml`; sin precio, el modelo va
    al final de `cheap`, no se descarta.
    """
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except OSError:
        log.warning("tiers.no_map", path=str(path), hint="corre scripts/probe_models.py")
        return {}

    pricing = load_pricing()
    models: dict[str, ModelInfo] = {}
    for model_id, card in (raw.get("models") or {}).items():
        family = card.get("family", "unknown")
        rate, _ = pricing.rate_for(model_id, family)
        cost = float("inf")
        if rate is not None:
            cost = float(rate.get("input", 0.0)) + float(rate.get("output", 0.0))
        image_rate = pricing.images.get(model_id) or {}
        models[model_id] = ModelInfo(
            id=model_id,
            family=family,
            capabilities=card.get("capabilities") or {},
            latency_s=float(card.get("chat_latency_s") or 999.0),
            cost=cost,
            # Sin medir todavía, un modelo va al final de `fast` en vez de
            # colarse primero por un 0 que nadie midió.
            image_latency_s=float(card.get("image_latency_s") or 999.0),
            image_cost=float(image_rate.get("per_image", float("inf"))),
        )
    return models
