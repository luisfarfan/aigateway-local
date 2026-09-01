"""
Costo de una llamada, calculado acá porque nadie más lo hace.

Ningún upstream devuelve el costo: CLIProxyAPI no lo calcula y los adaptadores
que leían `usage.cost_usd` del cuerpo recibían `0.0` en cada llamada sin
enterarse. Un cero indistinguible de un costo real es peor que no tener el dato,
así que este módulo separa las dos cosas:

  * `priced=False` — no hay precio para ese modelo. El costo es `None`, no cero.
    Un panel que suma `None` muestra un hueco; uno que suma ceros miente.
  * `charged=False` — hay precio, pero la llamada no se paga porque va contra
    una suscripción OAuth. El costo de lista se guarda igual como
    *equivalente*: es lo que habría costado por API, y sirve para dimensionar
    el ahorro y para comparar modelos entre sí.

La tabla vive en `config/pricing.yaml`, versionada, para que un cambio de precio
se vea en el diff y no mueva las cifras históricas en silencio.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger(__name__)

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[3] / "config" / "pricing.yaml"

_PER_TOKEN_DIVISOR = 1_000_000


@dataclass(frozen=True)
class Cost:
    """Qué costó una llamada, y si el número significa algo.

    `amount_usd` es lo que se cobra de verdad (0 con suscripción OAuth).
    `equivalent_usd` es el precio de lista, se pague o no.
    """

    amount_usd: float | None
    equivalent_usd: float | None
    priced: bool
    charged: bool
    source: str  # model | family | image | unknown

    @property
    def saved_usd(self) -> float:
        """Lo que se dejó de pagar por ir contra una suscripción."""
        if self.charged or self.equivalent_usd is None:
            return 0.0
        return self.equivalent_usd


def _lookup_keys(model: str) -> tuple[str, ...]:
    """El id tal cual, y sin su prefijo de backend o credencial.

    `paid/gpt-image-2` cuesta lo mismo que `gpt-image-2`: el prefijo dice POR
    DÓNDE entra la llamada, no qué modelo es. Sin esto, cargar una credencial de
    pago dejaba al modelo sin precio —`priced: False`— y su gasto desaparecía
    del reporte por la puerta de al lado justo cuando empezaba a ser real.

    Se busca primero el id completo por si alguien quiere fijarle un precio
    distinto a una vía concreta; el recorte es el respaldo.
    """
    if "/" in model:
        return (model, model.split("/", 1)[1])
    return (model,)


@dataclass(frozen=True)
class PricingTable:
    version: int
    models: dict[str, dict[str, float]]
    families: dict[str, dict[str, float]]
    images: dict[str, dict[str, float]]
    billing: dict[str, dict[str, bool]]
    api_key_prefixes: tuple[str, ...] = ()

    def rate_for(self, model: str, family: str) -> tuple[dict[str, float] | None, str]:
        """Tarifa por token del modelo. Exacto primero, familia después."""
        for candidate in _lookup_keys(model):
            if candidate in self.models:
                return self.models[candidate], "model"
        if family in self.families:
            return self.families[family], "family"
        return None, "unknown"

    def image_rate_for(self, model: str) -> dict[str, float] | None:
        """Precio por imagen. Mismo criterio de búsqueda que `rate_for`."""
        for candidate in _lookup_keys(model):
            if candidate in self.images:
                return self.images[candidate]
        return None

    def auth_mode_for(self, model: str) -> str:
        """`api_key` si el modelo va por una credencial de pago, si no `oauth`.

        Se decide por el prefijo del id porque es lo único que lo distingue: el
        upstream no informa qué credencial sirvió la llamada, y CLIProxyAPI
        expone las credenciales de pago con `prefix:` — así que `paid/gemini-…`
        es, literalmente, la señal.

        El sesgo va hacia `oauth` (no cobrado) a propósito, al revés que en
        `is_charged`: acá un falso positivo inventaría un gasto que no existe, y
        un reporte que exagera se descubre tarde y mal. Un modelo de pago sin su
        prefijo declarado aparece como equivalente, que es visible, en vez de
        como un cargo fantasma.
        """
        candidate = model.strip().lower()
        if any(candidate.startswith(p) for p in self.api_key_prefixes):
            return "api_key"
        return "oauth"

    def is_charged(self, auth_mode: str) -> bool:
        """`api_key` se paga; `oauth` va contra suscripción.

        Un modo desconocido se asume **cobrado**: subestimar el gasto es el
        error caro de los dos.
        """
        entry = self.billing.get(auth_mode)
        if entry is None:
            return True
        return bool(entry.get("charged", True))


@lru_cache(maxsize=1)
def load_pricing(path: str | None = None) -> PricingTable:
    """Carga la tabla, cacheada. Si falta el archivo, todo queda sin precio.

    Que falte no puede tumbar el gateway: se pierde la contabilidad, no el
    servicio. Pero se registra como error, no como aviso, porque un despliegue
    sin precios no se nota hasta que alguien pide el reporte de costos.
    """
    target = Path(path) if path else DEFAULT_PRICING_PATH
    try:
        raw: dict[str, Any] = yaml.safe_load(target.read_text()) or {}
    except OSError as exc:
        log.error("pricing.load_failed", path=str(target), error=str(exc))
        raw = {}

    return PricingTable(
        version=int(raw.get("version", 0)),
        models=raw.get("models") or {},
        families=raw.get("families") or {},
        images=raw.get("images") or {},
        billing=raw.get("billing") or {},
        api_key_prefixes=tuple(
            str(p).strip().lower()
            for p in ((raw.get("billing") or {}).get("api_key_prefixes") or ())
        ),
    )


def cost_of_tokens(
    *,
    model: str,
    family: str,
    prompt_tokens: int,
    completion_tokens: int,
    auth_mode: str = "oauth",
    table: PricingTable | None = None,
) -> Cost:
    pricing = table or load_pricing()
    rate, source = pricing.rate_for(model, family)

    if rate is None:
        log.debug("pricing.unknown_model", model=model, family=family)
        return Cost(None, None, priced=False, charged=False, source=source)

    equivalent = (
        prompt_tokens * float(rate.get("input", 0.0))
        + completion_tokens * float(rate.get("output", 0.0))
    ) / _PER_TOKEN_DIVISOR

    charged = pricing.is_charged(auth_mode)
    return Cost(
        amount_usd=equivalent if charged else 0.0,
        equivalent_usd=equivalent,
        priced=True,
        charged=charged,
        source=source,
    )


def cost_of_images(
    *,
    model: str,
    image_count: int,
    auth_mode: str = "oauth",
    table: PricingTable | None = None,
) -> Cost:
    """La imagen se cobra por unidad, no por token."""
    pricing = table or load_pricing()
    entry = pricing.image_rate_for(model)
    if entry is None:
        return Cost(None, None, priced=False, charged=False, source="unknown")

    equivalent = image_count * float(entry.get("per_image", 0.0))
    charged = pricing.is_charged(auth_mode)
    return Cost(
        amount_usd=equivalent if charged else 0.0,
        equivalent_usd=equivalent,
        priced=True,
        charged=charged,
        source="image",
    )
