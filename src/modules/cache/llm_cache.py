"""
Cache de respuestas de LLM, en Redis.

Es **best-effort por diseño**: cualquier error de Redis se registra y se trata
como fallo de cache. La petición nunca depende de que el cache funcione — sin él
simplemente se paga la llamada.

Sobre la clave. Lleva la **familia de proveedor**, no el id exacto del modelo:
dos modelos de la misma familia responden lo bastante parecido como para
compartir entrada, mientras que mezclar familias sí contamina — un `gemini-3` y
un `gpt-5` no son intercambiables. Ese matiz costó un cambio de esquema en un
proyecto anterior; acá está desde el principio.

Y lleva `project`, porque el mismo prompt en dos proyectos distintos es la misma
respuesta pero **contabilidad distinta**: sin esa separación, un proyecto vería
sus costos absorbidos por el cache de otro y las cifras de F3 mentirían.

La versión del espacio de nombres (`v1`) permite invalidar todo cambiándola, sin
vaciar Redis ni tocar las claves de nadie más.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import structlog

from src.core.redis import get_redis

log = structlog.get_logger(__name__)

NAMESPACE_VERSION = "v1"

# Estados posibles, para las métricas de F3.
HIT = "hit"
MISS = "miss"
DISABLED = "disabled"
BYPASS = "bypass"


@dataclass(frozen=True)
class CacheKey:
    project: str
    route: str
    digest: str

    def __str__(self) -> str:
        return f"proxima:llm:{NAMESPACE_VERSION}:{self.project}:{self.route}:{self.digest}"


def derive_key(
    *,
    project: str,
    route: str,
    family: str,
    payload: dict[str, Any],
) -> CacheKey:
    """Clave estable a partir de la identidad de la petición.

    `payload` se serializa con las claves ordenadas para que dos peticiones
    equivalentes escritas en distinto orden caigan en la misma entrada.
    """
    material = json.dumps(
        {"family": family, "payload": payload}, sort_keys=True, ensure_ascii=False
    )
    digest = hashlib.sha256(material.encode()).hexdigest()[:32]
    return CacheKey(project=_slug(project), route=_slug(route), digest=digest)


def _slug(value: str) -> str:
    """Deja el fragmento apto para una clave de Redis: sin `:` ni espacios."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in value.strip().lower())
    return cleaned or "default"


async def get(key: CacheKey) -> dict[str, Any] | None:
    """Entrada cacheada, o None ante fallo o miss.

    Una entrada corrupta se trata como miss y se registra: es preferible pagar
    la llamada a devolver basura que alguien guardó mal.
    """
    try:
        raw = await get_redis().get(str(key))
    except Exception as exc:  # noqa: BLE001 — el cache nunca rompe la petición
        log.warning("llm_cache.error", op="get", key=str(key), error=str(exc))
        return None

    if raw is None:
        return None

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("llm_cache.corrupt_entry", key=str(key), error=str(exc))
        return None


async def set(key: CacheKey, value: dict[str, Any], ttl_seconds: int) -> None:
    """Guarda la entrada. Un fallo se registra y se sigue."""
    try:
        await get_redis().set(str(key), json.dumps(value, ensure_ascii=False), ex=ttl_seconds)
    except Exception as exc:  # noqa: BLE001
        log.warning("llm_cache.error", op="set", key=str(key), error=str(exc))
