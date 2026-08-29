"""
Carga de `config/routing.yaml`.

La tabla es configuración y no código a propósito: cambiar el orden de una
cadena, o agregar un modelo nuevo, no debería requerir un despliegue con
recompilación mental de nadie. Y al estar versionada, un cambio de política se
ve en el diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger(__name__)

DEFAULT_ROUTING_PATH = Path(__file__).resolve().parents[3] / "config" / "routing.yaml"

# Si el archivo falta, se sigue respondiendo con el modelo que pida el cliente:
# se pierde el fallback, no el servicio.
_FALLBACK_ON = ("upstream_unavailable", "upstream_timeout", "no_credential", "invalid_output")


@dataclass(frozen=True)
class BreakerPolicy:
    window_s: int = 60
    failure_threshold: int = 5
    open_for_s: int = 120


@dataclass(frozen=True)
class RoutingTable:
    version: int = 0
    routes: dict[str, list[str]] = field(default_factory=dict)
    fallback_on: frozenset[str] = frozenset(_FALLBACK_ON)
    breaker: BreakerPolicy = field(default_factory=BreakerPolicy)
    watchdog_skip_routes: frozenset[str] = frozenset()
    # Cuando el cliente pide `X-Proxima-No-Fallback`: la cadena se recorta a un
    # solo candidato. Así, si ese modelo tiene el circuito abierto o falla, la
    # petición falla — en vez de degradar a un modelo más débil. Es la única
    # forma de que la cabecera cubra también el salto por circuito abierto, que
    # es una rama distinta de `fallback_on`.
    single_candidate: bool = False

    def probeable_models(self) -> list[str]:
        """Modelos que el watchdog puede sondear con una llamada de chat.

        Un modelo que sólo aparece en rutas excluidas queda fuera. Uno que
        aparece además en una ruta sondeable sí entra: si responde a chat, la
        credencial sirve, y eso es lo que la sonda comprueba.
        """
        probeable: set[str] = set()
        for route, chain in self.routes.items():
            if route in self.watchdog_skip_routes:
                continue
            probeable.update(chain)
        return sorted(probeable)

    def candidates(self, route: str, requested: str | None = None) -> list[str]:
        """Modelos a probar, en orden y sin repetidos.

        Un modelo pedido explícitamente va primero: es una elección del cliente y
        se respeta. La cadena queda detrás como red, y como el modelo que
        respondió viaja en la respuesta, el cambio nunca es silencioso.
        """
        chain = list(self.routes.get(route, ()))
        if requested:
            chain = [requested, *[m for m in chain if m != requested]]

        seen: set[str] = set()
        ordered: list[str] = []
        for model in chain:
            if model not in seen:
                seen.add(model)
                ordered.append(model)
        return ordered[:1] if self.single_candidate else ordered

    def should_fallback(self, error_kind: str) -> bool:
        return error_kind in self.fallback_on


@lru_cache(maxsize=1)
def load_routing(path: str | None = None) -> RoutingTable:
    target = Path(path) if path else DEFAULT_ROUTING_PATH
    try:
        raw: dict[str, Any] = yaml.safe_load(target.read_text()) or {}
    except OSError as exc:
        log.error("routing.load_failed", path=str(target), error=str(exc))
        return RoutingTable()

    breaker_raw = raw.get("circuit_breaker") or {}
    return RoutingTable(
        version=int(raw.get("version", 0)),
        routes={k: list(v or []) for k, v in (raw.get("routes") or {}).items()},
        fallback_on=frozenset(raw.get("fallback_on") or _FALLBACK_ON),
        watchdog_skip_routes=frozenset((raw.get("watchdog") or {}).get("skip_routes") or ()),
        breaker=BreakerPolicy(
            window_s=int(breaker_raw.get("window_s", 60)),
            failure_threshold=int(breaker_raw.get("failure_threshold", 5)),
            open_for_s=int(breaker_raw.get("open_for_s", 120)),
        ),
    )
