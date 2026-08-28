"""
Watchdog de modelos: comprueba cuáles responden de verdad.

Existe por un hallazgo concreto de F0: `GET /v1/models` **no es prueba de vida**.
Una instancia llegó a anunciar 62 modelos de los cuales 32 devolvían
`OAuth access token has been revoked` o `403 PERMISSION_DENIED`. Un selector que
filtre por esa lista elige con toda confianza un modelo muerto.

Por eso el watchdog **prueba**, no lista: manda la llamada más barata posible a
cada modelo de las cadenas y anota si contestó. El resultado se escribe en el
circuit breaker, que es donde el routing ya mira — dos tablas de salud distintas
acabarían contradiciéndose.

Cuánto cuesta: una llamada de pocos tokens por modelo y por barrido. Con el
intervalo por defecto (15 min) y cuatro modelos, son 16 llamadas mínimas por
hora. Bastante menos que descubrir un modelo muerto en medio del tráfico real.
"""

from __future__ import annotations

import asyncio

import structlog

from src.modules.backends.registry import BackendRegistry
from src.modules.routing.breaker import CircuitBreaker
from src.modules.routing.config import RoutingTable

log = structlog.get_logger(__name__)

# La sonda más barata que sigue ejercitando el camino completo: autenticación,
# resolución de modelo y generación. Un `GET /v1/models` no probaría ninguna.
_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]
_PROBE_MAX_TOKENS = 5


async def probe(registry: BackendRegistry, model: str) -> tuple[bool, str | None]:
    """`(vivo, motivo)`. Vivo es que devolvió algo sin error.

    El backend sale del registry: sondear un modelo local contra la API cloud
    lo marcaría muerto siempre.
    """
    resolved = registry.resolve(model)
    if resolved is None:
        return False, "backend no configurado"
    try:
        await resolved.backend.chat(
            _PROBE_MESSAGES, model=resolved.model, max_tokens=_PROBE_MAX_TOKENS
        )
    except Exception as exc:  # noqa: BLE001 — cualquier fallo cuenta como muerto
        return False, str(exc)[:200]
    return True, None


async def sweep(
    registry: BackendRegistry,
    table: RoutingTable,
    breaker: CircuitBreaker,
    *,
    open_for_s: int,
) -> dict[str, bool]:
    """Sondea los modelos sondeables y actualiza el breaker.

    "Sondeables" excluye las rutas que `routing.yaml` marca: un modelo de sólo
    imagen rechaza una llamada de chat aunque esté vivo, y marcarlo muerto por
    eso rompería su ruta entera.
    """
    models = table.probeable_models()
    results: dict[str, bool] = {}

    for model in models:
        alive, reason = await probe(registry, model)
        results[model] = alive
        if alive:
            await breaker.close(model)
        else:
            # El TTL se renueva en cada barrido, así que un modelo muerto sigue
            # cerrado mientras lo esté, y se reabre solo cuando vuelve.
            await breaker.open(model, open_for_s, reason=f"watchdog: {reason}")

    vivos = [m for m, ok in results.items() if ok]
    log.info("watchdog.sweep", alive=len(vivos), total=len(results), models=vivos)
    return results


async def run_forever(
    registry: BackendRegistry,
    table: RoutingTable,
    breaker: CircuitBreaker,
    *,
    interval_s: int,
) -> None:
    """Bucle de fondo. Un fallo de un barrido no mata el bucle.

    Se sondea al arrancar y después cada `interval_s`: si el gateway levanta con
    una cuenta ya caída, conviene saberlo antes de la primera petición real y no
    con ella.
    """
    open_for_s = max(interval_s * 2, 60)
    while True:
        try:
            await sweep(registry, table, breaker, open_for_s=open_for_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("watchdog.sweep_failed", error=str(exc))
        await asyncio.sleep(interval_s)
