"""
Vigila la cookie de sesión de la app web de Gemini y avisa cuando muere.

Por qué hace falta un vigilante propio y no basta el watchdog: el watchdog abre
el circuito del modelo que falla, y eso protege la latencia de las peticiones
—la cadena salta al siguiente candidato— pero **no le dice a nadie que hay que
ir a arreglarlo**. Con una API key eso da igual, porque nada se arregla solo
tampoco. Acá no: esta credencial es una sesión de navegador que expira, y
recuperarla exige que una persona abra Chrome y vuelva a entrar.

Sin esto, el modo de fallo es silencioso: la cookie muere, el breaker saca el
backend, y la ruta de imagen se queda sin su último recurso durante días sin que
nadie lo note — hasta el día que además caigan los otros dos y no haya fondo.

**Aviso por flanco, no por estado.** Se notifica en la transición viva → muerta,
y otra vez en muerta → viva. Avisar en cada chequeo convertiría la alerta en
ruido, y una alerta ruidosa se ignora, que es peor que no tenerla.
"""

from __future__ import annotations

import asyncio

import structlog

from src.core import metrics
from src.core.redis import get_redis
from src.modules.notifications.base import Notifier

log = structlog.get_logger(__name__)

# El estado vive en Redis y no en memoria del proceso porque API y worker son
# procesos distintos: con el flanco en memoria, cada uno avisaría por su cuenta
# del mismo incidente, y un reinicio del gateway volvería a avisar de algo ya
# sabido.
_STATE_KEY = "proxima:geminiweb:session_alive"


async def check_once(backend, notifier: Notifier) -> bool:
    """Un chequeo. Devuelve si la sesión está viva.

    Toda la lógica de "¿esto es novedad?" vive acá para que el bucle sea trivial
    y se pueda probar el flanco sin esperar intervalos.
    """
    alive, detail = await backend.check_session()

    metrics.gemini_web_session_valid.set(1 if alive else 0)
    metrics.gemini_web_session_checks_total.labels(outcome="ok" if alive else "expired").inc()

    previous = await _previous_state()
    await _remember(alive)

    if previous is None:
        # Primer chequeo tras arrancar: se registra el estado pero no se avisa.
        # Notificar acá haría que cada reinicio del gateway mande un mensaje.
        log.info("gemini_web.session_check", alive=alive, detail=detail, first=True)
        return alive

    if previous == alive:
        return alive

    if alive:
        # Cerrar el circuito a mano y no esperar a que venza: la sesión volvió
        # porque una persona la arregló, y hacerle esperar horas a que expire un
        # cooldown de 6 h sería castigar el arreglo.
        await _close_circuit()
        await notifier.send(
            "Gemini web: sesión recuperada",
            "La cookie volvió a funcionar. El backend geminiweb/ está otra vez "
            "disponible como último recurso de la ruta de imagen.",
        )
        log.info("gemini_web.session_recovered")
    else:
        await notifier.send(
            "Gemini web: sesión EXPIRADA",
            "La cookie de gemini.google.com dejó de funcionar, así que el "
            "backend geminiweb/ ya no responde. No se recupera solo: hace falta "
            "entrar a gemini.google.com en el navegador y luego correr\n\n"
            "  python scripts/gemini_web_login.py\n\n"
            f"Detalle del upstream: {detail}",
        )
        log.warning("gemini_web.session_expired", detail=detail)

    return alive


async def _close_circuit() -> None:
    """Devuelve el backend a la cadena en cuanto la sesión vuelve."""
    from src.modules.backends.gemini_web import CHAIN_MODEL_ID
    from src.modules.routing.breaker import CircuitBreaker
    from src.modules.routing.config import load_routing

    try:
        await CircuitBreaker(load_routing().breaker).close(CHAIN_MODEL_ID)
        log.info("gemini_web.circuit_closed", model=CHAIN_MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        # No poder cerrarlo no es motivo para perder el aviso de recuperación:
        # el circuito vence solo, sólo que más tarde.
        log.warning("gemini_web.circuit_close_failed", error=str(exc)[:150])


async def _previous_state() -> bool | None:
    """Último estado conocido, o `None` si no hay ninguno."""
    try:
        raw = await get_redis().get(_STATE_KEY)
    except Exception as exc:  # noqa: BLE001
        # Sin Redis se pierde el flanco, no el chequeo. Se devuelve None, que
        # hace que este chequeo no avise: mejor callar una vez que gritar en
        # cada barrido porque el estado no se pudo leer.
        log.warning("gemini_web.state_read_failed", error=str(exc)[:120])
        return None
    if raw is None:
        return None
    return raw in (b"1", "1")


async def _remember(alive: bool) -> None:
    try:
        await get_redis().set(_STATE_KEY, "1" if alive else "0")
    except Exception as exc:  # noqa: BLE001
        log.warning("gemini_web.state_write_failed", error=str(exc)[:120])


async def run_forever(backend, notifier: Notifier, *, interval_s: int) -> None:
    """Bucle de fondo. Un chequeo que falla no mata el bucle.

    Mismo patrón que el watchdog de routing: si esto muriera con la primera
    excepción, el gateway seguiría vivo y sin vigilancia, que es el peor de los
    dos mundos — parece cubierto y no lo está.
    """
    while True:
        try:
            await check_once(backend, notifier)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            metrics.gemini_web_session_checks_total.labels(outcome="error").inc()
            log.error("gemini_web.monitor_failed", error=str(exc)[:200])
        await asyncio.sleep(interval_s)
