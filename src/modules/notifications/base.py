"""
El puerto de notificación, y su implementación vacía.

Existe porque hay fallos que **ningún reintento arregla**: una cookie de sesión
que expiró necesita que una persona abra un navegador. El circuit breaker sabe
sacar de la cadena al backend caído —eso protege la latencia— pero nadie se
entera de que hay que ir a arreglarlo hasta que alguien mira Grafana.

Es un puerto y no una llamada directa a Telegram por la razón de siempre: el
canal es un detalle de despliegue. Hoy es un bot; mañana puede ser ntfy, un
webhook o un correo, y nada del código que detecta el problema debería cambiar
por eso.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)


@runtime_checkable
class Notifier(Protocol):
    """Manda un aviso al operador. Nunca levanta."""

    async def send(self, title: str, body: str) -> bool:
        """`True` si se entregó. Un fallo se registra, no se propaga.

        Que no se propague es deliberado: el aviso es *sobre* un incidente, no
        es el trabajo. Si el canal está caído, eso no puede tumbar además el
        bucle que vigila.
        """
        ...


class NullNotifier:
    """Sin canal configurado: se registra y se sigue.

    Es el default para que el monitor funcione sin exigir un bot. Sin esto,
    "no configuré Telegram" sería un `if notifier is not None` repartido por
    cada sitio que avisa.
    """

    @property
    def name(self) -> str:
        return "null"

    async def send(self, title: str, body: str) -> bool:
        log.info("notify.skipped", reason="sin canal configurado", title=title, body=body[:200])
        return False
