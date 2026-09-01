"""
Adaptador de Telegram. Un POST a la Bot API, sin dependencias nuevas.

Se eligió Telegram sobre correo por latencia y sobre ntfy por privacidad: los
topics públicos de ntfy los lee cualquiera que adivine el nombre. Un bot es
privado y llega al teléfono en segundos, que es lo que hace falta cuando lo que
se avisa exige ir a hacer algo a mano.
"""

from __future__ import annotations

import httpx
import structlog

log = structlog.get_logger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"

# Telegram corta en 4096 caracteres. Se recorta antes para que un cuerpo largo
# no convierta el aviso en un 400 y se pierda del todo.
_MAX_BODY = 3500


class TelegramNotifier:
    """Envía por un bot. No levanta nunca: devuelve `False` y lo registra."""

    def __init__(self, *, bot_token: str, chat_id: str, timeout_s: float = 15.0) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout_s

    @property
    def name(self) -> str:
        return "telegram"

    async def send(self, title: str, body: str) -> bool:
        if not self._token or not self._chat_id:
            log.warning("telegram.not_configured")
            return False

        text = f"*{_escape(title)}*\n\n{_escape(body[:_MAX_BODY])}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    _API.format(token=self._token),
                    json={
                        "chat_id": self._chat_id,
                        "text": text,
                        "parse_mode": "MarkdownV2",
                        # El aviso ya trae todo lo que hace falta; una tarjeta de
                        # previsualización sólo agrega ruido.
                        "disable_web_page_preview": True,
                    },
                )
        except httpx.HTTPError as exc:
            # El token NUNCA entra al log: la URL lo lleva embebido, así que se
            # registra el tipo de fallo y no la excepción cruda.
            log.error("telegram.send_failed", error=type(exc).__name__)
            return False

        if response.status_code >= 400:
            log.error("telegram.send_rejected", status=response.status_code)
            return False
        return True


def _escape(text: str) -> str:
    """Escapa lo que MarkdownV2 considera especial.

    Sin esto, un mensaje con un guion o un punto —o sea, casi cualquiera— se
    rechaza con un 400 y el aviso se pierde justo cuando importaba.
    """
    for char in r"_*[]()~`>#+-=|{}.!":
        text = text.replace(char, f"\\{char}")
    return text
