"""
El canal de avisos.

Lo que importa acá es que **avisar no pueda romper nada**. El aviso es sobre un
incidente, no es el trabajo: si Telegram está caído, eso no puede además tumbar
el bucle que vigila.
"""

from __future__ import annotations

import httpx
import pytest

from src.modules.notifications import NullNotifier, TelegramNotifier
from src.modules.notifications.telegram import _escape


@pytest.mark.asyncio
async def test_sin_canal_configurado_no_rompe_nada():
    assert await NullNotifier().send("t", "b") is False


@pytest.mark.asyncio
async def test_sin_token_no_intenta_salir_a_la_red():
    assert await TelegramNotifier(bot_token="", chat_id="").send("t", "b") is False


@pytest.mark.asyncio
async def test_un_canal_caido_devuelve_false_en_vez_de_levantar(monkeypatch):
    """Si esto propagara, una caída de Telegram mataría el monitor y nos
    quedaríamos sin vigilancia justo por intentar avisar."""

    class ClienteRoto:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, *a, **kw):
            raise httpx.ConnectError("sin red")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: ClienteRoto())
    assert await TelegramNotifier(bot_token="t", chat_id="c").send("t", "b") is False


def test_se_escapan_los_caracteres_que_rechaza_markdownv2():
    """Un guion o un punto sin escapar hacen que Telegram devuelva 400 y el
    aviso se pierda — justo el mensaje que más importaba entregar."""
    assert _escape("v2.1-beta (ok)") == r"v2\.1\-beta \(ok\)"
