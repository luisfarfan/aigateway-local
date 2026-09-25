"""
Configura el canal de avisos de Telegram en el `.env`, sin exponer el token.

    python scripts/telegram_setup.py           # interactivo: pide el token
    python scripts/telegram_setup.py --check   # prueba el canal ya configurado

Qué hace falta antes, y sólo se hace una vez:

  1. En Telegram, hablale a **@BotFather** → `/newbot` → nombre y usuario.
     Te devuelve un token con forma `123456789:AAH...`.
  2. Buscá tu bot por ese usuario y mandale `/start`. Sin ese primer mensaje
     tuyo, la Bot API no te deja escribirle — y el chat id no existe todavía.

El token NUNCA se imprime ni se registra: se pide sin eco, se valida contra
`getMe`, y va al `.env`, que está en `.gitignore` y queda en 0600.

El chat id se descubre solo leyendo `getUpdates`: pedirlo a mano manda a la
persona a pegar una URL con el token dentro en el navegador, que es la forma
más fácil de que termine en el historial.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from getpass import getpass
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ENV_PATH = REPO / ".env"
_API = "https://api.telegram.org/bot{token}/{method}"


async def _call(token: str, method: str) -> dict:
    """Una llamada a la Bot API. Devuelve `result`, o aborta con el motivo.

    Los errores se traducen porque los de Telegram son crípticos de una forma
    que cuesta caro: un 404 acá casi siempre es un token mal pegado, no un bot
    que no existe, y buscarlo en la dirección equivocada lleva un rato.
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.get(_API.format(token=token, method=method))
        except httpx.HTTPError as exc:
            sys.exit(f"no se pudo hablar con Telegram ({type(exc).__name__})")

    if response.status_code == 401:
        sys.exit("Telegram rechazó el token (401). Revisá que esté pegado entero.")
    if response.status_code == 404:
        sys.exit("404: el token tiene forma inválida. Copialo de nuevo de @BotFather.")
    if response.status_code >= 400:
        sys.exit(f"Telegram respondió {response.status_code}")

    payload = response.json()
    if not payload.get("ok"):
        sys.exit(f"Telegram: {payload.get('description', 'error desconocido')}")
    return payload["result"]


async def discover_chat_id(token: str) -> str | None:
    """El chat de la última persona que le escribió al bot.

    Se toma el ÚLTIMO update y no el primero: si el bot ya se usó antes, los
    viejos pueden ser de otra cuenta, y configurar los avisos para que lleguen
    al chat equivocado es un fallo que no se nota hasta que hace falta.
    """
    updates = await _call(token, "getUpdates")
    for update in reversed(updates):
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat") or {}
        if chat.get("id") is not None:
            quien = chat.get("username") or chat.get("title") or chat.get("first_name") or "?"
            print(f"  chat encontrado: {quien} (tipo {chat.get('type')})")
            return str(chat["id"])
    return None


async def send_test(token: str, chat_id: str) -> bool:
    """Prueba el canal con el mismo adaptador que usa el gateway.

    Con el adaptador y no con un POST suelto a propósito: así la prueba cubre
    también el escapado de MarkdownV2, que es de donde salen los 400 que hacen
    desaparecer un aviso en silencio.
    """
    from src.modules.notifications import TelegramNotifier

    return await TelegramNotifier(bot_token=token, chat_id=chat_id).send(
        "Proxima Gateway: canal configurado",
        "Si ves esto, los avisos del monitor llegan. El primero que vas a "
        "recibir de verdad será si la sesión de la app web de Gemini muere y "
        "el refresco diario no logra recuperarla.",
    )


def write_env(token: str, chat_id: str) -> None:
    """Reescribe las dos claves conservando el resto del archivo."""
    if not ENV_PATH.exists():
        sys.exit(f"no existe {ENV_PATH}")

    texto = ENV_PATH.read_text()
    for clave, valor in (("TELEGRAM_BOT_TOKEN", token), ("TELEGRAM_CHAT_ID", chat_id)):
        linea = f'{clave}="{valor}"'
        if re.search(rf"(?m)^{clave}=", texto):
            texto = re.sub(rf"(?m)^{clave}=.*$", linea, texto)
        else:
            texto = texto.rstrip() + f"\n{linea}\n"
    ENV_PATH.write_text(texto)
    ENV_PATH.chmod(0o600)


async def run_check() -> int:
    from src.core.config import get_settings

    s = get_settings()
    if not s.telegram_bot_token or not s.telegram_chat_id:
        print("no hay canal configurado en .env — corré esto sin --check")
        return 1
    ok = await send_test(s.telegram_bot_token, s.telegram_chat_id)
    print("mensaje de prueba: " + ("ENVIADO" if ok else "FALLÓ (mirá el log)"))
    return 0 if ok else 2


async def run_setup() -> int:
    print(__doc__.split("Qué hace falta antes")[1].split("El token NUNCA")[0].strip())
    print()

    token = getpass("Token de @BotFather (no se muestra al escribir): ").strip()
    if not token:
        print("sin token, nada que hacer", file=sys.stderr)
        return 1

    bot = await _call(token, "getMe")
    print(f"\ntoken válido — bot @{bot.get('username')} (largo del token: {len(token)})")

    print("\nBuscando tu chat. Si no aparece, mandale /start al bot y reintentá.")
    chat_id = await discover_chat_id(token)
    if chat_id is None:
        print(
            f"\nNingún mensaje todavía. Abrí https://t.me/{bot.get('username')}, "
            "mandale /start,\ny volvé a correr esto.",
            file=sys.stderr,
        )
        return 1

    if not await send_test(token, chat_id):
        print("\nNo se pudo entregar el mensaje de prueba. No se escribió nada.", file=sys.stderr)
        return 2

    write_env(token, chat_id)
    print(f"\nescritas en {ENV_PATH} (chmod 600, el token nunca se imprime)")
    print("Aplicar con:  systemctl --user restart aigateway")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Configura los avisos por Telegram.")
    parser.add_argument("--check", action="store_true", help="probar el canal ya configurado")
    args = parser.parse_args()
    return asyncio.run(run_check() if args.check else run_setup())


if __name__ == "__main__":
    raise SystemExit(main())
