"""
Rehace la sesión de la app web de Gemini en el `.env`, sin exponer la cookie.

    python scripts/gemini_web_login.py              # extrae del navegador
    python scripts/gemini_web_login.py --check      # sólo dice si la actual vive
    python scripts/gemini_web_login.py --psid X --psidts Y   # pegándolas a mano

Por qué existe: el backend `geminiweb/` se autentica con la cookie de sesión de
`gemini.google.com`, no con una API key. La librería rota `__Secure-1PSIDTS`
sola cada pocos minutos, así que el vencimiento corto está cubierto — pero
cuando la sesión muere de verdad (cerraste sesión, cambiaste la contraseña,
Google la invalidó) **no hay forma de recuperarla sin un navegador**. Eso es lo
que avisa el monitor, y esto es lo que se corre después.

Los valores NUNCA se imprimen: sólo si se encontraron y su longitud. Van al
`.env`, que está en `.gitignore`.

Requiere `browser-cookie3` y, en Chrome bajo Linux, el llavero DESBLOQUEADO —
correrlo desde la terminal del escritorio, no por SSH.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ENV_PATH = REPO / ".env"
WANTED = ("__Secure-1PSID", "__Secure-1PSIDTS")


def from_browser() -> dict[str, str]:
    """Cookies del primer navegador que tenga las dos.

    Se prueban varios porque el navegador donde la persona inició sesión no es
    predecible, y fallar con "no encontré nada" cuando estaba en Firefox sería
    mandarla a buscar el problema donde no está.
    """
    try:
        import browser_cookie3 as bc
    except ImportError:
        sys.exit("falta browser-cookie3:  pip install browser-cookie3")

    parciales: dict[str, str] = {}
    for nombre, fn in (
        ("chrome", bc.chrome),
        ("firefox", bc.firefox),
        ("chromium", bc.chromium),
        ("edge", bc.edge),
        ("brave", bc.brave),
    ):
        try:
            jar = fn(domain_name="google.com")
        except Exception as exc:  # noqa: BLE001
            print(f"  {nombre:9} no disponible ({str(exc)[:60]})")
            continue
        got = {c.name: c.value for c in jar if c.name in WANTED and c.value}
        print(f"  {nombre:9} encontradas: {sorted(got) or 'ninguna'}")
        if all(k in got for k in WANTED):
            print(f"  -> usando {nombre}")
            return got
        parciales = parciales or got
    return parciales


def write_env(cookies: dict[str, str]) -> None:
    """Reescribe las claves en `.env` conservando el resto del archivo."""
    if not ENV_PATH.exists():
        sys.exit(f"no existe {ENV_PATH}")

    texto = ENV_PATH.read_text()
    for clave, valor in (
        ("GEMINI_WEB_SECURE_1PSID", cookies["__Secure-1PSID"]),
        ("GEMINI_WEB_SECURE_1PSIDTS", cookies["__Secure-1PSIDTS"]),
    ):
        linea = f'{clave}="{valor}"'
        if re.search(rf"(?m)^{clave}=", texto):
            texto = re.sub(rf"(?m)^{clave}=.*$", linea, texto)
        else:
            texto = texto.rstrip() + f"\n{linea}\n"
    if not re.search(r"(?m)^ENABLE_BACKEND_GEMINI_WEB=", texto):
        texto = texto.rstrip() + "\nENABLE_BACKEND_GEMINI_WEB=true\n"
    ENV_PATH.write_text(texto)
    ENV_PATH.chmod(0o600)


async def check(psid: str, psidts: str) -> bool:
    """Valida contra el MISMO caché de cookies que usa el gateway.

    Sin esto la librería cae a su default en `/tmp`, donde quedan sesiones
    viejas de barridos anteriores — y entonces este script valida una
    credencial distinta de la que el gateway está usando. Pasó: reportó
    "sesión MUERTA" con el gateway funcionando perfecto.
    """
    from src.core.config import get_settings
    from src.modules.backends.gemini_web import GeminiWebBackend

    backend = GeminiWebBackend(
        secure_1psid=psid,
        secure_1psidts=psidts,
        cookie_cache_dir=str(Path(get_settings().gemini_web_cookie_cache_dir).expanduser()),
    )
    alive, detalle = await backend.check_session()
    print(f"\nsesión: {'VIVA' if alive else 'MUERTA'} — {detalle}")
    await backend.aclose()
    return alive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="sólo comprobar la del .env")
    parser.add_argument("--psid", help="__Secure-1PSID, si se pega a mano")
    parser.add_argument("--psidts", help="__Secure-1PSIDTS, si se pega a mano")
    args = parser.parse_args()

    if args.check:
        from src.core.config import get_settings

        s = get_settings()
        if not s.gemini_web_secure_1psid:
            print("no hay cookie configurada en .env")
            return 1
        return (
            0 if asyncio.run(check(s.gemini_web_secure_1psid, s.gemini_web_secure_1psidts)) else 2
        )

    if args.psid and args.psidts:
        cookies = {"__Secure-1PSID": args.psid, "__Secure-1PSIDTS": args.psidts}
    else:
        print("Buscando la sesión de Gemini en los navegadores locales…")
        cookies = from_browser()

    faltan = [k for k in WANTED if k not in cookies]
    if faltan:
        print(
            f"\nNo se encontraron: {faltan}\n"
            "Entrá a https://gemini.google.com en el navegador, comprobá que la\n"
            "sesión esté realmente iniciada, y volvé a correr esto. Si usás Chrome\n"
            "en Linux, tiene que ser desde la terminal del escritorio (el llavero\n"
            "debe estar desbloqueado), no por SSH.",
            file=sys.stderr,
        )
        return 1

    print(f"\nencontradas (largos: {[len(cookies[k]) for k in WANTED]})")
    if not asyncio.run(check(cookies["__Secure-1PSID"], cookies["__Secure-1PSIDTS"])):
        print("\nLa cookie nueva TAMPOCO valida. No se escribió nada.", file=sys.stderr)
        return 2

    write_env(cookies)
    print(f"\nescritas en {ENV_PATH} (chmod 600, nunca impresas)")
    print("Aplicar con:  systemctl --user restart aigateway")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
