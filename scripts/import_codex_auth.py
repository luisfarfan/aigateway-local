"""
Importa a CLIProxyAPI una sesión de Codex que ya está logueada en OTRA máquina.

    # en la PC que tiene el Codex CLI logueado
    python3 import_codex_auth.py --host 192.168.1.12:8417 --key <secret-key>
    python3 import_codex_auth.py --host ... --key ... --dry-run   # solo mostrar

Por qué existe: el login OAuth de CLIProxyAPI no se puede completar desde otra
PC. El `redirect_uri` que se le manda a OpenAI es `http://localhost:1455/auth/
callback`, con el puerto fijo, y los dos saltos del callback los hace el
NAVEGADOR — así que ese `localhost` es la PC del navegador, no el servidor.
Abrir el puerto a la LAN no lo arregla; haría falta un túnel SSH (ver
docs/CLIPROXY-AUTH.md).

Pero si esa PC ya tiene el Codex CLI logueado, el OAuth ya ocurrió: los tokens
están en `~/.codex/auth.json`. Este script los traduce a la forma que guarda
CLIProxyAPI y los sube por la Management API, sin navegador y sin túnel.

Sin dependencias fuera de la stdlib: corre en la otra PC, que no tiene el repo.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_AUTH = Path.home() / ".codex" / "auth.json"

# Claim donde el id_token de OpenAI cuelga la identidad de ChatGPT.
_OPENAI_AUTH_CLAIM = "https://api.openai.com/auth"


def jwt_claims(token: str) -> dict:
    """Payload de un JWT, sin verificar firma.

    No se valida a propósito: acá el token no se está aceptando como prueba de
    nada, sólo se le leen los campos para nombrar el archivo. Quien lo verifica
    es OpenAI cuando CLIProxyAPI lo use.
    """
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # el base64url del JWT va sin padding
        return json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError, json.JSONDecodeError):
        return {}


def to_cliproxy(raw: dict) -> dict:
    """Traduce `~/.codex/auth.json` a la forma que CLIProxyAPI guarda en disco.

    El Codex CLI anida los tokens bajo `tokens`; CLIProxyAPI los quiere planos y
    con `type`, que es el campo por el que decide qué proveedor es.
    """
    tokens = raw.get("tokens") or {}
    id_token = tokens.get("id_token") or ""
    claims = jwt_claims(id_token)
    openai_auth = claims.get(_OPENAI_AUTH_CLAIM) or {}

    access_token = tokens.get("access_token") or ""
    # `expired` sale del `exp` del access_token: es el único sitio donde está la
    # verdad. Si no se puede leer, se deja vencido — CLIProxyAPI lo refresca con
    # el refresh_token en la primera llamada, que es el fallo barato de los dos.
    exp = jwt_claims(access_token).get("exp")
    expired = (
        datetime.fromtimestamp(int(exp), UTC).isoformat()
        if exp
        else datetime.now(UTC).isoformat()
    )

    return {
        "type": "codex",
        "email": claims.get("email") or "",
        "account_id": openai_auth.get("chatgpt_account_id")
        or tokens.get("account_id")
        or "",
        "access_token": access_token,
        "id_token": id_token,
        "refresh_token": tokens.get("refresh_token") or "",
        "last_refresh": raw.get("last_refresh") or datetime.now(UTC).isoformat(),
        "expired": expired,
        "disabled": False,
    }


def filename_for(entry: dict) -> str:
    """`codex-<8 hex>-<email>-<plan>.json`, el patrón que usa CLIProxyAPI.

    El nombre es etiqueta, no identidad: el proveedor se decide por el campo
    `type` de adentro. Se respeta el patrón igual para que el listado del panel
    se lea como el resto.
    """
    claims = jwt_claims(entry["id_token"])
    plan = (claims.get(_OPENAI_AUTH_CLAIM) or {}).get("chatgpt_plan_type") or "unknown"
    short = (entry["account_id"] or "00000000").replace("-", "")[:8]
    return f"codex-{short}-{entry['email'] or 'sin-email'}-{plan}.json"


def upload(host: str, key: str, name: str, entry: dict) -> None:
    url = f"http://{host}/v0/management/auth-files?name={urllib.parse.quote(name)}"
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(entry).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"{resp.status} {resp.read().decode().strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="host:puerto del panel, ej 192.168.1.12:8417")
    parser.add_argument("--key", required=True, help="remote-management.secret-key")
    parser.add_argument("--auth", type=Path, default=DEFAULT_AUTH, help=f"por defecto {DEFAULT_AUTH}")
    parser.add_argument("--dry-run", action="store_true", help="mostrar qué se subiría, sin subir")
    args = parser.parse_args()

    if not args.auth.exists():
        print(f"No existe {args.auth}. ¿El Codex CLI está logueado en esta PC?", file=sys.stderr)
        return 1

    entry = to_cliproxy(json.loads(args.auth.read_text()))
    if not entry["refresh_token"]:
        print(
            f"{args.auth} no trae refresh_token. Sin él la credencial muere al vencer "
            "el access_token; correr `codex login` de nuevo.",
            file=sys.stderr,
        )
        return 1

    name = filename_for(entry)
    print(f"archivo:  {name}")
    print(f"email:    {entry['email']}")
    print(f"account:  {entry['account_id']}")
    print(f"expira:   {entry['expired']}")

    if args.dry_run:
        print("\n--dry-run: no se subió nada.")
        return 0

    try:
        upload(args.host, args.key, name, entry)
    except urllib.error.HTTPError as exc:
        print(f"Falló la subida: {exc.code} {exc.read().decode()}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"No se pudo alcanzar {args.host}: {exc}", file=sys.stderr)
        return 1

    print(f"\nListo. Verificar en http://{args.host}/management.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
