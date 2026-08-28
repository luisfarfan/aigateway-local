"""
Record real CLIProxyAPI responses as offline test fixtures (F0).

Every translation route the gateway will implement is exercised against a live,
authenticated CLIProxyAPI instance, and the request/response pair is written to
`tests/fixtures/cliproxy/<case>.json`. Tests then replay those files instead of
calling the network, so the translation layer can be developed and refactored
without burning provider quota or depending on which accounts happen to be
logged in that day.

Negative cases are recorded on purpose: a route that vanilla CLIProxyAPI does
NOT support (the Gemini web_search tool block on /v1/chat/completions) is as
much a contract as a working one — it is precisely why translate.py has to
reach for the Gemini-native surface instead.

Usage:
    export CLIPROXY_BASE_URL=http://127.0.0.1:8417
    export CLIPROXY_API_KEY=...          # from docker/cliproxy/config.yaml
    python scripts/record_fixtures.py               # all cases
    python scripts/record_fixtures.py chat_plain    # one case

Secrets never reach disk: the Authorization header is not recorded, and any
base64 blob is replaced by its prefix, length and SHA-256 so fixtures stay small
and diffable while tests can still assert on shape and identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "cliproxy"

# Strings longer than this are treated as blobs (base64 images, data URIs).
BLOB_THRESHOLD = 512
BLOB_PREFIX_CHARS = 64

TIMEOUT_SECONDS = 120.0


@dataclass
class Case:
    """One recorded route.

    `expect` documents the intent, so a fixture that flips from `ok` to `error`
    (or back) is a loud signal that upstream behaviour changed, not a silent
    re-record.
    """

    name: str
    method: str
    path: str
    body: dict[str, Any] | None
    why: str
    expect: str = "ok"  # "ok" | "error"
    requires: list[str] = field(default_factory=list)


CASES: list[Case] = [
    Case(
        name="models_list",
        method="GET",
        path="/v1/models",
        body=None,
        why="Inventario de modelos. Nota de diseño: listar != estar vivo, ver "
        "docs/F0-VANILLA-CAPABILITIES.md.",
    ),
    Case(
        name="chat_plain",
        method="POST",
        path="/v1/chat/completions",
        body={
            "model": "gpt-5.4-mini",
            "messages": [{"role": "user", "content": "Responde solo con la palabra PONG."}],
            "max_tokens": 20,
        },
        why="Superficie base OpenAI-compatible: forma de choices/usage que el "
        "gateway debe devolver hacia afuera.",
        requires=["codex"],
    ),
    Case(
        name="websearch_codex_responses",
        method="POST",
        path="/v1/responses",
        body={
            "model": "gpt-5.4-mini",
            "input": "Precio actual del Bitcoin. Cita la URL de la fuente.",
            "tools": [{"type": "web_search_preview"}],
        },
        why="Websearch de Codex: endpoint y forma de respuesta distintos "
        "(output[] con web_search_call + message). Funciona en vanilla.",
        requires=["codex"],
    ),
    Case(
        name="websearch_gemini_openai_compat_UNSUPPORTED",
        method="POST",
        path="/v1/chat/completions",
        body={
            "model": "gemini-3-flash",
            "messages": [{"role": "user", "content": "Precio actual del Bitcoin. Cita la URL."}],
            "max_tokens": 250,
            "tools": [{"type": "web_search"}],
        },
        why="NEGATIVO a propósito: vanilla descarta el tool block y el modelo "
        "responde sin buscar. Justifica usar la superficie Gemini-nativa.",
        expect="ok",  # HTTP 200, pero sin grounding — el fallo es semántico
        requires=["antigravity"],
    ),
    Case(
        name="websearch_gemini_native",
        method="POST",
        path="/v1beta/models/gemini-3-flash:generateContent",
        body={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Precio actual del Bitcoin. Cita la URL de la fuente."}],
                }
            ],
            "tools": [{"googleSearch": {}}],
        },
        why="Websearch de Gemini que SI funciona en vanilla. Trae "
        "groundingMetadata — de ahi salen las fuentes que el path "
        "OpenAI-compatible descarta.",
        requires=["antigravity"],
    ),
    Case(
        name="image_gemini_chat",
        method="POST",
        path="/v1/chat/completions",
        body={
            "model": "gemini-3.1-flash-image",
            "messages": [{"role": "user", "content": "Genera una imagen: un cubo rojo simple"}],
            "max_tokens": 100,
        },
        why="Imagen por Gemini: llega en message.images[0].image_url.url como "
        "data URI, no por /v1/images/generations.",
        requires=["antigravity"],
    ),
    Case(
        name="image_openai_generations_NOAUTH",
        method="POST",
        path="/v1/images/generations",
        body={
            "model": "gpt-image-2",
            "prompt": "un cubo rojo simple sobre fondo blanco",
            "response_format": "b64_json",
            "size": "1024x1024",
            "quality": "low",
        },
        why="NEGATIVO: forma exacta del error cuando ninguna credencial cubre "
        "el modelo. El routing (F4) tiene que distinguirlo de un fallo de red.",
        expect="error",
        requires=["codex-plus"],
    ),
]


def _sanitize(value: Any) -> Any:
    """Replace blobs by a fingerprint, recursively.

    Keeps fixtures small and diffable while preserving everything a test needs
    to assert: the prefix (so `data:image/jpeg;base64,` stays visible), the
    original length, and a digest that pins the exact bytes.
    """
    if isinstance(value, str):
        if len(value) <= BLOB_THRESHOLD:
            return value
        digest = hashlib.sha256(value.encode()).hexdigest()
        return {
            "__blob__": True,
            "prefix": value[:BLOB_PREFIX_CHARS],
            "length": len(value),
            "sha256": digest,
        }
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def record(case: Case, client: httpx.Client, base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{case.path}"
    response = client.get(url) if case.method == "GET" else client.post(url, json=case.body)

    try:
        payload: Any = response.json()
    except ValueError:
        payload = {"__non_json_body__": response.text[:2000]}

    return {
        "case": case.name,
        "why": case.why,
        "expect": case.expect,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "request": {"method": case.method, "path": case.path, "body": case.body},
        "response": {"status": response.status_code, "body": _sanitize(payload)},
    }


def _verdict(fixture: dict[str, Any]) -> str:
    """`ok` / `error` actually observed, independent of what the case expected."""
    body = fixture["response"]["body"]
    if fixture["response"]["status"] >= 400:
        return "error"
    if isinstance(body, dict) and body.get("error"):
        return "error"
    return "ok"


def main(argv: list[str]) -> int:
    base_url = os.environ.get("CLIPROXY_BASE_URL", "http://127.0.0.1:8417")
    api_key = os.environ.get("CLIPROXY_API_KEY", "")
    if not api_key:
        print("CLIPROXY_API_KEY no está definida (mírala en docker/cliproxy/config.yaml)")
        return 2

    selected = set(argv[1:])
    cases = [c for c in CASES if not selected or c.name in selected]
    if not cases:
        print(f"Ningún caso coincide con {sorted(selected)}")
        return 2

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    mismatches = 0
    with httpx.Client(timeout=TIMEOUT_SECONDS, headers=headers) as client:
        for case in cases:
            print(f"→ {case.name} ... ", end="", flush=True)
            try:
                fixture = record(case, client, base_url)
            except httpx.HTTPError as exc:
                print(f"FALLO DE TRANSPORTE: {exc}")
                mismatches += 1
                continue

            got = _verdict(fixture)
            path = FIXTURE_DIR / f"{case.name}.json"
            path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")

            flag = "" if got == case.expect else f"  ← esperaba {case.expect}, obtuvo {got}"
            if flag:
                mismatches += 1
            print(f"{fixture['response']['status']} {got}{flag}")

    print(f"\nFixtures en {FIXTURE_DIR}")
    if mismatches:
        print(f"{mismatches} caso(s) no coinciden con lo esperado — revisar antes de commitear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
