# F0 — Qué sabe hacer CLIProxyAPI vanilla

> Medido contra `agl_cliproxy` (imagen `eceasy/cli-proxy-api@sha256:c6fef087…`,
> v7.2.143, commit `4b5f1ea` = upstream HEAD). **Sin parches del fork.**
> Cuentas: codex `lucho.farfan9` (free) + antigravity ×2.
> Fecha: 2026-08-27

Objetivo: decidir qué tiene que hacer `translate.py` y qué se puede delegar al binario.

## Resultados

| Ruta | Vanilla | Evidencia |
|---|---|---|
| `POST /v1/chat/completions` — chat plano | ✅ | `gpt-5.4-mini` → `PONG` |
| `POST /v1/responses` + `web_search_preview` | ✅ | `output[].type = ['web_search_call','message']`, precio real + URL |
| `POST /v1/chat/completions` + `tools:[{"type":"web_search"}]` (forma Gemini) | ❌ | el modelo responde *"No tengo acceso a información en tiempo real"* — el tool block se descarta |
| `POST /v1beta/models/{model}:generateContent` + `tools:[{"googleSearch":{}}]` | ✅ | grounding real **y `groundingMetadata` presente** |
| Imagen Gemini vía chat → `message.images[0].image_url.url` | ✅ | data URI `data:image/jpeg;base64,…` |
| `POST /v1/images/generations` con `gpt-image-2` | ❌ | `auth_not_found: no auth available (providers=codex, model=gpt-image-2)` — la cuenta Codex es *free*; requiere Plus |
| Anthropic nativo (`web_search_20250305` + `anthropic-beta`) | ⬜ | sin cuenta Anthropic conectada |

Contraste: la misma llamada de websearch-Gemini por `/v1/chat/completions` **sí** funciona
contra la instancia del fork parcheado (`make-montages-cliproxy`, build del 7-ago), que
citó `fifa.com`. O sea, el parche Go hace trabajo real en esa ruta.

## Consecuencias para `translate.py`

1. **El websearch de Gemini va por la superficie nativa, no por la OpenAI-compat.**
   `/v1beta/models/{model}:generateContent` con `tools:[{"googleSearch":{}}]`. Es lo que
   el parche del fork emula desde el path OpenAI; desde Python se llega directo, sin
   parche. **La decisión de §3 del plan (vanilla + Python) queda validada por medición.**

2. **`groundingMetadata` es la fuente de las fuentes.** El path OpenAI-compat lo descarta,
   y por eso `WebSearchResult.sources` de intel-v2 sale siempre vacío (auditoría §2.1).
   Yendo por la superficie nativa el campo llega entero. El bug no se "arregla": desaparece
   porque cambiamos de superficie.

3. **Codex no necesita nada.** `/v1/responses` + `web_search_preview` funciona tal cual;
   `translate.py` sólo enruta y normaliza la respuesta a forma OpenAI.

4. **Imagen Gemini tampoco necesita parche.** Sale por `/v1/chat/completions` en
   `message.images[0].image_url.url` como data URI, que es justo el
   `ImageGenEndpoint.CHAT_IMAGES` que intel-v2 ya tenía documentado.

5. **La superficie interna del gateway no puede ser sólo OpenAI-compat.** Al menos
   websearch-Gemini exige hablar Gemini nativo aguas arriba. El contrato hacia afuera
   sigue siendo `/v1/*`; la traducción es interna.

## Huecos de cobertura para los fixtures

| Falta | Qué desbloquea | Cómo |
|---|---|---|
| Cuenta **Anthropic** | fixture del bloque `web_search_20250305` + header `anthropic-beta`, y del path Claude nativo | login en el panel |
| Cuenta **Codex Plus** | fixture de `gpt-image-2` por `/v1/images/generations` | la conectada es *free*; intel-v2 tiene una `-plus` |
| Cuenta **Gemini directa** (no antigravity) | confirmar que el path nativo se comporta igual sin pasar por antigravity | login en el panel |

Los modelos Claude que expone antigravity (`claude-sonnet-4-6`) **no sirven** para este
fixture: llegan por Vertex (`req_vrtx_…`) y no ejercitan el header de Anthropic.

## Nota de diseño para F4

`GET /v1/models` **no es prueba de vida**. En la instancia de intel-v2 lista 62 modelos, de
los cuales anthropic (13) responde `OAuth access token has been revoked` y google (19)
responde `403 PERMISSION_DENIED`. Un selector que filtre por esa lista — como el de
make-montages — elegiría un modelo muerto. El watchdog tiene que **probar**, no listar.
