# F0 — Auditoría de las 3 implementaciones de CLIProxyAPI

> Insumo para diseñar el módulo único. Sólo lectura, nada modificado.
> Fecha: 2026-08-26

Fuentes:
- **A** = `proxima-intelligence-v2/src/proxima_intelligence/llm/` (async, httpx.AsyncClient)
- **B** = `make-montages/src/make_montages/infrastructure/cliproxyapi_*.py` (**sync**, httpx.Client)
- **C** = `aigateway-local/src/modules/providers/` (async, no habla con cloud)

---

## 1. Matriz de capacidades

| Capacidad | A (intel-v2) | B (make-montages) | Gana |
|---|---|---|---|
| Chat completions | ✅ | ✅ | empate |
| Websearch Gemini/Claude | ✅ | ✅ | empate |
| Websearch Codex (`/v1/responses`) | ✅ | ✅ | empate |
| **Extracción de fuentes del websearch** | ❌ declara `sources` y nunca lo llena | ✅ `_extract_inline_sources` → `SearchSource` | **B** |
| Visión | ✅ (por `chat_completion`) | ✅ adapter dedicado | B (más explícito) |
| Generación de imagen | ✅ 2 endpoints + registry con tokens medidos | ✅ sólo `/v1/images/generations` | **A** |
| Embeddings | ✅ `embeddings_client.py` | ❌ | **A** |
| Strict JSON Schema (OpenAI) | ❌ | ✅ `to_strict_json_schema` recursivo | **B** |
| Stripping de code fences | ❌ | ✅ | **B** |
| **Schema completo inyectado en la conversación** | ❌ sólo `_schema_description` (7 líneas) | ✅ `_schema_instruction` + `_describe_type` con valores de enum | **B** |
| Reparación tras fallo de validación | ✅ 3 intentos: call → repair → strict-retry | ✅ 1 repair retry | **A** |
| Log persistente por intento | ✅ `llm_call_logs` en Postgres | ❌ | **A** |
| Cache | ✅ Redis, TTL por `search_type` | ❌ | **A** |
| Selección de modelo | push: watchdog 30min → Redis | pull: `GET /v1/models` en vivo + preferencias por capability | **híbrido** |
| Errores tipados retryables | ✅ 401/403/408/429/5xx → `LLMTransportError` | ❌ todo `httpx.HTTPError` → `LLMGatewayError` fatal | **A** |
| Truncado de blobs en logs/errores | ❌ | ✅ `truncated_repr` (base64 multi-KB) | **B** |
| Async | ✅ | ❌ sync | **A** |

Ninguna de las dos domina. El módulo único es **A ∪ B**, no "adoptar A" ni "adoptar B".

---

## 2. Bugs y trampas encontradas

### 2.1 `sources` siempre vacío en intel-v2 — bug real
`llm/websearch.py`: `WebSearchResult` declara `sources: list[str] = field(default_factory=list)`,
pero ni `_gemini_search` ni `_codex_search` lo pueblan. Todo consumidor de websearch en
intel-v2 recibe la lista vacía. B sí extrae las fuentes inline.

### 2.2 `cost_usd` siempre 0.0 en make-montages
`cliproxyapi_adapter.py` lee `usage_raw.get("cost_usd", 0.0)` del body de CLIProxyAPI.
CLIProxyAPI **no devuelve ese campo**. El costo tiene que calcularse en el gateway
contra `pricing.yaml`, nunca leerse del upstream.

### 2.3 Detección de familia de proveedor duplicada 3 veces, con reglas distintas
- `guard.py:77` `_infer_provider(model)`
- `websearch.py` `is_codex_model(model)` → `gpt-*` o `codex-*`
- `search_adapter.py:89` `_infer_provider(model)`
- `constants.py` `websearch_tool_block_for(model)` → `claude*` vs resto

Cuatro sitios decidiendo lo mismo por prefijos de string. Una tabla de familias única.

### 2.4 `base_url` con y sin `/v1` — trampa de migración
- B: `DEFAULT_BASE_URL = "http://localhost:8317/v1"` y luego `f"{base_url}/chat/completions"`
- A: `cliproxy_base_url` sin `/v1`, y pega `"/v1/chat/completions"`

Al apuntar ambos al gateway, uno de los dos rompe silenciosamente con 404. El gateway
debe **aceptar las dos formas** (montar el router en `/v1` y tolerar `/v1/v1` con un
redirect) o la migración se cae en la primera request.

### 2.5 Errores no retryables en make-montages
B convierte cualquier `httpx.HTTPError` en `LLMGatewayError` fatal. Un 429 de Gemini
mata el job. A ya aprendió esto y clasifica 401/403/408/429/5xx como retryables
(comentario en `cliproxy_client.py`: "Gemini's documented auth_unavailable under
concurrent load"). El routing del gateway hereda la clasificación de A.

---

## 3. La pieza más valiosa: el prompting del schema (B)

`_schema_instruction` + `_describe_fields` + `_describe_type` + `_resolve_ref` resuelven
un problema que A tiene y no ha detectado:

> `response_format: {type: json_schema, strict: true}` lo **descartan** los proveedores
> no-OpenAI detrás de CLIProxyAPI (Gemini/Claude). El modelo nunca ve el schema, así que
> inventa nombres de campo. Único sitio donde queda ponerlo: la conversación misma.

Y `_describe_type` renderiza **los valores reales del enum**, no la palabra "enum" —
finding de instancia real: un modelo respondía `'Technology'` cuando el contrato decía
`science_tech`.

A inyecta `_schema_description` (7 líneas) y confía en `response_format`. Con Gemini como
modelo por defecto, A está pagando reparaciones que B no necesita.

---

## 4. Selección de modelo: dos filosofías opuestas

| | A (intel-v2) | B (make-montages) |
|---|---|---|
| Mecanismo | **push** — watchdog cada 30 min prueba modelos y escribe a Redis (`proxima:model:chat`) | **pull** — `GET /v1/models` en vivo por request |
| Granularidad | 3 llaves fijas (chat, websearch, image-disabled) | por **capability**: `fast_text`, `reasoning`, `vision`, `image`, `websearch` |
| Config | env + Redis | preferencias ordenadas inyectadas desde bootstrap |
| Si no hay nada | cae al default de settings, silencioso | `ModelSelectionError` ruidoso |

El diseño correcto es el híbrido, y es exactamente el `routing.yaml` del plan:
**preferencias ordenadas por capability (B)**, resueltas contra la disponibilidad viva
del gateway, con el resultado cacheado en Redis por el watchdog (A) para no pagar
`GET /v1/models` en cada request. Fallo ruidoso por defecto (B), con opción de degradar.

---

## 5. Consecuencias para el plan

1. **F4 (guard unificado) sube de prioridad.** No es dedupe cosmético: A gana calidad de
   output el día que adopta el prompting de B, y B gana cache/log/reintentos el día que
   adopta A. Considerar adelantarlo antes que F3.
2. **El SDK necesita superficie sync y async.** B es sync (`httpx.Client`) en todo su
   stack; forzarlo a async es reescribir make-montages, fuera de alcance.
3. **`pricing.yaml` es obligatorio, no opcional.** Ningún upstream devuelve costo.
4. **Añadir a F1 un test de compatibilidad de `base_url`** con y sin `/v1`.
5. **Dos bugs se arreglan solos al unificar**: `sources` vacío (A) y 429 fatal (B).

## 6. Inventario para F5 (borrado de copias)

```
A: llm/{cliproxy_client,guard,cache,model_selector,constants,websearch,
        image_models,image_generation,embeddings_client,extract}.py   ~1.100 LOC
   + tests/unit/test_{cliproxy_client,llm_cache,llm_constants}.py
   + scripts/smoke_cliproxy.py
   + servicio model-watchdog en docker-compose.yml

B: infrastructure/cliproxyapi_{common,adapter,image_adapter,
        search_adapter,vision_adapter,model_selector}.py                  980 LOC
   + tests/unit/test_cliproxyapi_*.py (6 archivos)
   + tests/smoke/test_cliproxy_smoke.py
   + poc/cliproxy_explore.py
   + config/cliproxy/config.example.yaml
```

Nota: `make-montages` canónico es `~/projects/make-montages`
(`git@github.com:luisfarfan/make-montages.git`). `rein-wt-configure-runs-and-finish-workbench`
es un worktree del mismo repo — no migrar ahí.
