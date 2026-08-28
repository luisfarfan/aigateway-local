# F2 — Por qué el guard gana su sitio, medido

> Contra `agl_cliproxy` (vanilla v7.2.143), modelo `gemini-3-flash` vía antigravity.
> Fecha: 2026-08-27

## Schema fácil: `response_format` alcanza

Objeto plano, un enum de tres valores legibles (`science_tech`, `deportes`, `finanzas`).

| | Resultado |
|---|---|
| Sólo `response_format` | ✅ JSON válido, enum correcto |
| Con guard | ✅ 0 reparaciones |

**Conclusión honesta: acá el guard no aporta nada al parseo.** La suposición de que
Gemini descarta `response_format` siempre no se sostiene en este modelo y esta ruta.

## Schema duro: `response_format` se rompe

Unión discriminada (`oneOf` + `discriminator`) con `const` como discriminante, y un
enum de valores poco naturales (`lvl_a1`, `lvl_b2`, `lvl_c3`).

**Sin guard**, `gemini-3-flash`:

```json
{"secciones": [{"tipo": "Introducción y Propósito", "cuerpo": "Las GPUs …"}]}
```

`tipo` tiene que ser el `const` `"txt_blk"`. El modelo escribió una descripción en
prosa. Es exactamente el modo de fallo que documentaba la auditoría —
`'Technology'` donde el contrato decía `science_tech`— reproducido en esta instancia.

**Con guard**, mismo modelo, mismo prompt:

```
reparaciones: 0 | intentos: ['ok']
nivel: lvl_b2
tipos: ['txt_blk', 'txt_blk']
```

Salió bien **al primer intento**, sin gastar reparaciones. Lo que cambió no fue
reintentar: fue que el schema viajó dentro de la conversación con los valores de los
`const` y del enum escritos. El modelo no los adivinó porque no tuvo que adivinarlos.

`claude-sonnet-4-6` (vía antigravity) acertó el `const` sin guard — la fragilidad no es
uniforme entre modelos, que es justo por qué conviene que el guard sea el camino único
en lugar de una decisión por proveedor.

## Qué queda demostrado

1. `response_format` es **suficiente para lo simple e insuficiente para lo compuesto**.
   Uniones discriminadas y enums de valores arbitrarios son donde se cae.
2. El aporte del guard no es el bucle de reintentos: es **poner el contrato donde el
   modelo sí lo lee**. Los reintentos son la red debajo.
3. Se **manda** el schema estricto y se **valida** contra el original. Lo estricto es
   una exigencia del transporte de OpenAI (`additionalProperties: false`, todo en
   `required`); validar contra él rechazaría respuestas correctas que omiten un campo
   opcional.

## El cache, degradando en vivo

Durante estas pruebas Redis no estaba levantado. El resultado:

```
[warning] llm_cache.error op=get error="Error 111 connecting to localhost:6379"
[warning] llm_cache.error op=set error="Error 111 connecting to localhost:6379"
B) CON guard: 200 — parsed OK
```

La petición se sirvió igual. El cache es best-effort de verdad, no de palabra: sin
Redis se paga la llamada y nada más.
