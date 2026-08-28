# F6 — SDK y evals

> Fecha: 2026-08-27

```
sdk/python/proxima_llm/
├── client.py      Gateway (async) y SyncGateway
├── _protocol.py   construcción y lectura, compartido
├── types.py       Completion, Source, Image
└── errors.py      ProximaError
evals/
├── datasets/structured.yaml
└── run.py
```

## El SDK

Habla HTTP con el gateway. No reimplementa traducciones, fallback, cache ni costos:
si algo se puede arreglar del lado del gateway, no debería requerir publicar una
versión nueva del SDK.

```python
from proxima_llm import SyncGateway

gw = SyncGateway("http://gateway:8000", project="tienda")
gw.chat("¿Capital de Perú?").text
gw.search("Precio del Bitcoin hoy").sources      # [Source(uri=..., title=...)]
gw.structured("clasifica esto", schema=SCHEMA).parsed   # dict ya validado
gw.image("un cubo rojo").images[0].url           # data URI
```

### Dos superficies, un solo comportamiento

`SyncGateway` es síncrono **de verdad** —`httpx.Client`—, no un `asyncio.run`
envolviendo al async: envolver rompe dentro de un bucle ya corriendo, que es
exactamente donde más molesta.

Las dos comparten `_protocol.py`, así que la única diferencia posible entre ellas es
el transporte. Hay un test que compara sus APIs públicas y otro que ejercita la
síncrona contra un servidor uvicorn real — sin eso, `SyncGateway` sería una
superficie que nadie probó nunca.

### El error trae la decisión, no sólo el código

`ProximaError.retryable` lo dice el gateway, no se deduce del status. Sólo él sabe si
ya agotó la cadena de modelos o si el problema es transitorio. Un consumidor que
dedujera "503 → reintento" quedaría reintentando para siempre un modelo que ninguna
credencial cubre.

Cuando falla la salida estructurada, `ProximaError.attempts` trae qué pasó en cada
intento.

## Los evals

Convierten "¿este modelo sirve?" en un número que se puede mirar **antes** de cambiar
`routing.yaml`, y volver a correr después para ver si algo empeoró.

```bash
python evals/run.py --models gemini-3-flash gpt-5.4-mini ollama/qwen2.5:7b
```

Los casos no son ejemplos bonitos: cada uno reproduce un modo de fallo observado
—enum de valores no naturales, unión discriminada con `const`, opcional que debe
quedar nulo, enteros que llegan como string—. Y las afirmaciones van sobre el
contenido, no sólo sobre la validación: un JSON que cumple el schema puede tener el
dato equivocado, que es justo lo que interesa medir.

Se corre con `no_cache`: un acierto de cache mediría la memoria del gateway, no el
modelo.

### Encontraron un bug de verdad, en la primera corrida

```
gpt-5.4-mini   FALLA union_discriminada
  400: Invalid schema for response_format: In context=(... 'tipo'),
       schema must have a 'type' key.
```

El modo estricto de OpenAI exige que **todo** nodo lleve `type`, incluidos los que se
describen sólo por su valor. Pydantic emite el discriminante de una unión como
`{"const": "txt_blk"}` pelado, sin `type`.

Lo grave es dónde falla: es un **HTTP 400**, así que la petición nunca llega al
modelo y el bucle de reparación del guard no puede salvarla. Con Gemini el caso
pasaba, así que sin evals esto se habría descubierto el día que alguien cambiara la
cadena a un modelo de OpenAI.

Arreglo: `to_strict_json_schema` deduce el `type` de un `const`, y de un `enum` cuyos
valores sean todos del mismo tipo. Un enum de tipos mezclados se deja como está —
inventar un tipo ahí sería adivinar. Cuatro tests de regresión.

### Después del arreglo

```
modelo                         acierto  reparac.   mediana    tokens
gemini-3-flash               4/4         100%        0      1.4s      1116
gpt-5.4-mini                 4/4         100%        0      2.5s      1468
ollama/qwen2.5:7b            4/4         100%        0      3.2s      1191
```

12 de 12, cero reparaciones. Vale notar el tercero: **el modelo local pasa todos los
casos**, más lento (19s en el más duro contra 4s del cloud) pero sin fallar ninguno.
Como último recurso de la cadena no es un consuelo, es una alternativa real.
