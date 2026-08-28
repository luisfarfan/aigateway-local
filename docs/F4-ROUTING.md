# F4 — Routing, fallback y watchdog

> Fecha: 2026-08-27

```
config/routing.yaml
src/modules/routing/
├── config.py     cadenas por capacidad, política del breaker
├── errors.py     un fallo, un nombre
├── breaker.py    circuit breaker por modelo, estado en Redis
├── executor.py   ejecuta sobre la cadena
└── watchdog.py   prueba modelos, no los lista
```

## Las reglas que definen el comportamiento

**1. No todo fallo justifica un fallback.** Un 429 sí: otro modelo tiene otra cuota.
Un 400 no: la petición está mal formada y el siguiente la rechaza igual, así que
reintentar sólo gasta cuota y latencia para llegar al mismo error. Qué sí justifica
saltar está en `routing.yaml`, no repartido por el código.

**2. Un fallo tiene un solo nombre.** `routing/errors.py` es el único sitio que
clasifica. El código HTTP que ve el cliente, la decisión de fallback y la etiqueta
de la métrica salen de ahí. Cuando cada capa clasificaba por su cuenta, podían
divergir: el cliente veía un 503 que invita a reintentar mientras el routing ya
había decidido que ese modelo no sirve.

**3. El fallback nunca es silencioso.** El modelo que respondió va en el cuerpo, y
si hubo salto se dice desde dónde:

```json
"proxima": {"fell_back_from": "gpt-image-2", "served_by": "gemini-3-flash"}
```

**4. Si toda la cadena falla, se levanta el último error**, no el primero: describe
el estado actual del sistema. Devolver el primero escondería que se intentaron los
demás.

**5. "Toda la cadena abierta" es un error distinto de "todos fallaron".** El primero
significa que ni se pudo intentar — casi siempre un breaker demasiado agresivo, no
un upstream caído. Confundirlos manda a diagnosticar el lado equivocado.

## El breaker vive en Redis

La API y los workers son procesos separados. Un breaker por proceso obliga a cada
uno a quemarse por su cuenta y multiplica los fallos por la cantidad de procesos.
Mismo patrón de contador con TTL que ya usa `ModalityScheduler`.

Degrada hacia **dejar pasar**: si Redis no está, `is_open` devuelve False. Bloquear
por no poder consultar el estado sería inventarse una caída.

## El watchdog prueba, no lista

Hallazgo de F0: `GET /v1/models` **no es prueba de vida**. Una instancia anunció 62
modelos de los cuales 32 devolvían `token revoked` o `403`. Un selector que filtre
por esa lista elige con confianza un modelo muerto.

El watchdog manda una llamada de pocos tokens a cada modelo y escribe el resultado
**en el mismo circuit breaker** que consulta el routing. Dos tablas de salud
distintas acabarían contradiciéndose.

### Un bug que sólo apareció corriéndolo

El primer barrido real marcó `gpt-image-2` como muerto:

```
breaker.opened model=gpt-image-2 reason='watchdog: 503: model gpt-image-2 is only
supported on /v1/images/generations and /v1/images/edits'
```

El modelo está perfectamente vivo. La sonda es una llamada de **chat**, y un modelo
de sólo imagen la rechaza. El watchdog habría abierto su circuito cada 15 minutos y
roto la ruta de imagen entera, sin que ninguna prueba unitaria lo notara.

Arreglo: `routing.yaml` declara qué rutas no se sondean.

```yaml
watchdog:
  skip_routes:
    - image
```

Sondear imagen de verdad costaría generar una imagen por barrido — caro y lento. Los
modelos de imagen se descartan por fallos reales a través del breaker, no por sondeo.
Un modelo que aparece **además** en una ruta sondeable sí se sondea: si responde a
chat, la credencial sirve, que es lo que la sonda comprueba.

Hay dos tests de regresión para esto.

## Verificado en vivo

Fallback, pidiendo un modelo que no sirve para chat:

```
[warning] routing.fallback from_model=gpt-image-2 kind=upstream_unavailable route=chat
status: 200
  modelo que respondió: gemini-3-flash
  proxima: {"fell_back_from": "gpt-image-2", "served_by": "gemini-3-flash"}
```

Barrido del watchdog, después del arreglo:

```
watchdog.sweep alive=3 total=3 models=['claude-sonnet-4-6','gemini-3-flash','gpt-5.4-mini']
gpt-image-2            open=False   ← ya no se sondea, ya no se marca muerto
gemini-3.1-flash-image open=False
```

## Costo del watchdog

Una llamada de pocos tokens por modelo sondeable y por barrido. Con el intervalo por
defecto (15 min) y tres modelos, 12 llamadas mínimas por hora. Menos que descubrir un
modelo muerto en medio del tráfico real, una petición a la vez.
