# F3 — Trazas, métricas y costo

> Estado: **hecho y verificado en vivo**, Langfuse propio incluido.
> Fecha: 2026-08-27

## Qué se agregó

```
config/pricing.yaml                      tarifas versionadas
src/modules/observability/
├── pricing.py     costo por llamada, con la distinción cero ≠ desconocido
├── tracing.py     OTel GenAI semconv → OTLP → Langfuse
├── models.py      llm_requests + llm_attempts
└── recorder.py    un solo sitio donde todo se publica
src/core/metrics.py                      8 métricas nuevas `gateway_llm_*`
```

## La decisión que ordena todo lo demás

**Un costo desconocido no es un costo cero.** Tres estados, no dos:

| | `priced` | `cost_usd` | `cost_equivalent_usd` |
|---|---|---|---|
| Modelo con tarifa, pagado por API | `true` | el costo real | igual |
| Modelo con tarifa, vía suscripción OAuth | `true` | `0.0` | precio de lista |
| Modelo sin tarifa en `pricing.yaml` | `false` | `null` | `null` |

El tercer caso además incrementa `gateway_llm_unpriced_total`. Sin esa señal, un
hueco en la tabla de precios no se nota por ningún lado hasta que alguien pide el
reporte de costos y las cifras están mal desde hace semanas.

El `cost_equivalent_usd` se guarda incluso cuando no se cobra: es lo que costaría la
misma carga por API. Sirve para dimensionar cuánto ahorran las suscripciones y para
comparar modelos entre sí, que es la decisión que uno quiere tomar con estos datos.

## Por qué OTel y no el SDK de Langfuse

Los atributos quedan en vocabulario estándar (`gen_ai.system`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens`), así que el mismo span puede ir
a Langfuse, a Phoenix o a Tempo cambiando un endpoint. Langfuse acepta OTLP sobre
HTTP con auth Basic, así que no hace falta ningún exportador propietario.

Sin claves configuradas el proveedor de trazas queda activo pero sin exportar: el
código que abre spans no tiene que saber si alguien escucha.

## Por qué dos tablas

- `llm_requests` — una fila por petición **lógica**. Unidad de facturación.
- `llm_attempts` — una fila por llamada **física**. Unidad de diagnóstico.

Con una sola habría que elegir entre contar el costo dos veces o perder el detalle
de por qué una petición necesitó tres llamadas.

Se guarda además de exportar a Langfuse porque CLIProxyAPI retiene sus estadísticas
**60 segundos en memoria**, y Langfuse es un sistema aparte que puede caerse,
cambiarse o purgarse.

## Verificado en vivo

Petición real por el plano síncrono, con Postgres levantado:

```
filas en llm_requests: 2
  demo-costos  chat  gpt-5.4-mini  out=ok              tok=9/213  priced=True cost=0.0 equiv=0.00214125 trace=5e2baa6b4432
  demo-costos  chat  gpt-image-2   out=upstream_error  tok=0/0    priced=True cost=0.0 equiv=0.0        trace=f0ded990a255
```

El fallo queda registrado igual, con su `trace_id`. Métricas:

```
gateway_llm_requests_total{model="gemini-3-flash",outcome="ok",project="tienda",route="chat"} 1.0
gateway_llm_tokens_total{direction="input",model="gemini-3-flash",project="tienda"} 4.0
gateway_llm_tokens_total{direction="output",model="gemini-3-flash",project="tienda"} 123.0
gateway_llm_cost_equivalent_usd_total{model="gemini-3-flash",project="tienda"} 0.0003087
gateway_llm_cache_total{project="tienda",result="disabled"} 1.0
```

`gateway_llm_cost_usd_total` no aparece porque vale cero: con suscripción no se cobra,
y un contador no se incrementa con cero. Es el comportamiento correcto — el gasto real
es cero y el panel debe mostrar cero, no el precio de lista.

## Nada de esto puede tumbar una petición

Persistir el histórico es best-effort, igual que el cache: si Postgres no está, se
registra un warning y la respuesta sale. La observabilidad no puede ser el motivo de
que algo falle.

Una `Observation` nace marcada como **fallo** y sólo pasa a `ok` si alguien lo
declara. Si el endpoint revienta por una excepción inesperada, el registro dice que
falló en vez de desaparecer de las cifras.

## Langfuse, propio de este repo

Instancia propia en el `docker-compose.yml` de este repo, no la del PoC que vive en
otro proyecto. La razón es el objetivo del gateway: si otro proyecto tiene que poder
borrarse sin afectar a este, el gateway no puede depender de un contenedor que vive
dentro de él.

Reutiliza el Postgres, el Redis y el MinIO que ya levanta este compose — son cuatro
contenedores menos que un stack de Langfuse aparte. Sólo se agregan tres:

| Servicio | Puerto host | Para qué |
|---|---|---|
| `agl_langfuse_web` | `127.0.0.1:3778` | UI + endpoint OTLP |
| `agl_langfuse_worker` | — | procesa la cola de ingesta |
| `agl_clickhouse` | — | las trazas, que son series de alto volumen |

Más `agl_postgres_init`, que crea la base `langfuse` en el Postgres existente (un
script en `/docker-entrypoint-initdb.d` no serviría: sólo corre con el volumen vacío).

### Sin paso manual por la UI

Las claves se **siembran** al primer arranque con `LANGFUSE_INIT_*`, y son las mismas
que el gateway ya tiene en su `.env`. No hay que entrar a crear un proyecto ni copiar
nada: el gateway exporta trazas desde la primera petición.

Org `proxima`, proyecto `proxima-gateway`, y el usuario de la UI con las credenciales
del `.env` (`LANGFUSE_INIT_USER_*`).

### Verificado

```
tracing.exporter_ready endpoint=http://localhost:3778/api/public/otel/v1/traces
tienda: 200 modelo=gemini-3-flash
intel:  200 modelo=gpt-5.4-mini-2026-03-17

trazas en Langfuse: 2
  llm.websearch  GENERATION  gpt-5.4-mini-2026-03-17  usage: in=2903 out=104 total=3007
  llm.chat       GENERATION  gemini-3-flash           usage: in=4 out=2064 total=2068
```

Langfuse las clasificó como `GENERATION` con modelo y tokens correctos **sin
configuración específica**: es lo que se gana instrumentando con la convención GenAI
de OpenTelemetry en vez de atributos propios. El mismo span serviría igual para
Phoenix o Tempo.

## Puertos remapeados

Tres puertos del host ya estaban ocupados por otros servicios de la máquina. Dentro
de la red de compose nada cambia; sólo afecta a procesos corriendo en el host:

| Servicio | Antes | Ahora |
|---|---|---|
| Postgres | 5432 | `127.0.0.1:5442` |
| MinIO API | 9000 | `127.0.0.1:9010` |
| MinIO consola | 9001 | `127.0.0.1:9011` |
| Langfuse | (3777 es del PoC ajeno) | `127.0.0.1:3778` |
