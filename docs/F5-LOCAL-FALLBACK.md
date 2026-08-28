# F5 — Fallback cloud → local

> Fecha: 2026-08-27

Lo que sólo se puede hacer teniendo un gateway único: se acaba la cuota de todos los
proveedores cloud y la petición la sirve un modelo en la GPU de esta máquina, en vez
de fallar.

```
src/modules/backends/
├── base.py       el contrato que cumple cualquier cosa que sirva un modelo
├── ollama.py     backend local
└── registry.py   qué backend sirve cada modelo
```

## Cómo se elige el backend

Un prefijo explícito, no una heurística:

```yaml
routes:
  chat:
    - gemini-3-flash
    - gpt-5.4-mini
    - claude-sonnet-4-6
    - ollama/qwen2.5:7b     # último recurso, GPU local
```

Adivinar por el nombre —"si dice `qwen` es local"— se rompe el día que un proveedor
cloud sirva un Qwen. Ya pasa hoy: `gpt-oss-120b` lo sirve antigravity y no es de
OpenAI.

El prefijo se quita antes de llamar, así que el backend recibe el id que su API
entiende.

## Ollama habla OpenAI

Su endpoint `/v1/chat/completions` acepta la misma forma, así que el backend local
reutiliza la traducción y el parseo del camino cloud. Un segundo formato de mensajes
sólo habría sido una segunda cosa que mantener.

## Lo que el backend local no puede

No hace búsqueda web ni genera imágenes. Decirlo con un error propio
(`BackendCapabilityError` → `unsupported_capability`) en vez de con un 500 genérico
permite que el routing lo trate como lo que es: motivo para probar el siguiente
candidato, no para abortar.

Ese kind está en `fallback_on`. Hacia afuera es un **501**: la petición es válida, es
este gateway el que no la puede servir con lo que tiene configurado.

## Verificado en vivo

Modelo local pedido explícitamente:

```
1) local directo: 200 | modelo: qwen2.5:7b | 'PONG'
```

Cloud completamente apagado (`base_url` a un puerto muerto):

```
routing.fallback from_model=gemini-3-flash     kind=upstream_timeout
routing.fallback from_model=gpt-5.4-mini       kind=upstream_timeout
routing.fallback from_model=claude-sonnet-4-6  kind=upstream_timeout
2) cloud apagado: 200 | modelo: qwen2.5:7b
   proxima: {"fell_back_from": "gemini-3-flash", "served_by": "ollama/qwen2.5:7b"}
```

Tres proveedores cloud caídos y la petición se sirvió igual, declarando desde dónde
cayó.

## Dos bugs que sólo aparecieron corriéndolo

### `CliproxyClient` no cumplía su propio protocolo

Le faltaba `name`. El router reventaba con un `AttributeError` que el routing
clasificó como `upstream_error` — que **no** está en `fallback_on`, así que la
petición ni siquiera caía al siguiente modelo: fallaba entera.

No se vio en los tests porque los dobles sí tenían `name`. El arreglo incluye un test
de conformidad que comprueba las implementaciones **reales** contra el protocolo
(`Backend` es `runtime_checkable`), no contra un doble.

### El breaker se abrió por mi propio bug

Mientras el bug anterior estaba vivo, cada petición fallida contaba, y tras cinco el
circuito de varios modelos quedó abierto. El breaker hizo exactamente lo que debía;
sirve como confirmación de que funciona fuera de los tests.

### Un candidato sin backend tapaba el error real

Al agotarse la cadena se levantaba el **último** error. Si el último candidato es un
`ollama/…` y Ollama no está configurado, ese error es `unsupported_capability`, que
describe la configuración y no el problema: tapaba el 429 que había causado todo y
mandaba a diagnosticar el lado equivocado.

Ahora se levanta el último error **sustantivo**, ignorando los candidatos
inservibles. Con test.

## Precio de lo local

Los modelos locales se listan en `pricing.yaml` con tarifa cero explícita:

```yaml
families:
  local: {input: 0, output: 0}
```

No se dejan fuera. Un modelo ausente de la tabla incrementa
`gateway_llm_unpriced_total`, y ese contador tiene que señalar **huecos en la tabla**,
no modelos que legítimamente son gratis. Listándolos, `priced=True` y `cost=0`, que es
la verdad.
