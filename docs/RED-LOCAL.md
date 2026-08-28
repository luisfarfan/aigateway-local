# Usar el gateway desde cualquier dispositivo del WiFi

> Verificado el 2026-08-28 desde `192.168.1.12:8000`.

## Antes que nada: la clave

El plano `/v1/*` **exige autenticación**. Sin ella, exponerlo a la red sería dejar
que cualquiera en el WiFi gaste tus cuentas de Gemini, Codex y Claude, y las suyas
tienen cuota mensual.

La clave está en `.env`, en `API_KEYS`, y ese archivo **no se versiona**. Para rotarla:

```bash
sed -i "s/^API_KEYS=.*/API_KEYS=pxg-$(openssl rand -hex 24)/" .env
```

`API_KEYS` acepta varias separadas por coma, así que se puede dar una distinta a cada
consumidor y revocar sólo esa.

**Con `API_KEYS` vacío el gateway queda abierto**: es el modo cómodo para desarrollo
en local, y es exactamente lo que no debe usarse escuchando en la red.

## Levantarlo

```bash
make serve      # uvicorn en 0.0.0.0:8000, sin --reload
```

Para que quede corriendo siempre, incluso tras reiniciar, hay dos servicios de
usuario de systemd (API y worker). Instalación y trampas del `.env` en el README,
sección **Servicio nativo**:

```bash
mkdir -p ~/.config/systemd/user
cp ops/aigateway.service ops/aigateway-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now aigateway aigateway-worker
sudo loginctl enable-linger $USER   # sin esto, systemd los mata al cerrar sesión
```

Los contenedores (cliproxy, postgres, redis, minio, langfuse) los levanta
`docker compose up -d` aparte.

## Qué queda expuesto y qué no

| Servicio | Dirección | Alcance |
|---|---|---|
| **Gateway `/v1/*`** | `0.0.0.0:8000` | **toda la red**, con API key |
| CLIProxyAPI + su panel | `127.0.0.1:8417` | sólo esta máquina |
| Langfuse | `127.0.0.1:3778` | sólo esta máquina |
| Postgres · Redis · MinIO · ClickHouse | `127.0.0.1` | sólo esta máquina |

Sólo se abre el gateway. El panel de gestión de CLIProxyAPI **no** se expone: desde
ahí se pueden descargar los tokens OAuth de las cuentas, así que darle acceso de red
sería peor que exponer el gateway sin clave.

## Desde otro dispositivo

```bash
curl -X POST http://192.168.1.12:8000/v1/chat/completions \
  -H "Authorization: Bearer <tu-API_KEYS>" \
  -H "Content-Type: application/json" \
  -H "X-Proxima-Project: mi-app" \
  -d '{"messages":[{"role":"user","content":"Di PONG"}],"max_tokens":20}'
```

Con el SDK:

```python
from proxima_llm import SyncGateway

gw = SyncGateway("http://192.168.1.12:8000", api_key="...", project="mi-app")
gw.chat("¿Capital de Perú?").text
```

`X-Proxima-Project` no es decorativo: es lo que separa el costo y las trazas por
consumidor. Sin él todo el gasto cae en un mismo balde y los reportes no sirven.

## Si la IP cambia

Es DHCP: el router puede darle otra IP tras un reinicio. Para que no se rompa,
reservarle la IP en el router por MAC, o llamarlo por nombre — muchos routers
resuelven `<hostname>.local`.

## Verificado

```
escuchando en:   0.0.0.0:8000
sin clave:       401
con clave:       200, gemini-3-flash, 'PONG'
```

Y la suite completa del SDK contra `192.168.1.12:8000`:

```
ok  models: cloud + local     36 modelos, 16 locales
ok  chat                      gemini-3-flash · 9 tok
ok  search con fuentes        3 fuentes: binance.com, coinmarketcap.com, coingecko.com
ok  structured validado       {"nombre": "iPhone 15 Pro Max", "categoria": "telefono"}
ok  modelo local              qwen2.5:7b
ok  imagen                    803 KB
ok  clave mala rechazada      HTTP 401
ok  sync chat
```
