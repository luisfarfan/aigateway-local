# Autenticar CLIProxyAPI (proveedores cloud)

> Para F0: grabar fixtures de respuestas reales por proveedor.
> El servicio vive en `docker-compose.yml` como `cliproxy`.

## Por qué hace falta un túnel SSH

El login OAuth **no se puede completar apuntando el navegador a la IP de la LAN**.
Verificado en el código del binario (v7.2.143):

1. El `redirect_uri` que se le manda al proveedor es `http://localhost:<puerto>/callback`,
   con el puerto fijo por proveedor. Es el único valor que Google/Anthropic/OpenAI
   tienen en su whitelist — no se puede cambiar por una IP.
2. El *callback forwarder* que atiende ese puerto responde un 302 hacia
   `http://127.0.0.1:8317/<proveedor>/callback`
   (`internal/api/handlers/management/auth_files.go:225`, hardcodeado).

Los dos saltos los hace **el navegador**. Si el navegador está en la MacBook, ese
`localhost` es la MacBook, no el servidor, y el login muere en el segundo salto.

El túnel resuelve las dos cosas a la vez: hace que los puertos del servidor aparezcan
como `localhost` en la MacBook. Efecto secundario bueno — nada queda expuesto a la LAN,
todos los puertos están bind a `127.0.0.1` en el servidor.

## Atajo: importar una sesión ya logueada (sin túnel)

El túnel de arriba sólo hace falta para completar un OAuth **nuevo**. Si la otra
PC ya tiene el Codex CLI logueado, ese OAuth ya ocurrió y los tokens están en
`~/.codex/auth.json`. Se pueden subir directamente:

```bash
# EN LA OTRA PC (no necesita el repo: sólo stdlib)
python3 import_codex_auth.py --host 192.168.1.12:8417 --key <secret-key>
```

`scripts/import_codex_auth.py` traduce la forma del Codex CLI (tokens anidados
bajo `tokens`) a la que CLIProxyAPI guarda en disco (planos, con `type: codex`)
y la sube por `POST /v0/management/auth-files?name=<archivo>.json`. Con
`--dry-run` muestra qué subiría sin subirlo.

Para esto el panel tiene que ser alcanzable desde la LAN — por eso el puerto
8417 está bind a `0.0.0.0` en `docker-compose.yml`, y sólo ése. Los puertos de
callback siguen en `127.0.0.1`: no sirven de nada desde afuera, porque el
`redirect_uri` apunta al `localhost` del navegador.

Esta vía no existe para un proveedor cuya sesión no esté ya en algún lado. Para
eso, el túnel.

## Mapa de puertos

Los puertos **dentro** del contenedor son fijos (hardcodeados en el binario). Los del
host están desplazados para no chocar con otras instancias de CLIProxyAPI en la máquina.

| Proveedor | En el contenedor | En el host | Constante |
|---|---|---|---|
| API + panel | 8317 | 8417 | — |
| codex | 1455 | 11455 | `codexCallbackPort` |
| gemini | 8085 | 18085 | `geminiCallbackPort` |
| antigravity | 51121 | 51221 | `antigravity.CallbackPort` |
| anthropic | 54545 | 54645 | `anthropicCallbackPort` |
| xai | 56121 | 56221 | `xai.CallbackPort` |

En la MacBook el túnel tiene que publicar los puertos **del contenedor**, porque son los
que el binario espera ver en `localhost`.

## Pasos

### 1. En el servidor

```bash
docker compose up -d cliproxy
```

### 2. En la MacBook — abrir el túnel

```bash
ssh -N \
  -L 8317:127.0.0.1:8417 \
  -L 1455:127.0.0.1:11455 \
  -L 8085:127.0.0.1:18085 \
  -L 51121:127.0.0.1:51221 \
  -L 54545:127.0.0.1:54645 \
  -L 56121:127.0.0.1:56221 \
  lucho@192.168.1.12
```

Dejarlo corriendo durante todo el login.

### 3. Abrir el panel

`http://localhost:8317/management.html`

Pide el `secret-key`, que está en `docker/cliproxy/config.yaml` bajo
`remote-management.secret-key`. Ese archivo **no se versiona** (lleva los secretos en
claro; el binario no interpola variables de entorno). La plantilla versionada es
`config.example.yaml`.

### 4. Autenticar cada proveedor

En el panel, sección de cuentas → añadir. Prioridad para los fixtures:

- **Gemini** (Google) — modelo por defecto de casi todo
- **Anthropic / Claude** — su bloque de websearch es distinto
- **Codex / OpenAI** — usa `/v1/responses`, otro endpoint entero
- Antigravity y xAI, opcionales

Ya hay una sesión de *antigravity* en otra instancia
(`~/.make-montages/cliproxy-auth`), pero expone los 13 modelos con
`owned_by: antigravity` — no ejercita las diferencias de traducción entre proveedores,
que es justo lo que los fixtures tienen que capturar.

### 5. Verificar

```bash
curl -s -H "Authorization: Bearer <api-key de config.yaml>" http://127.0.0.1:8417/v1/models
```

Debe listar modelos con `owned_by` de cada proveedor autenticado.

### 6. Cerrar el túnel

Ctrl-C. Los tokens quedan en `~/.proxima-gateway/cliproxy-auth/` (bind-mount del host,
sobrevive a `docker compose down -v`) y se refrescan solos.

## Notas

- `allow-remote: true` es obligatorio: la petición entra por el bridge de Docker
  (172.x.x.x) y el modo estricto la rechazaría como remota.
- El directorio de auth es propio de este repo. No se comparte con
  `~/.make-montages/cliproxy-auth` ni con `~/.proxima-intel/cliproxy-auth` a propósito —
  no tocamos las sesiones de otros proyectos.
- `usage-statistics-enabled: true` queda activo, pero la retención en memoria es de 60s.
  Por eso F3 persiste el uso a Postgres.
