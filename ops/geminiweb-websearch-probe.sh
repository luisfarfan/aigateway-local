#!/usr/bin/env bash
# Responde UNA pregunta abierta: la búsqueda de imágenes web de `geminiweb/`,
# ¿está disponible siempre, o sólo cuando la cuota de generación se agotó?
#
# Importa porque decide si sirve como fuente estable para un pipeline o sólo de
# rebote. Todas las capturas que motivaron el campo `proxima_origin` se hicieron
# con la cuota ya agotada, así que el caso "con cuota disponible" nunca se midió.
#
# La librería no expone ninguna perilla —`generate_content()` no tiene un
# `search_images=`— así que lo decide Gemini leyendo el prompt. Lo que se mide
# acá es si esa decisión cambia según haya cuota o no.
#
# Corre de madrugada UTC, después del barrido de imagen, con la cuota diaria
# recién repuesta. Gasta 1-2 imágenes de esa cuota: ese es el costo del dato.
#
# SE DESACTIVA SOLO al llegar a un veredicto. Es una pregunta de una vez, no una
# vigilancia: dejar el timer corriendo gastaría cuota cada semana para volver a
# aprender lo mismo.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

CLAVE="$(printf '%s' "${API_KEYS:-}" | cut -d, -f1)"
BASE="http://localhost:8000"
MODELO="geminiweb/nano-banana-web"
comunes=(-H "Authorization: Bearer $CLAVE" -H "X-Proxima-Project: geminiweb-probe"
         -H "X-Proxima-No-Fallback: 1" -H "Content-Type: application/json")

pedir() {  # $1 = prompt, $2... = cabeceras extra
    local prompt="$1"; shift
    curl -s -m 320 "$BASE/v1/images/generations" "${comunes[@]}" "$@" \
        -d "$(printf '{"model":"%s","prompt":%s}' "$MODELO" "$(printf '%s' "$prompt" | .venv/bin/python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")"
}

veredicto() {  # $1 = texto
    echo "VEREDICTO: $1"
    # La pregunta ya está respondida; el timer no tiene nada más que medir.
    systemctl --user disable --now geminiweb-websearch-probe.timer 2>&1 || true
}

echo "== paso 1: ¿hay cuota de generación?"
gen="$(pedir 'Genera una imagen de un cuadrado rojo sobre fondo blanco')"
if ! printf '%s' "$gen" | grep -q '"proxima_origin": *"generated"'; then
    echo "$gen" | head -c 300
    # Sin generación disponible el experimento no distingue nada: es el mismo
    # escenario que ya se midió. Se reintenta en la próxima ventana.
    echo "INCONCLUSO: no hay cuota de generación en esta ventana; no se desactiva"
    exit 0
fi
echo "   ok: generó. La cuota de generación está disponible."

echo "== paso 2: con cuota disponible, ¿busca igual en la web?"
web="$(pedir 'Muestrame fotos reales de la torre Eiffel' -H 'X-Proxima-Allow-Web-Images: 1')"

if printf '%s' "$web" | grep -q '"proxima_origin": *"web"'; then
    veredicto "la búsqueda web NO depende de la cuota: la elige el prompt. \
Sirve como fuente estable — ver README, 'Procedencia'."
elif printf '%s' "$web" | grep -q '"proxima_origin": *"generated"'; then
    veredicto "con cuota disponible GENERA en vez de buscar: la búsqueda web es \
sólo un rebote de la cuota agotada. No sirve como fuente estable."
else
    echo "$web" | head -c 400
    echo "INCONCLUSO: ni web ni generated; no se desactiva"
fi
