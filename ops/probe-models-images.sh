#!/usr/bin/env bash
# Barrido CARO: el único que sondea la capacidad de imagen de verdad.
#
# Por qué va aparte del barrido de cada 6 h: `--images` genera una imagen real
# por modelo, sobre los ~39 vivos. Eso es lento y consume cuota de las mismas
# suscripciones que sirven al tráfico, así que no puede correr seguido. El
# barrido barato conserva las etiquetas de imagen que deja éste
# (`prober.py::carry_forward`), así que el mapa no las pierde entre corridas.
#
# Y por qué justo después de medianoche UTC: la cuota diaria de imagen de la app
# web de Gemini se repone a esa hora. Sondear antes mide un modelo agotado y lo
# graba como `image: false` — que es exactamente la etiqueta equivocada que esto
# viene a corregir. Media hora de margen para no competir con el reset.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
exec .venv/bin/python scripts/probe_models.py --websearch --images \
    --out config/capabilities.generated.yaml
