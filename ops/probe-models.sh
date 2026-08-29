#!/usr/bin/env bash
# Regenera el mapa de capacidades. Lo corre el timer systemd; el gateway lo
# recoge solo (load_tiers relee cuando cambia el mtime del archivo).
#
# Sin --images a propósito: la imagen tarda 30-150 s por modelo y gasta cuota;
# un barrido programado no debe pagar eso cada vez. La capacidad de imagen se
# fija en el barrido manual del inicio y cambia poco. Websearch sí (es barato).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
exec .venv/bin/python scripts/probe_models.py --websearch --out config/capabilities.generated.yaml
