#!/usr/bin/env bash
# Renueva la cookie de la app web de Gemini desde el navegador local, antes de
# que se pudra. Lo corre gemini-web-refresh.timer una vez al día.
#
# Por qué hace falta: `gemini_webapi` rota `__Secure-1PSIDTS` sola, pero sólo
# **mientras hay tráfico**. Este backend es el último recurso de la cadena de
# imagen y casi nunca recibe peticiones, así que su copia se queda quieta y
# Google la caduca a los pocos días. En Chrome, en cambio, la sesión sigue
# sana porque la persona usa gemini.google.com. Esto copia la sana sobre la
# podrida.
#
# UNA VEZ AL DÍA, NO MÁS. Cada corrida abre una sesión autenticada nueva contra
# Google al validar la cookie. Eso no es gratis: medido, 11 sesiones en hora y
# media desde un servidor terminaron con la cuenta invalidada — ver
# `gemini_web.py::check_session`. La sonda que vigila la sesión no debe ser la
# que la mate. Si algún día esto se vuelve más frecuente, hay que revisar ese
# comentario primero.
#
# Requiere el LLAVERO DESBLOQUEADO para descifrar el store de Chrome, o sea una
# sesión gráfica iniciada. Por eso el timer cuelga de graphical-session.target.
set -euo pipefail
cd "$(dirname "$0")/.."

env_psid() { grep -m1 '^GEMINI_WEB_SECURE_1PSID=' .env | cut -d= -f2- | tr -d '"'; }

antes="$(env_psid || true)"

# El script valida la cookie nueva ANTES de escribir: si no autentica, no toca
# el .env y sale distinto de 0. Con `set -e` eso aborta acá, que es lo correcto
# — no hay nada que reiniciar y el monitor ya avisará si la vieja también murió.
.venv/bin/python scripts/gemini_web_login.py

despues="$(env_psid)"

# Reiniciar sólo si la credencial cambió de verdad. El gateway lee el .env al
# arrancar y mantiene un cliente vivo; sin cookie nueva, el reinicio sería
# cortar peticiones en vuelo para no cambiar nada.
if [[ "$antes" == "$despues" ]]; then
    echo "cookie sin cambios — no se reinicia el gateway"
    exit 0
fi

echo "cookie renovada — reiniciando el gateway"
systemctl --user restart aigateway
