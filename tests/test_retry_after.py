"""
Respetar el tiempo de espera que el upstream declara.

Por qué importa, medido: el 429 de Gemini dice "quota will reset after 116h" y
el breaker abría 120 s fijos. Con eso, un modelo muerto por días se reintenta en
cada petición para siempre.

Y peor: el contador de fallos —5 en una ventana de 60 s— está pensado para chat.
Una ruta de imagen tarda 20-90 s por llamada y se usa de a una, así que NUNCA
junta cinco dentro de la ventana. Sin leer el `retry_after`, el circuito de un
modelo de imagen no se abre nunca.
"""

from __future__ import annotations

import pytest

from src.modules.providers.cliproxy.errors import (
    MAX_RETRY_AFTER_S,
    classify,
    parse_retry_after,
)

# El cuerpo tal cual lo devolvió Google, recortado.
CUOTA_AGOTADA = {
    "error": {
        "code": 429,
        "message": (
            "You have exhausted your capacity on this model. "
            "Your quota will reset after 116h29m24s."
        ),
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "metadata": {"quotaResetDelay": "116h29m24.770887609s"},
            },
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "419364.770887609s",
            },
        ],
    }
}


def test_se_lee_el_retry_del_429_real():
    """Contra el cuerpo real, no uno inventado."""
    assert parse_retry_after(CUOTA_AGOTADA) == MAX_RETRY_AFTER_S


def test_se_topa_para_que_un_valor_absurdo_no_saque_un_modelo_para_siempre():
    """116 h son 4 días y medio. Sin tope, un parseo malo o un upstream que
    miente dejarían el modelo fuera de la cadena, y eso se nota tarde."""
    enorme = {"error": {"details": [{"retryDelay": "999999h"}]}}
    assert parse_retry_after(enorme) == MAX_RETRY_AFTER_S


@pytest.mark.parametrize(
    ("payload", "esperado"),
    [
        ({"error": {"details": [{"retryDelay": "90s"}]}}, 90),
        ({"error": {"details": [{"metadata": {"quotaResetDelay": "2h30m"}}]}}, 9000),
        ({"error": {"message": "quota will reset after 50h58m28s"}}, MAX_RETRY_AFTER_S),
        ({"error": {"message": "quota will reset in 45s"}}, 45),
    ],
)
def test_los_tres_sitios_donde_google_lo_esconde(payload: dict, esperado: int):
    """Google repite el dato en tres lugares del mismo 429 y ninguno está
    garantizado. Se prueban los tres."""
    assert parse_retry_after(payload) == esperado


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"message": "boom"}},
        {"error": {"details": [{"retryDelay": "no es una duración"}]}},
        {"error": "no es un dict"},
        {},
        None,
    ],
)
def test_sin_dato_legible_devuelve_none_en_vez_de_inventar(payload):
    """`None` hace que el breaker use su default. Inventar un número sería peor
    que no tener el dato."""
    assert parse_retry_after(payload) is None


def test_el_error_clasificado_lleva_la_espera():
    """El routing lee `retry_after_s` de la excepción, así que tiene que
    sobrevivir a la clasificación."""
    err = classify(429, CUOTA_AGOTADA, path="/v1/chat/completions")
    assert err.retry_after_s == MAX_RETRY_AFTER_S
