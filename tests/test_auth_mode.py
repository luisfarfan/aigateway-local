"""
Que `cost_usd` sea real cuando la llamada se paga de verdad.

`Observation.auth_mode` estaba fijo en `"oauth"` y nadie lo cambiaba nunca, así
que `is_charged()` devolvía siempre `False` y **`cost_usd` salía 0.00 en todas
las llamadas**. Mientras todo va por suscripción eso es correcto por accidente;
el día que se carga una credencial de pago, el gasto real desaparecería del
reporte sin que nada lo señalara.

El upstream no ayuda: CLIProxyAPI devuelve un `X-Cpa-Trace-Id` y nada sobre qué
credencial sirvió. La única señal es el prefijo del modelo, porque una
credencial de pago se declara allá con `prefix:`.
"""

from __future__ import annotations

import pytest

from src.modules.observability.pricing import load_pricing
from src.modules.observability.recorder import Observation


@pytest.mark.parametrize(
    ("modelo", "esperado"),
    [
        ("paid/gemini-3.1-flash-image", "api_key"),
        ("PAID/GPT-IMAGE-2", "api_key"),  # el id puede venir en cualquier caja
        ("gemini-3-flash", "oauth"),
        ("gpt-image-2", "oauth"),
        ("geminiweb/nano-banana-web", "oauth"),
        ("ollama/qwen2.5:7b", "oauth"),
    ],
)
def test_el_prefijo_decide_si_se_paga(modelo: str, esperado: str):
    assert load_pricing().auth_mode_for(modelo) == esperado


def test_un_modelo_de_pago_produce_un_costo_REAL():
    """Lo que estaba roto: antes esto daba `amount_usd = 0.0`."""
    obs = Observation(project="p", route="image", requested_model="paid/gpt-image-2")
    obs.response_model = "paid/gpt-image-2"
    obs.image_count = 1

    cost = obs.cost()
    assert cost.charged is True
    assert cost.amount_usd == pytest.approx(0.04)
    assert cost.saved_usd == 0.0, "no se ahorró nada: esto se pagó"


def test_una_suscripcion_sigue_costando_cero_pero_deja_el_equivalente():
    """El costo marginal de una llamada por OAuth es cero, pero el precio de
    lista se guarda igual: sirve para dimensionar el ahorro y comparar modelos."""
    obs = Observation(project="p", route="image", requested_model="gpt-image-2")
    obs.response_model = "gpt-image-2"
    obs.image_count = 1

    cost = obs.cost()
    assert cost.charged is False
    assert cost.amount_usd == 0.0
    assert cost.equivalent_usd == pytest.approx(0.04)
    assert cost.saved_usd == pytest.approx(0.04)


def test_manda_el_modelo_que_RESPONDIO_no_el_que_se_pidio():
    """Una cadena puede empezar en una credencial de pago y terminar sirviendo
    por suscripción. Cobrar por el primero sería inventar un gasto que no
    ocurrió — y al revés, perder uno que sí."""
    obs = Observation(project="p", route="image", requested_model="paid/gpt-image-2")
    obs.response_model = "gpt-image-2"  # cayó al de suscripción
    obs.image_count = 1

    assert obs.cost().charged is False

    obs2 = Observation(project="p", route="image", requested_model="gpt-image-2")
    obs2.response_model = "paid/gpt-image-2"  # subió al de pago
    obs2.image_count = 1

    assert obs2.cost().charged is True


def test_se_puede_fijar_a_mano_cuando_quien_llama_lo_sabe():
    """La deducción es un default, no una imposición: un provider que conozca su
    credencial puede declararla."""
    obs = Observation(project="p", route="chat", requested_model="gemini-3-flash")
    obs.response_model = "gemini-3-flash"
    obs.auth_mode = "api_key"
    obs.prompt_tokens = 1_000_000

    assert obs.cost().charged is True


def test_un_prefijo_no_declarado_no_inventa_un_cargo():
    """El sesgo va hacia NO cobrado, al revés que en `is_charged`: un falso
    positivo inventaría un gasto que no existe, y un reporte que exagera se
    descubre tarde y mal. Un modelo de pago sin declarar aparece como
    equivalente, que es visible."""
    assert load_pricing().auth_mode_for("otracosa/gpt-image-2") == "oauth"
