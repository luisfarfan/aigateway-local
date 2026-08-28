"""
Costo, métricas y traza de una llamada.

El eje de casi todos estos tests es la misma distinción, que es fácil de perder
y cara cuando se pierde: **un costo desconocido no es un costo cero**. Un panel
que suma ceros por modelos sin tarifa muestra un gasto que nadie está pagando y
esconde la parte que sí.
"""

from __future__ import annotations

import pytest

from src.modules.observability.pricing import (
    PricingTable,
    cost_of_images,
    cost_of_tokens,
    load_pricing,
)
from src.modules.observability.recorder import AttemptRecord, Observation, observe

TABLE = PricingTable(
    version=1,
    models={"gemini-3-flash": {"input": 0.30, "output": 2.50}},
    families={"openai": {"input": 1.25, "output": 10.00}},
    images={"gpt-image-2": {"per_image": 0.04}},
    billing={"oauth": {"charged": False}, "api_key": {"charged": True}},
)


@pytest.fixture(autouse=True)
def sin_historico(monkeypatch):
    """Los tests no escriben en Postgres."""
    from src.modules.observability import recorder

    async def noop(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr(recorder, "_persist", noop)


# ─── Precio ───────────────────────────────────────────────────────────────────


def test_el_id_exacto_gana_sobre_la_familia():
    cost = cost_of_tokens(
        model="gemini-3-flash",
        family="openai",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        table=TABLE,
    )
    assert cost.source == "model"
    assert cost.equivalent_usd == pytest.approx(0.30)


def test_sin_id_exacto_cae_a_la_familia():
    """Un modelo nuevo no debe quedar sin precio sólo por haberse renombrado."""
    cost = cost_of_tokens(
        model="gpt-5.9-inedito",
        family="openai",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        table=TABLE,
    )
    assert cost.source == "family"
    assert cost.equivalent_usd == pytest.approx(1.25)


def test_un_modelo_sin_tarifa_no_cuesta_cero_sino_nada():
    cost = cost_of_tokens(
        model="?", family="?", prompt_tokens=1000, completion_tokens=1000, table=TABLE
    )
    assert cost.priced is False
    assert cost.amount_usd is None
    assert cost.equivalent_usd is None


def test_oauth_no_cobra_pero_deja_el_precio_de_lista():
    """El costo marginal contra suscripción es cero; el equivalente sirve para
    dimensionar el ahorro y comparar modelos."""
    cost = cost_of_tokens(
        model="gemini-3-flash",
        family="google",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        auth_mode="oauth",
        table=TABLE,
    )
    assert cost.amount_usd == 0.0
    assert cost.equivalent_usd == pytest.approx(0.30)
    assert cost.saved_usd == pytest.approx(0.30)


def test_un_modo_de_cobro_desconocido_se_asume_pagado():
    """Subestimar el gasto es el error caro de los dos."""
    cost = cost_of_tokens(
        model="gemini-3-flash",
        family="google",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        auth_mode="algo_nuevo",
        table=TABLE,
    )
    assert cost.charged is True
    assert cost.amount_usd == pytest.approx(0.30)


def test_la_imagen_se_tarifa_por_unidad():
    cost = cost_of_images(model="gpt-image-2", image_count=3, table=TABLE)
    assert cost.equivalent_usd == pytest.approx(0.12)
    assert cost.source == "image"


def test_la_tabla_del_repo_carga_y_cubre_los_modelos_conectados():
    """Si esto falla, el reporte de costos tiene huecos."""
    table = load_pricing()
    assert table.version >= 1
    for model in ("gemini-3-flash", "gpt-5.4-mini"):
        rate, source = table.rate_for(model, "unknown")
        assert rate is not None, f"{model} sin precio"
        assert source == "model"


# ─── Observación ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_una_observacion_sin_tocar_queda_como_fallo():
    """El sesgo va hacia reportar de más: si algo revienta y nadie marca el
    éxito, el registro tiene que decir que falló."""
    obs = Observation(project="p", route="chat", requested_model="m")
    async with observe(obs):
        pass
    assert obs.outcome == "upstream_error"


@pytest.mark.asyncio
async def test_una_excepcion_igual_queda_contabilizada():
    obs = Observation(project="p", route="chat", requested_model="m")
    with pytest.raises(RuntimeError):
        async with observe(obs):
            raise RuntimeError("boom")
    assert obs.outcome == "upstream_error"


@pytest.mark.asyncio
async def test_los_tokens_y_el_costo_llegan_a_las_metricas():
    from src.core import metrics

    def total() -> float:
        return metrics.llm_tokens_total.labels(
            "proj-metricas", "gemini-3-flash", "input"
        )._value.get()

    antes = total()
    obs = Observation(
        project="proj-metricas", route="chat", requested_model="gemini-3-flash", family="google"
    )
    async with observe(obs):
        obs.prompt_tokens = 120
        obs.completion_tokens = 30
        obs.succeeded(model="gemini-3-flash")

    assert total() - antes == 120


@pytest.mark.asyncio
async def test_un_modelo_sin_precio_se_cuenta_aparte():
    """Sin este contador, un hueco en `pricing.yaml` no se nota por ningún lado
    hasta que alguien pide el reporte."""
    from src.core import metrics

    def total() -> float:
        return metrics.llm_unpriced_total.labels("modelo-fantasma")._value.get()

    antes = total()
    obs = Observation(project="p", route="chat", requested_model="modelo-fantasma")
    async with observe(obs):
        obs.prompt_tokens = 10
        obs.succeeded(model="modelo-fantasma")

    assert total() - antes == 1


@pytest.mark.asyncio
async def test_un_acierto_de_cache_no_infla_el_gasto():
    """No consumió tokens ni costó nada: tiene que registrarse en cero."""
    obs = Observation(project="p", route="structured", requested_model="gemini-3-flash")
    async with observe(obs):
        obs.cache = "hit"
        obs.succeeded(model="gemini-3-flash")

    assert obs.prompt_tokens == 0
    assert obs.cost().equivalent_usd == 0.0


@pytest.mark.asyncio
async def test_los_intentos_del_guard_quedan_en_la_observacion():
    obs = Observation(project="p", route="structured", requested_model="gemini-3-flash")
    async with observe(obs):
        obs.attempts = [
            AttemptRecord(1, "schema_invalid", 0.5, 10, 5, "enum inválido"),
            AttemptRecord(2, "ok", 0.4, 12, 6),
        ]
        obs.succeeded(model="gemini-3-flash")

    assert [a.outcome for a in obs.attempts] == ["schema_invalid", "ok"]
