"""
Resolución de tiers a cadenas de modelos, sin red.

Lo que se fija: que un tier es una POLÍTICA sobre el mapa, no una lista. `cheap`
y `fast` se derivan de medición (costo, latencia); `smart` es orden curado. Y que
la intersección con la capacidad de la ruta funciona: `cheap` en websearch es
barato Y con websearch.
"""

from __future__ import annotations

from src.modules.routing.tiers import ModelInfo, TierPolicy, TierTable


def _table() -> TierTable:
    return TierTable(
        policies={
            "cheap": TierPolicy("cheap", require=("chat",), rank_by="cost_asc"),
            "fast": TierPolicy("fast", require=("chat",), rank_by="latency_asc"),
            "smart": TierPolicy("smart", require=("chat",), order=("opus", "gpt", "sonnet")),
            "cheap_no_local": TierPolicy(
                "cheap_no_local",
                require=("chat",),
                rank_by="cost_asc",
                exclude_family=frozenset({"local"}),
            ),
        },
        models={
            "opus": ModelInfo("opus", "anthropic", {"chat": True, "vision": True}, 3.0, 18.0),
            "gpt": ModelInfo("gpt", "openai", {"chat": True, "tools": True}, 1.5, 11.0),
            "sonnet": ModelInfo("sonnet", "anthropic", {"chat": True}, 2.0, 18.0),
            "flash": ModelInfo("flash", "google", {"chat": True, "websearch": True}, 1.6, 0.5),
            "localqwen": ModelInfo("localqwen", "local", {"chat": True}, 8.0, 0.0),
        },
    )


def test_cheap_ordena_por_costo():
    """El más barato primero, medido — no un nombre elegido a mano."""
    chain = _table().resolve("cheap", route="chat")
    assert chain[0] == "localqwen"  # costo 0
    assert chain.index("flash") < chain.index("gpt") < chain.index("opus")


def test_fast_ordena_por_latencia():
    chain = _table().resolve("fast", route="chat")
    assert chain[0] == "gpt"  # 1.5 s, el más rápido


def test_smart_respeta_el_orden_curado():
    """Calidad = juicio, no medición. El orden es el que se puso."""
    assert _table().resolve("smart", route="chat") == ["opus", "gpt", "sonnet"]


def test_exclude_family_saca_lo_local_de_fast_y_cheap():
    chain = _table().resolve("cheap_no_local", route="chat")
    assert "localqwen" not in chain


def test_la_ruta_añade_su_capacidad():
    """`cheap` en websearch = barato Y con websearch. Sólo flash lo tiene."""
    assert _table().resolve("cheap", route="websearch") == ["flash"]


def test_un_tier_sin_modelos_para_la_ruta_devuelve_vacio():
    """Honesto: si el mapa no cubre esa capacidad, no inventa. El router decide
    caer a la ruta normal."""
    # image no lo tiene ningún modelo del fixture
    assert _table().resolve("cheap", route="image") == []


def test_smart_deja_caer_un_id_curado_que_ya_no_existe():
    """Un modelo retirado del `order` no rompe: se ignora. Y `unclassified` lo
    señala para revisar."""
    t = _table()
    t.policies["smart"] = TierPolicy("smart", require=("chat",), order=("opus", "retirado", "gpt"))
    assert t.resolve("smart", route="chat") == ["opus", "gpt"]
    assert t.unclassified("smart") == ["retirado"]


def test_is_tier_distingue_tier_de_modelo():
    t = _table()
    assert t.is_tier("cheap")
    assert not t.is_tier("gemini-3-flash")
