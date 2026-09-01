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


# ─── Tiers en rutas de imagen ─────────────────────────────────────────────────


def _tabla_imagen():
    """Mapa mínimo con dos modelos de imagen y uno de chat, para probar que la
    ruta —y no el tier— es la que exige la capacidad."""
    from src.modules.routing.tiers import ModelInfo, TierPolicy, TierTable

    return TierTable(
        policies={
            "fast": TierPolicy(name="fast", rank_by="latency_asc"),
            "cheap": TierPolicy(name="cheap", rank_by="cost_asc"),
        },
        models={
            "rapido-caro": ModelInfo(
                id="rapido-caro",
                family="openai",
                capabilities={"image": True},
                latency_s=1.0,
                cost=1.0,
                image_latency_s=15.0,
                image_cost=0.04,
            ),
            "lento-barato": ModelInfo(
                id="lento-barato",
                family="google",
                capabilities={"image": True},
                latency_s=99.0,
                cost=99.0,
                image_latency_s=90.0,
                image_cost=0.03,
            ),
            "solo-chat": ModelInfo(
                id="solo-chat",
                family="openai",
                capabilities={"chat": True},
                latency_s=0.5,
                cost=0.1,
            ),
        },
    )


def test_fast_en_imagen_ordena_por_latencia_de_imagen():
    """Un sistema que genera de a una y espera pide `fast` y recibe el más
    rápido MEDIDO EN IMAGEN — 15 s contra 90 s. Ordenarlo por la latencia de
    chat sería ordenar por otra cosa: son órdenes de magnitud distintos."""
    assert _tabla_imagen().resolve("fast", route="image") == ["rapido-caro", "lento-barato"]


def test_cheap_en_imagen_ordena_por_precio_POR_IMAGEN():
    """Los modelos de imagen se tarifan por unidad, no por token. Con el precio
    de token, gpt-image-2 heredaría la tarifa de su familia (11.25) y quedaría
    último por una cifra que no aplica a esta ruta."""
    assert _tabla_imagen().resolve("cheap", route="image") == ["lento-barato", "rapido-caro"]


def test_un_modelo_de_solo_chat_no_entra_en_una_ruta_de_imagen():
    """La ruta aporta su capacidad requerida. Es lo que permite que los tiers no
    repitan `require: [chat]` y sirvan igual en todas las rutas."""
    for tier in ("fast", "cheap"):
        assert "solo-chat" not in _tabla_imagen().resolve(tier, route="image")


def test_image_edit_exige_la_misma_capacidad_que_image():
    """Sin mapear `image_edit`, un tier sobre esa ruta no exigiría NADA y
    dejaría entrar modelos de sólo texto a una ruta de imagen."""
    from src.modules.routing.tiers import ROUTE_CAPABILITY

    assert ROUTE_CAPABILITY["image_edit"] == "image"
    assert _tabla_imagen().resolve("fast", route="image_edit") == ["rapido-caro", "lento-barato"]


def test_los_tiers_ya_no_repiten_la_capacidad_de_la_ruta():
    """Regresión de la política: `require: [chat]` en los tiers medibles los
    dejaba inservibles fuera de las rutas de texto — un modelo de imagen tiene
    `chat: False` y quedaba filtrado de `fast` y de `cheap`."""
    from src.modules.routing.tiers import load_tiers

    tabla = load_tiers()
    for nombre in ("cheap", "fast"):
        politica = tabla.policies.get(nombre)
        assert politica is not None
        assert "chat" not in politica.require, (
            f"{nombre} repite la capacidad de la ruta; volvería a excluir imagen"
        )
