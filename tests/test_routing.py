"""
Cadenas de modelos, fallback y circuit breaker.

El eje: **no todo fallo justifica probar otro modelo**. Un 429 sí, porque otro
modelo tiene otra cuota. Un 400 no, porque la petición está mal y el siguiente
la va a rechazar igual — reintentar sólo gasta cuota y latencia para llegar al
mismo error.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.modules.backends.registry import BackendRegistry
from src.modules.providers.cliproxy.errors import (
    CliproxyNoCredentialError,
    CliproxyRequestError,
    CliproxyRetryableError,
)
from src.modules.routing import errors as routing_errors
from src.modules.routing.config import BreakerPolicy, RoutingTable, load_routing
from src.modules.routing.executor import NoCandidatesError, run_with_fallback
from src.modules.structured.guard import InvalidStructuredOutput

TABLE = RoutingTable(
    version=1,
    routes={"chat": ["modelo-a", "modelo-b", "modelo-c"], "vacia": []},
    fallback_on=frozenset({"upstream_unavailable", "no_credential", "invalid_output"}),
    breaker=BreakerPolicy(),
)


class FakeBreaker:
    """Breaker en memoria. El real usa Redis; la lógica que se prueba es la misma."""

    def __init__(self, abiertos: set[str] | None = None) -> None:
        self.abiertos = abiertos or set()
        self.fallos: list[str] = []
        self.aciertos: list[str] = []

    async def is_open(self, model: str) -> bool:
        return model in self.abiertos

    async def record_failure(self, model: str) -> bool:
        self.fallos.append(model)
        return False

    async def record_success(self, model: str) -> None:
        self.aciertos.append(model)


def caller(fallos: dict[str, Exception]):
    """Devuelve una función que falla para los modelos indicados."""
    llamados: list[str] = []

    async def call(model: str) -> str:
        llamados.append(model)
        if model in fallos:
            raise fallos[model]
        return f"respuesta-de-{model}"

    call.llamados = llamados  # type: ignore[attr-defined]
    return call


# ─── Cadenas ──────────────────────────────────────────────────────────────────


def test_el_modelo_pedido_encabeza_la_cadena():
    """Es una elección del cliente: se respeta, y el resto queda como red."""
    assert TABLE.candidates("chat", "modelo-c") == ["modelo-c", "modelo-a", "modelo-b"]


def test_no_se_repite_un_modelo_ya_presente():
    assert TABLE.candidates("chat", "modelo-a") == ["modelo-a", "modelo-b", "modelo-c"]


def test_una_ruta_desconocida_solo_prueba_lo_pedido():
    assert TABLE.candidates("inexistente", "modelo-x") == ["modelo-x"]


def test_la_tabla_del_repo_tiene_cadena_para_cada_ruta():
    table = load_routing()
    for route in ("chat", "websearch", "structured", "image"):
        assert table.candidates(route), f"{route} sin modelos"


# ─── Clasificación compartida ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("exc", "kind", "status"),
    [
        (CliproxyRetryableError("429"), "upstream_unavailable", 503),
        (CliproxyNoCredentialError("auth_not_found"), "no_credential", 502),
        (CliproxyRequestError("mal"), "invalid_request", 400),
        (InvalidStructuredOutput("no valida", []), "invalid_output", 422),
    ],
)
def test_un_fallo_tiene_un_solo_nombre(exc: Exception, kind: str, status: int):
    """El código que ve el cliente y el que decide el fallback salen del mismo
    sitio: si divergieran, el cliente reintentaría lo que el routing ya descartó."""
    assert routing_errors.kind_of(exc) == kind
    assert routing_errors.http_status_of(kind) == status


# ─── Fallback ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sin_fallos_responde_el_primero():
    call = caller({})
    result = await run_with_fallback(call, route="chat", table=TABLE, breaker=FakeBreaker())

    assert result.model == "modelo-a"
    assert result.fell_back is False
    assert call.llamados == ["modelo-a"]


@pytest.mark.asyncio
async def test_un_429_cae_al_siguiente():
    breaker = FakeBreaker()
    call = caller({"modelo-a": CliproxyRetryableError("429")})
    result = await run_with_fallback(call, route="chat", table=TABLE, breaker=breaker)

    assert result.model == "modelo-b"
    assert result.fell_back is True
    assert result.first_choice == "modelo-a"
    assert breaker.fallos == ["modelo-a"]
    assert breaker.aciertos == ["modelo-b"]


@pytest.mark.asyncio
async def test_un_400_no_cae_al_siguiente():
    """El siguiente modelo rechazaría la misma petición. Se corta acá."""
    call = caller({"modelo-a": CliproxyRequestError("body inválido")})
    with pytest.raises(CliproxyRequestError):
        await run_with_fallback(call, route="chat", table=TABLE, breaker=FakeBreaker())

    assert call.llamados == ["modelo-a"]


@pytest.mark.asyncio
async def test_sin_credencial_tambien_cae_al_siguiente():
    """Que ninguna cuenta cubra un modelo es permanente para *ese* modelo, no
    para la petición: otro de la cadena puede servirla."""
    call = caller({"modelo-a": CliproxyNoCredentialError("auth_not_found")})
    result = await run_with_fallback(call, route="chat", table=TABLE, breaker=FakeBreaker())
    assert result.model == "modelo-b"


@pytest.mark.asyncio
async def test_un_modelo_con_el_circuito_abierto_ni_se_intenta():
    """Descubrir en cada petición que un modelo está caído cuesta una llamada
    perdida por petición."""
    call = caller({})
    breaker = FakeBreaker(abiertos={"modelo-a"})
    result = await run_with_fallback(call, route="chat", table=TABLE, breaker=breaker)

    assert result.model == "modelo-b"
    assert call.llamados == ["modelo-b"]
    assert result.attempts[0].outcome == "skipped_open"


@pytest.mark.asyncio
async def test_si_toda_la_cadena_falla_se_levanta_el_ultimo_error():
    """El último describe el estado actual; el primero escondería que se
    intentaron los demás."""
    call = caller(
        {
            "modelo-a": CliproxyRetryableError("primero"),
            "modelo-b": CliproxyRetryableError("segundo"),
            "modelo-c": CliproxyNoCredentialError("ultimo"),
        }
    )
    with pytest.raises(CliproxyNoCredentialError, match="ultimo"):
        await run_with_fallback(call, route="chat", table=TABLE, breaker=FakeBreaker())


@pytest.mark.asyncio
async def test_toda_la_cadena_abierta_es_distinto_de_todos_fallaron():
    """Acá no se pudo ni intentar. Merece su propio error para no confundir un
    upstream caído con un breaker demasiado agresivo."""
    call = caller({})
    breaker = FakeBreaker(abiertos={"modelo-a", "modelo-b", "modelo-c"})
    with pytest.raises(NoCandidatesError, match="circuito abierto"):
        await run_with_fallback(call, route="chat", table=TABLE, breaker=breaker)


@pytest.mark.asyncio
async def test_una_ruta_sin_modelos_falla_rapido():
    call = caller({})
    with pytest.raises(NoCandidatesError, match="no tiene modelos"):
        await run_with_fallback(call, route="vacia", table=TABLE, breaker=FakeBreaker())


@pytest.mark.asyncio
async def test_el_fallback_queda_registrado_intento_por_intento():
    call = caller({"modelo-a": CliproxyRetryableError("429")})
    result = await run_with_fallback(call, route="chat", table=TABLE, breaker=FakeBreaker())

    assert [(a.model, a.outcome) for a in result.attempts] == [
        ("modelo-a", "upstream_unavailable"),
        ("modelo-b", "ok"),
    ]


@pytest.mark.asyncio
async def test_un_candidato_sin_backend_no_tapa_el_error_real():
    """Si el último de la cadena es un modelo cuyo backend no está configurado,
    su `unsupported_capability` describe la configuración, no el problema.
    Levantarlo escondería el 429 que causó todo."""
    from src.modules.backends.base import BackendCapabilityError

    table = RoutingTable(
        version=1,
        routes={"chat": ["modelo-a", "sin-backend"]},
        fallback_on=frozenset({"upstream_unavailable", "unsupported_capability"}),
    )
    call = caller(
        {
            "modelo-a": CliproxyRetryableError("429 cuota agotada"),
            "sin-backend": BackendCapabilityError("no hay backend"),
        }
    )
    with pytest.raises(CliproxyRetryableError, match="429"):
        await run_with_fallback(call, route="chat", table=table, breaker=FakeBreaker())


# ─── Watchdog ─────────────────────────────────────────────────────────────────


class ProbeClient:
    """Backend falso para el watchdog."""

    name = "fake"

    def __init__(self, vivos: set[str]) -> None:
        self._vivos = vivos

    async def chat(self, messages: Any, *, model: str, max_tokens: int) -> Any:
        if model not in self._vivos:
            raise CliproxyNoCredentialError("token revocado")
        return object()


def probe_registry(vivos: set[str]) -> BackendRegistry:
    return BackendRegistry(cloud=ProbeClient(vivos))


class RecordingBreaker(FakeBreaker):
    def __init__(self) -> None:
        super().__init__()
        self.abiertos_por_watchdog: list[str] = []
        self.cerrados: list[str] = []

    async def open(self, model: str, seconds: int, *, reason: str) -> None:
        self.abiertos_por_watchdog.append(model)

    async def close(self, model: str) -> None:
        self.cerrados.append(model)


@pytest.mark.asyncio
async def test_el_watchdog_prueba_en_vez_de_listar():
    """Medido en F0: `GET /v1/models` anunció 62 modelos y 32 devolvían
    `token revoked` o `403`. Listar no es prueba de vida."""
    from src.modules.routing.watchdog import sweep

    breaker = RecordingBreaker()
    results = await sweep(probe_registry({"modelo-a"}), TABLE, breaker, open_for_s=120)

    assert results == {"modelo-a": True, "modelo-b": False, "modelo-c": False}
    assert breaker.cerrados == ["modelo-a"]
    assert sorted(breaker.abiertos_por_watchdog) == ["modelo-b", "modelo-c"]


@pytest.mark.asyncio
async def test_el_watchdog_no_sondea_modelos_de_solo_imagen():
    """Bug encontrado corriéndolo: la sonda es una llamada de chat, y
    `gpt-image-2` la rechaza con 503 aunque esté vivo — sólo existe en
    /v1/images/generations. Sondearlo lo marcaba muerto y abría su circuito,
    rompiendo la ruta de imagen entera."""
    from src.modules.routing.watchdog import sweep

    table = RoutingTable(
        version=1,
        routes={"chat": ["modelo-a"], "image": ["solo-imagen"]},
        watchdog_skip_routes=frozenset({"image"}),
        breaker=BreakerPolicy(),
    )
    breaker = RecordingBreaker()
    results = await sweep(probe_registry({"modelo-a"}), table, breaker, open_for_s=120)

    assert "solo-imagen" not in results
    assert breaker.abiertos_por_watchdog == []


def test_un_modelo_en_dos_rutas_sigue_siendo_sondeable():
    """Si aparece además en una ruta sondeable, responder a chat sí prueba que
    la credencial sirve."""
    table = RoutingTable(
        version=1,
        routes={"chat": ["compartido"], "image": ["compartido", "solo-imagen"]},
        watchdog_skip_routes=frozenset({"image"}),
    )
    assert table.probeable_models() == ["compartido"]


# ─── Conformidad de backends ──────────────────────────────────────────────────


@pytest.mark.parametrize("factory", ["cliproxy", "ollama"])
def test_los_backends_reales_cumplen_el_protocolo(factory: str):
    """Bug encontrado en vivo: `CliproxyClient` no tenía `name`, y el router
    reventaba con un AttributeError que el routing clasificó como
    `upstream_error` — que no está en `fallback_on`, así que ni siquiera cayó al
    siguiente modelo. Los dobles de test sí lo tenían, por eso no se vio.

    `Backend` es `runtime_checkable`, así que esto compara contra la
    implementación real y no contra un doble."""
    from src.modules.backends.base import Backend
    from src.modules.backends.ollama import OllamaBackend
    from src.modules.providers.cliproxy.client import CliproxyClient

    backend = (
        CliproxyClient(base_url="http://x", api_key="k")
        if factory == "cliproxy"
        else OllamaBackend(base_url="http://x")
    )
    assert isinstance(backend, Backend)
    assert backend.name
