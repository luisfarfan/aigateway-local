"""
El vigilante de la sesión de la app web de Gemini.

Lo que se prueba es el **flanco**, no la sonda. Una alerta que se repite en cada
barrido se ignora a los dos días, y una alerta ignorada es peor que ninguna:
da la sensación de estar cubierto sin estarlo.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.modules.backends import gemini_web_monitor as monitor


class FakeBackend:
    def __init__(self, *estados: tuple[bool, str]):
        self._estados = list(estados)
        self.checks = 0

    async def check_session(self) -> tuple[bool, str]:
        self.checks += 1
        return self._estados[min(self.checks - 1, len(self._estados) - 1)]


class FakeNotifier:
    name = "fake"

    def __init__(self, *, falla: bool = False):
        self.enviados: list[tuple[str, str]] = []
        self._falla = falla

    async def send(self, title: str, body: str) -> bool:
        if self._falla:
            raise RuntimeError("canal caído")
        self.enviados.append((title, body))
        return True


@pytest.fixture
def redis_falso(monkeypatch):
    """Redis en memoria: el flanco se guarda ahí porque API y worker son
    procesos distintos y cada uno avisaría por su cuenta."""
    store: dict[str, Any] = {}

    class FakeRedis:
        async def get(self, key: str):
            return store.get(key)

        async def set(self, key: str, value: str, **_: Any):
            store[key] = value

    monkeypatch.setattr(monitor, "get_redis", lambda: FakeRedis())
    return store


@pytest.mark.asyncio
async def test_el_primer_chequeo_no_avisa(redis_falso):
    """Si avisara, cada reinicio del gateway mandaría un mensaje — y el gateway
    se reinicia por muchas razones que no son que la cookie murió."""
    notifier = FakeNotifier()
    await monitor.check_once(FakeBackend((False, "expirada")), notifier)
    assert notifier.enviados == []


@pytest.mark.asyncio
async def test_avisa_una_sola_vez_al_expirar(redis_falso):
    """El aviso es por transición. Repetirlo en cada barrido —uno cada 30 min,
    para siempre— convierte la alerta en ruido de fondo."""
    backend = FakeBackend((True, "ok"), (False, "expirada"), (False, "expirada"))
    notifier = FakeNotifier()

    await monitor.check_once(backend, notifier)  # viva: registra, no avisa
    await monitor.check_once(backend, notifier)  # viva -> muerta: AVISA
    await monitor.check_once(backend, notifier)  # sigue muerta: calla

    assert len(notifier.enviados) == 1
    titulo, cuerpo = notifier.enviados[0]
    assert "EXPIRADA" in titulo
    # El aviso tiene que decir QUÉ hacer: quien lo recibe puede estar en la calle
    # y necesita saber si esto espera o no.
    assert "gemini_web_login.py" in cuerpo


@pytest.mark.asyncio
async def test_tambien_avisa_cuando_se_recupera(redis_falso):
    """Sin esto, quien recibió la alerta no sabe si su arreglo funcionó salvo
    que vaya a mirar Grafana."""
    backend = FakeBackend((False, "expirada"), (True, "ok"))
    notifier = FakeNotifier()

    await monitor.check_once(backend, notifier)  # primer chequeo: registra
    backend2 = FakeBackend((True, "ok"))
    await monitor.check_once(backend2, notifier)  # muerta -> viva: AVISA

    assert len(notifier.enviados) == 1
    assert "recuperada" in notifier.enviados[0][0]


@pytest.mark.asyncio
async def test_la_metrica_refleja_el_estado_actual(redis_falso):
    """El gauge es lo que ve Grafana. Si no se actualiza, el panel miente."""
    from src.core import metrics

    await monitor.check_once(FakeBackend((False, "expirada")), FakeNotifier())
    assert metrics.gemini_web_session_valid._value.get() == 0

    await monitor.check_once(FakeBackend((True, "ok")), FakeNotifier())
    assert metrics.gemini_web_session_valid._value.get() == 1


@pytest.mark.asyncio
async def test_sin_redis_no_avisa_en_vez_de_avisar_de_mas(redis_falso, monkeypatch):
    """Si el estado previo no se puede leer, se calla. Gritar en cada barrido
    porque Redis está caído sería alertar del problema equivocado."""

    class RedisRoto:
        async def get(self, key: str):
            raise RuntimeError("redis caído")

        async def set(self, key: str, value: str, **_: Any):
            raise RuntimeError("redis caído")

    monkeypatch.setattr(monitor, "get_redis", lambda: RedisRoto())
    notifier = FakeNotifier()
    alive = await monitor.check_once(FakeBackend((False, "expirada")), notifier)

    assert alive is False  # el chequeo igual se hizo
    assert notifier.enviados == []


@pytest.mark.asyncio
async def test_al_recuperarse_cierra_el_circuito_sin_esperar_a_que_venza(redis_falso, monkeypatch):
    """El cooldown por fallo de auth es de horas. Si la persona arregla la
    sesión a los cinco minutos, hacerle esperar el resto sería castigar el
    arreglo: el monitor devuelve el backend a la cadena en cuanto lo detecta."""
    cerrados: list[str] = []

    async def fake_close():
        cerrados.append("geminiweb/nano-banana-web")

    monkeypatch.setattr(monitor, "_close_circuit", fake_close)

    backend = FakeBackend((False, "expirada"))
    notifier = FakeNotifier()
    await monitor.check_once(backend, notifier)  # primer chequeo: registra

    await monitor.check_once(FakeBackend((True, "ok")), notifier)  # muerta -> viva

    assert cerrados == ["geminiweb/nano-banana-web"]


@pytest.mark.asyncio
async def test_no_toca_el_circuito_si_la_sesion_sigue_muerta(redis_falso, monkeypatch):
    """Cerrar el circuito de algo que sigue roto lo devolvería a la cadena para
    que falle otra vez — y cada fallo es una sesión nueva contra Google."""
    cerrados: list[str] = []
    monkeypatch.setattr(monitor, "_close_circuit", lambda: cerrados.append("x"))

    backend = FakeBackend((False, "expirada"), (False, "expirada"))
    await monitor.check_once(backend, FakeNotifier())
    await monitor.check_once(backend, FakeNotifier())

    assert cerrados == []
