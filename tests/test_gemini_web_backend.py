"""
El backend de último recurso: la app web de Gemini.

No se prueba contra Google. Lo que importa acá no es que la app web responda
—eso ya se midió a mano— sino que este backend se comporte como un **candidato
de cadena**: que declare lo que no puede hacer con el error que el routing
entiende, que no filtre las fotos del cliente a disco, y que la ausencia de la
dependencia AGPL sea un candidato inservible y no una caída.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from src.modules.backends.base import BackendCapabilityError
from src.modules.backends.gemini_web import MODEL_ID, GeminiWebBackend, _as_temp_files
from src.modules.backends.registry import BackendRegistry
from src.modules.providers.cliproxy.translate import InputImage


class FakeResponse:
    def __init__(self, text: str = "", images: list[Any] | None = None):
        self.text = text
        self.images = images or []


def build(**kw: Any) -> GeminiWebBackend:
    return GeminiWebBackend(secure_1psid="psid", secure_1psidts="psidts", **kw)


# ─── Capacidades ausentes ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("nombre", "llamada"),
    [
        ("search", lambda b: b.search([{"role": "user", "content": "x"}], model=MODEL_ID)),
        ("embed", lambda b: b.embed(["x"], model=MODEL_ID)),
    ],
)
async def test_lo_que_no_puede_lo_dice_con_el_error_del_routing(nombre: str, llamada: Any):
    """`BackendCapabilityError` y no un 500: es un candidato mal elegido, no un
    fallo de la petición. La cadena tiene que pasar al siguiente."""
    with pytest.raises(BackendCapabilityError):
        await llamada(build())


@pytest.mark.asyncio
async def test_chat_con_tools_se_rechaza_en_vez_de_ignorarlos():
    """Aceptar `tools` y devolver texto suelto sería peor que fallar: el cliente
    esperaría una decisión del agente y recibiría prosa, sin nada que lo
    explique. Se declara la incapacidad y el routing prueba otro."""
    with pytest.raises(BackendCapabilityError):
        await build().chat(
            [{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )


@pytest.mark.asyncio
async def test_image_edit_sin_imagenes_falla_antes_de_salir_a_la_red():
    with pytest.raises(BackendCapabilityError):
        await build().image_edit("x", images=[])


@pytest.mark.asyncio
async def test_sin_la_dependencia_agpl_es_candidato_inservible_no_una_caida(monkeypatch):
    """`gemini_webapi` es opcional. Si no está instalada, el backend debe
    declararse incapaz para que la cadena siga — no reventar la petición."""
    import builtins

    real_import = builtins.__import__

    def sin_gemini(name: str, *args: Any, **kw: Any):
        if name == "gemini_webapi":
            raise ImportError("no module named gemini_webapi")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", sin_gemini)
    with pytest.raises(BackendCapabilityError, match="no está instalado"):
        await build().image("un cubo")


@pytest.mark.asyncio
async def test_una_cookie_invalida_no_pasa_por_valida(monkeypatch):
    """REGRESIÓN. `init()` de la librería NO levanta con cookies inválidas:
    medido, con dos cookies basura devolvió éxito. Intenta, en orden, su caché
    en disco, las cookies dadas y las del navegador local — y cualquiera de las
    otras dos tapa que las nuestras murieron.

    El daño real no es una petición fallida: es que el chequeo de sesión diría
    "viva" para siempre y el aviso de expiración no llegaría NUNCA. Peor que no
    tener monitor, porque parece que lo hay.

    `access_token` vacío es la señal de que no hubo sesión autenticada.
    """

    class ClienteSinToken:
        access_token = ""

        def __init__(self, *a: Any, **kw: Any):
            pass

        async def init(self, **kw: Any):
            return None  # no levanta: exactamente el comportamiento real

        async def close(self):
            return None

    fake = type(sys)("gemini_webapi")
    fake.GeminiClient = ClienteSinToken
    monkeypatch.setitem(sys.modules, "gemini_webapi", fake)

    alive, detalle = await build().check_session()
    assert alive is False
    assert "no autentica" in detalle


@pytest.mark.asyncio
async def test_una_cookie_valida_si_pasa(monkeypatch):
    """La otra mitad del par: con token, la sesión se declara viva."""

    class ClienteConToken:
        access_token = "at-123"

        def __init__(self, *a: Any, **kw: Any):
            pass

        async def init(self, **kw: Any):
            return None

        async def close(self):
            return None

    fake = type(sys)("gemini_webapi")
    fake.GeminiClient = ClienteConToken
    monkeypatch.setitem(sys.modules, "gemini_webapi", fake)

    alive, _ = await build().check_session()
    assert alive is True


@pytest.mark.asyncio
async def test_la_sonda_no_abre_una_sesion_nueva_en_cada_chequeo():
    """REGRESIÓN, y de las caras. La primera versión forzaba cliente nuevo en
    cada chequeo. Cada `init()` abre una sesión autenticada distinta contra
    Google: medido en el caché de la librería, **11 sesiones en hora y media**
    con la misma cuenta desde un servidor. La cuenta terminó invalidada.

    O sea: la sonda que vigilaba la sesión era la que la mataba. El chequeo
    tiene que reutilizar el cliente vivo y refrescar el estado sobre él.
    """
    inits = 0

    class ClienteVivo:
        access_token = "at-123"
        account_status = type("E", (), {"name": "AVAILABLE"})()

        def __init__(self, *a: Any, **kw: Any):
            nonlocal inits
            inits += 1

        async def init(self, **kw: Any):
            return None

        async def _fetch_user_status(self):
            return None

        async def close(self):
            return None

    backend = build()
    backend._client = ClienteVivo()  # ya hay sesión viva
    inits = 0  # no contar la de arriba

    for _ in range(5):
        alive, _detalle = await backend.check_session()
        assert alive is True

    assert inits == 0, f"abrió {inits} sesiones nuevas; debía reutilizar la viva"


@pytest.mark.asyncio
async def test_una_sesion_muerta_no_pasa_por_viva_aunque_conserve_el_token():
    """REGRESIÓN, y de las que más duelen: el monitor MINTIENDO.

    `access_token` se llena en el `init()` inicial y **no se limpia** cuando
    Google invalida la sesión después. Al reutilizar el cliente vivo —necesario
    para no abrir sesiones nuevas en cada chequeo— mirar el token dejó al
    monitor encadenando 30 chequeos "ok" sobre una cuenta ya muerta, con la
    métrica en 1 y sin un solo aviso.

    Lo que sí refresca `_fetch_user_status()` es `account_status`. Ese decide.
    """

    class Estado:
        name = "UNAUTHENTICATED"

    class ClienteMuertoConToken:
        access_token = "at-viejo-que-nadie-limpio"
        account_status = Estado()

        async def _fetch_user_status(self):
            return None

        async def close(self):
            return None

    backend = build()
    backend._client = ClienteMuertoConToken()

    alive, detalle = await backend.check_session()
    assert alive is False
    assert "UNAUTHENTICATED" in detalle


@pytest.mark.asyncio
async def test_solo_available_cuenta_como_sesion_util():
    """`TOS_PENDING`, `ACCOUNT_REJECTED`, `LOCATION_REJECTED`… cada uno es un
    motivo distinto para no poder trabajar, y ninguno se arregla reintentando.
    Sólo `AVAILABLE` significa que la sesión sirve."""

    class ClienteConEstado:
        access_token = "at"

        def __init__(self, nombre: str):
            self.account_status = type("E", (), {"name": nombre})()

        async def _fetch_user_status(self):
            return None

        async def close(self):
            return None

    for nombre, esperado in [
        ("AVAILABLE", True),
        ("TOS_PENDING", False),
        ("ACCOUNT_REJECTED", False),
        ("ACCESS_TEMPORARILY_UNAVAILABLE", False),
    ]:
        backend = build()
        backend._client = ClienteConEstado(nombre)
        alive, _ = await backend.check_session()
        assert alive is esperado, nombre


@pytest.mark.asyncio
async def test_si_la_sesion_viva_pierde_el_token_se_declara_muerta():
    """Reutilizar el cliente no puede significar dar por buena una sesión que
    murió: `_fetch_user_status()` la refresca y `access_token` decide."""

    class ClienteQueMuere:
        """Sin `account_status`: obliga a caer al respaldo por token."""

        account_status = None

        def __init__(self):
            self.access_token = "at-123"

        async def _fetch_user_status(self):
            self.access_token = ""  # Google la invalidó

        async def close(self):
            return None

    backend = build()
    backend._client = ClienteQueMuere()

    alive, detalle = await backend.check_session()
    assert alive is False
    assert "dejó de autenticar" in detalle


@pytest.mark.asyncio
async def test_el_fallo_de_auth_saca_el_backend_de_la_cadena_por_horas(monkeypatch):
    """Una cookie muerta NO se cura sola: necesita que alguien abra el navegador.

    Reintentar mientras tanto no sólo pierde tiempo — cada intento abre una
    sesión nueva contra Google, y ese churn fue lo que invalidó la cuenta la
    primera vez. El error tiene que llevar una espera larga para que el routing
    abra el circuito, en vez de dejar que el contador de fallos (que en imagen
    nunca se llena) lo reintente en cada petición.
    """
    from src.modules.backends.gemini_web import AUTH_FAILURE_COOLDOWN_S
    from src.modules.providers.cliproxy.errors import CliproxyTransportError

    class ClienteSinToken:
        access_token = ""

        def __init__(self, *a: Any, **kw: Any):
            pass

        async def init(self, **kw: Any):
            return None

        async def close(self):
            return None

    fake = type(sys)("gemini_webapi")
    fake.GeminiClient = ClienteSinToken
    monkeypatch.setitem(sys.modules, "gemini_webapi", fake)

    with pytest.raises(CliproxyTransportError) as exc:
        await build().image("un cubo")
    assert exc.value.retry_after_s == AUTH_FAILURE_COOLDOWN_S
    assert AUTH_FAILURE_COOLDOWN_S >= 3600, "una espera corta reabriría el churn"


# ─── Datos del cliente ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_las_fotos_se_borran_del_disco_al_terminar():
    """La librería sube archivos por ruta, así que hay que escribirlas. Son
    fotos de producto de un cliente: no pueden quedar en /tmp después."""
    imgs = [InputImage(content=b"\x89PNG-A", filename="producto.png")]
    async with _as_temp_files(imgs) as paths:
        assert len(paths) == 1
        dentro = Path(paths[0])
        assert dentro.read_bytes() == b"\x89PNG-A"
    assert not dentro.exists()
    assert not dentro.parent.exists()


# ─── Ruteo ────────────────────────────────────────────────────────────────────


def test_el_prefijo_lo_manda_a_su_backend_y_se_le_quita():
    """Mismo contrato que `ollama/`: el backend recibe el id sin prefijo."""
    cloud, web = object(), object()
    reg = BackendRegistry(cloud=cloud, gemini_web=web)  # type: ignore[arg-type]

    resuelto = reg.resolve("geminiweb/nano-banana-web")
    assert resuelto is not None
    assert resuelto.backend is web
    assert resuelto.model == "nano-banana-web"


def test_sin_backend_configurado_el_candidato_se_salta():
    """Apagado por defecto. Un candidato que no se puede servir devuelve None y
    el routing pasa al siguiente, en vez de fallar la petición entera."""
    reg = BackendRegistry(cloud=object())  # type: ignore[arg-type]
    assert reg.resolve("geminiweb/nano-banana-web") is None
    assert reg.gemini_web_available is False


def test_no_se_come_los_modelos_de_los_otros_backends():
    """Regresión: el prefijo nuevo no debe capturar nada que no le toque."""
    cloud, local, web = object(), object(), object()
    reg = BackendRegistry(cloud=cloud, local=local, gemini_web=web)  # type: ignore[arg-type]

    assert reg.resolve("gpt-image-2").backend is cloud  # type: ignore[union-attr]
    assert reg.resolve("ollama/qwen2.5:7b").backend is local  # type: ignore[union-attr]


def test_la_edicion_lidera_con_lo_verificado_y_deja_la_cookie_al_final():
    """En `image_edit` hay una foto de producto que preservar, y eso manda.

    `gpt-image-2` es el único verificado conservando la identidad del producto;
    la app web lo reinterpretó. Una imagen barata que no es del producto que
    vendés no sirve de nada, así que acá la fidelidad gana al costo. Si algún
    día esto cambia, que sea una decisión y no un descuido.
    """
    from src.modules.routing.config import load_routing

    cadena = load_routing().candidates("image_edit")
    assert cadena[0] == "gpt-image-2", cadena
    assert cadena[-1] == "geminiweb/nano-banana-web", cadena


def test_la_generacion_prefiere_lo_gratis_y_reserva_la_cuota_medida():
    """En `image` (texto -> imagen) NO hay producto que preservar, así que la
    fidelidad deja de decidir y decide el costo.

    `gpt-image-2` va último aquí por costo de oportunidad, no por calidad: su
    ventana de 5 h se comparte con el texto de todo el gateway, y vale más en
    `image_edit`. Antes de llegar a él se gastan las dos fuentes que no la
    tocan.
    """
    from src.modules.routing.config import load_routing

    cadena = load_routing().candidates("image")
    assert cadena[-1] == "gpt-image-2", cadena
    assert cadena.index("geminiweb/nano-banana-web") < cadena.index("gpt-image-2"), cadena
