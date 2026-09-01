"""
Backend de último recurso: la app web de Gemini, por cookie de sesión.

Existe para un caso concreto y medido: cuando la ruta de imagen se queda sin
nada. `gemini-3.1-flash-image` por OAuth de antigravity choca contra un tope
SEMANAL (medido: 429 con reset a ~116 h), y `gpt-image-2` depende de que la
suscripción Codex Plus siga viva. Si las dos caen, sin esto la ruta de imagen
no tiene fondo.

Va **último en la cadena** y no por preferencia estética. Tres razones medidas:

  * **Fidelidad menor.** Editando la misma foto de producto, `gpt-image-2`
    conservó la taza original; esta vía la reinterpretó —asa distinta,
    proporción más alta—. Para un catálogo eso es el fallo que no se puede
    tener: el cliente compra lo que ve.
  * **Es ingeniería inversa.** No hay contrato de API. Cuando Google cambie la
    app web, esto se rompe sin aviso y sin changelog.
  * **La credencial es una cookie de sesión de una cuenta de Google**, no una
    API key revocable. Si algo sale mal, el alcance es la cuenta entera.

Sobre la licencia: `gemini_webapi` es AGPL-3.0. Por eso es una dependencia
OPCIONAL (`pip install -e ".[geminiweb]"`) y el import es perezoso — sin
instalarla, este backend no existe y las cadenas siguen igual. Para uso privado
en red local no dispara obligaciones; si algún día el gateway se distribuye o se
expone a terceros, hay que revisarlo.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import structlog

from src.modules.backends.base import BackendCapabilityError
from src.modules.providers.cliproxy.errors import (
    CliproxyRetryableError,
    CliproxyTransportError,
)
from src.modules.providers.cliproxy.families import Family
from src.modules.providers.cliproxy.translate import (
    EmbeddingResult,
    InputImage,
    LLMResult,
    Message,
)

log = structlog.get_logger(__name__)

# Modelo único: la app web no expone ids. El nombre es una etiqueta para las
# cadenas y las trazas, no algo que viaje al upstream.
MODEL_ID = "nano-banana-web"
# Cómo aparece en las cadenas de `routing.yaml`, con el prefijo de backend. El
# breaker indexa por ese id, así que el monitor lo necesita para cerrar el
# circuito cuando la sesión vuelve.
CHAIN_MODEL_ID = "geminiweb/nano-banana-web"

# Cuánto se saca de la cadena cuando la cookie deja de autenticar.
#
# Largo a propósito: esto NO se cura solo. Hace falta que una persona abra el
# navegador. Reintentar mientras tanto no sólo pierde tiempo — cada intento abre
# una sesión nueva contra Google, y ese churn es lo que invalidó la cuenta la
# primera vez. El circuito lo cierra el monitor en cuanto detecta que volvió,
# así que este número es un techo, no una espera obligatoria.
AUTH_FAILURE_COOLDOWN_S = 6 * 3600


class GeminiWebBackend:
    """Adaptador de la app web de Gemini al protocolo `Backend`.

    El cliente se crea perezosamente y se reutiliza: `init()` hace una llamada
    de red y rehacerla en cada petición sería pagar un round-trip de más.
    """

    def __init__(
        self,
        *,
        secure_1psid: str,
        secure_1psidts: str,
        timeout_s: float = 300.0,
        cookie_cache_dir: str | None = None,
    ) -> None:
        self._psid = secure_1psid
        self._psidts = secure_1psidts
        self._timeout = timeout_s
        self._client: Any = None

        # La librería guarda las cookies YA ROTADAS en un caché, y por defecto lo
        # pone en /tmp — que systemd-tmpfiles vacía en cada arranque. El efecto
        # es un ciclo que mata la sesión sola: al reiniciar la máquina se pierden
        # las cookies frescas y se vuelve a la semilla del `.env`, que para
        # entonces lleva días envejeciendo. Se mueve a un sitio persistente.
        if cookie_cache_dir:
            import os
            from pathlib import Path

            Path(cookie_cache_dir).mkdir(parents=True, exist_ok=True)
            Path(cookie_cache_dir).chmod(0o700)  # son credenciales
            os.environ["GEMINI_COOKIE_PATH"] = cookie_cache_dir

    @property
    def name(self) -> str:
        return "gemini_web"

    async def family_of(self, model: str) -> Family:
        return Family.GOOGLE

    async def models(self, **_: Any) -> list[dict[str, Any]]:
        """Un solo modelo: la app web no expone un catálogo.

        Se anuncia con el prefijo de backend porque así es como lo nombran las
        cadenas, y es el id con el que el prober lo guarda en el mapa. Sin esto
        el backend no existe para los tiers y `fast`/`cheap` no pueden elegirlo.
        """
        return [{"id": CHAIN_MODEL_ID, "owned_by": "gemini_web"}]

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from gemini_webapi import GeminiClient
        except ImportError as exc:
            # Dependencia opcional ausente: es un candidato inservible, no un
            # fallo de la petición. El routing debe pasar al siguiente.
            raise BackendCapabilityError(
                "gemini_webapi no está instalado; "
                'instalar con `pip install -e ".[geminiweb]"` o quitar '
                "geminiweb/ de las cadenas"
            ) from exc

        client = GeminiClient(self._psid, self._psidts)
        try:
            await client.init(timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 — la librería no tipa sus fallos
            # Cookie vencida o rotada: se reporta como reintentable porque un
            # refresco de sesión lo arregla sin tocar la petición.
            raise CliproxyTransportError(
                f"no se pudo iniciar sesión en la app web de Gemini: {str(exc)[:200]}"
            ) from exc

        # `init()` NO levanta con cookies inválidas — medido, y de la peor forma:
        # con dos cookies basura devolvió éxito. La librería intenta, en orden,
        # su caché en disco, las cookies dadas, y por último las del navegador
        # local; cualquiera de las otras dos puede tapar que las nuestras ya no
        # sirven. Con eso, un chequeo de sesión daría "viva" para siempre y el
        # aviso de expiración no llegaría nunca.
        #
        # `access_token` es el discriminante: sin sesión autenticada queda vacío.
        if not client.access_token:
            await _quiet_close(client)
            raise CliproxyTransportError(
                "la cookie de la app web de Gemini no autentica "
                "(sesión expirada o revocada): correr `python scripts/gemini_web_login.py`",
                retry_after_s=AUTH_FAILURE_COOLDOWN_S,
            )

        self._client = client
        return client

    async def _generate(self, prompt: str, *, files: list[str] | None = None) -> Any:
        client = await self._ensure_client()
        try:
            return await client.generate_content(prompt, files=files or [])
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            # La app web limita por ráfaga sin devolver un status HTTP claro.
            # Se clasifica como reintentable para que el breaker lo trate como
            # saturación y no como un modelo muerto.
            raise CliproxyRetryableError(
                f"app web de Gemini: {msg[:200]}", status_code=503
            ) from exc

    # ── Capacidades ───────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str = MODEL_ID,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> LLMResult:
        """Chat plano. Sin `tools` ni `response_format`: la app web no los tiene.

        Se declaran en la firma porque el protocolo los exige, y se ignoran a
        propósito. Si el cliente pidió function calling o salida estructurada,
        este backend no puede cumplirlo y hay que decirlo — no devolver texto
        suelto y dejar que el guard lo descubra.
        """
        if tools or response_format:
            raise BackendCapabilityError(
                "la app web de Gemini no soporta tools ni salida estructurada; "
                "el routing debe probar otro candidato"
            )
        prompt = "\n\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") != "system"
        )
        response = await self._generate(prompt)
        return LLMResult(text=response.text or "", model=model)

    async def search(
        self, messages: list[Message], *, model: str = MODEL_ID, max_tokens: int = 4096
    ) -> LLMResult:
        raise BackendCapabilityError(
            "la app web de Gemini no expone las fuentes del websearch de forma "
            "utilizable; el routing debe probar otro candidato"
        )

    async def image(
        self,
        prompt: str,
        *,
        model: str = MODEL_ID,
        size: str | None = None,
        quality: str | None = None,
    ) -> LLMResult:
        """Genera. `size` y `quality` se ignoran: la app web no los acepta."""
        response = await self._generate(prompt)
        return LLMResult(
            text="", model=model, images=_require_images(response, await _as_data_uris(response))
        )

    async def image_edit(
        self,
        prompt: str,
        *,
        images: list[InputImage],
        model: str = MODEL_ID,
        size: str | None = None,
        quality: str | None = None,
    ) -> LLMResult:
        """Edita desde una foto. La librería sube archivos, no bytes en memoria,
        así que las imágenes se escriben a un temporal que se borra al salir."""
        if not images:
            raise BackendCapabilityError("image_edit necesita al menos una imagen de entrada")
        async with _as_temp_files(images) as paths:
            response = await self._generate(prompt, files=paths)
        return LLMResult(
            text="", model=model, images=_require_images(response, await _as_data_uris(response))
        )

    async def embed(self, texts: list[str], *, model: str = MODEL_ID) -> EmbeddingResult:
        raise BackendCapabilityError(
            "la app web de Gemini no expone embeddings; el routing debe probar otro candidato"
        )

    @asynccontextmanager
    async def stream_chat(
        self,
        messages: list[Message],
        *,
        model: str = MODEL_ID,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ):
        raise BackendCapabilityError(
            "la app web de Gemini no hace streaming; el routing debe probar otro candidato"
        )
        yield  # pragma: no cover — inalcanzable, pero hace de esto un generador

    async def check_session(self) -> tuple[bool, str]:
        """¿Sigue viva la cookie? `(viva, detalle)`.

        **Reutiliza el cliente vivo a propósito.** La primera versión forzaba uno
        nuevo en cada chequeo —para que un cliente viejo no diera por buena una
        sesión muerta— y eso resultó ser el error: cada `init()` abre una sesión
        autenticada distinta contra Google. Medido en el caché de la librería,
        un rato de pruebas dejó **11 sesiones en hora y media** con la misma
        cuenta desde un servidor, y la cuenta terminó invalidada. La sonda que
        vigilaba la sesión la estaba matando.

        La detección se hace sin sesión nueva: `_fetch_user_status()` refresca
        `account_status` sobre el cliente que ya existe. Sólo se crea uno si no
        hay ninguno.
        """
        if self._client is not None:
            refresh = getattr(self._client, "_fetch_user_status", None)
            if refresh is not None:
                try:
                    await refresh()
                except Exception as exc:  # noqa: BLE001
                    return False, f"no se pudo comprobar el estado: {str(exc)[:200]}"
            return _judge(self._client)

        try:
            await self._ensure_client()
        except BackendCapabilityError as exc:
            # Falta la dependencia opcional: no es que la sesión esté muerta, es
            # que este backend no está instalado. Distinguirlo importa — avisar
            # "expiró la cookie" cuando falta un `pip install` manda a la persona
            # a arreglar lo que no está roto.
            return False, f"backend no disponible: {exc}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:300]
        return True, "sesión válida"

    async def aclose(self) -> None:
        if self._client is not None:
            await _quiet_close(self._client)
            self._client = None


# ── Interno ───────────────────────────────────────────────────────────────────


# Lo único que significa "la sesión sirve". Todo lo demás —UNAUTHENTICATED,
# ACCOUNT_REJECTED, TOS_PENDING…— es un motivo distinto para no poder trabajar,
# y ninguno se arregla reintentando.
_STATUS_OK = "AVAILABLE"


def _judge(client: Any) -> tuple[bool, str]:
    """Decide si la sesión sirve, mirando lo que SÍ se refresca.

    `access_token` no vale para esto y esa es la trampa: se llena en el `init()`
    inicial y **no se limpia** cuando Google invalida la sesión después. Medido:
    el monitor encadenó 30 chequeos "ok" sobre una sesión ya muerta, porque
    reutilizaba el cliente vivo y le miraba el token.

    `account_status` sí lo actualiza `_fetch_user_status()`, así que es el que
    decide. El token queda como respaldo para el caso en que no haya estado —
    una versión de la librería que no lo exponga.
    """
    status = getattr(client, "account_status", None)
    if status is not None:
        nombre = getattr(status, "name", str(status))
        if nombre != _STATUS_OK:
            return False, f"la sesión no está disponible ({nombre})"
        return True, "sesión válida (AVAILABLE)"

    if not client.access_token:
        return False, "la sesión dejó de autenticar (cookie expirada o revocada)"
    return True, "sesión válida (sin estado de cuenta; se juzga por el token)"


async def _quiet_close(client: Any) -> None:
    """Cierra sin dejar que un fallo al cerrar tape el fallo real."""
    try:
        await client.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("gemini_web.close_failed", error=str(exc)[:150])


def _require_images(response: Any, uris: list[str]) -> list[str]:
    """Cero imágenes es un FALLO, no un éxito vacío.

    La app web contesta con texto cuando decide no generar —"no puedo editar
    imágenes", un rechazo de contenido, o simplemente prosa—. Devolver eso como
    200 con `data: []` deja al cliente sin imagen y sin error que lo explique, y
    la cadena no salta al siguiente candidato. Se levanta reintentable para que
    el routing haga su trabajo.
    """
    if uris:
        return uris
    texto = (getattr(response, "text", "") or "").strip()
    detalle = f": respondió texto en vez de imagen ({texto[:120]})" if texto else ""
    raise CliproxyRetryableError(
        f"la app web de Gemini no devolvió ninguna imagen{detalle}", status_code=503
    )


@asynccontextmanager
async def _as_temp_files(images: list[InputImage]):
    """Escribe las imágenes a disco y las borra al salir.

    `gemini_webapi` sube archivos por ruta, no bytes. El directorio es temporal
    y se destruye pase lo que pase: son fotos de producto de un cliente y no
    tienen por qué quedar tiradas en /tmp.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="proxima-geminiweb-") as tmp:
        paths: list[str] = []
        for index, img in enumerate(images):
            suffix = Path(img.filename).suffix or ".png"
            path = Path(tmp) / f"input_{index}{suffix}"
            path.write_bytes(img.content)
            paths.append(str(path))
        yield paths


async def _as_data_uris(response: Any) -> list[str]:
    """Materializa las imágenes generadas como data URIs.

    Se usa el `save()` de la librería y no una descarga propia: la URL de Google
    exige la sesión autenticada, que vive dentro del objeto. Bajarla con un
    cliente HTTP limpio devuelve vacío — comprobado, y de la peor forma: un 200
    con `data: []`.

    `full_size=True` porque el objeto apunta por defecto a una versión reducida,
    y una ficha de producto necesita el original.
    """
    import base64
    import tempfile
    from pathlib import Path

    images = response.images or []
    if not images:
        return []

    out: list[str] = []
    with tempfile.TemporaryDirectory(prefix="proxima-geminiweb-out-") as tmp:
        for index, img in enumerate(images):
            try:
                try:
                    saved = await img.save(path=tmp, filename=f"out_{index}.png", full_size=True)
                except TypeError:
                    # `full_size` sólo lo acepta GeneratedImage; el resto no.
                    saved = await img.save(path=tmp, filename=f"out_{index}.png")
            except Exception as exc:  # noqa: BLE001
                log.warning("gemini_web.image_save_failed", error=str(exc)[:150])
                continue
            data = Path(saved).read_bytes()
            if data:
                out.append(f"data:{_mime_of(data)};base64,{base64.b64encode(data).decode()}")
    return out


def _mime_of(data: bytes) -> str:
    """Tipo real por los bytes, no por la extensión que le pusimos al guardar.

    Google devuelve JPEG algunas veces y PNG otras. Etiquetar mal un JPEG como
    PNG rompe a quien decodifique el data URI por la cabecera.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"
