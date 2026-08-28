"""
Provider de modelos cloud (Gemini, Codex, …) para el plano asíncrono de jobs.

Es la segunda puerta a lo mismo: el plano síncrono `/v1/*` habla con
`CliproxyClient` directo, y este adaptador expone las mismas capacidades como
jobs encolados, con progreso por SSE y artefactos en MinIO. Ambos comparten
`translate.py`, así que una traducción se arregla una sola vez.

Cuándo conviene cada uno: el plano síncrono para un chat que el cliente espera;
el de jobs para lotes largos, para no bloquear al cliente, o cuando el resultado
es un archivo que hay que guardar igual.

A diferencia de los providers locales, este **no toca la GPU**: su límite es la
cuota del proveedor de arriba, no la VRAM.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from typing import Any
from uuid import UUID

import structlog

from src.core.config import get_settings
from src.core.domain import JobType, Modality
from src.core.storage import storage
from src.modules.providers.base import (
    BaseProvider,
    ExecutionContext,
    ProviderCapability,
    ProviderResult,
)
from src.modules.providers.cliproxy.client import CliproxyClient
from src.modules.providers.cliproxy.errors import (
    CliproxyError,
    CliproxyNoCredentialError,
    CliproxyRetryableError,
    CliproxyTransportError,
)
from src.modules.providers.cliproxy.translate import LLMResult

log = structlog.get_logger(__name__)

_SUPPORTED_JOB_TYPES = [JobType.TEXT_GENERATION, JobType.IMAGE_GENERATION]

# Sin GPU de por medio, el techo real es la cuota de arriba. Se deja holgado
# para no estrangular de este lado algo que el proveedor sí aguanta.
_MAX_CONCURRENT_JOBS = 16

_MIME_BY_PREFIX = {
    "data:image/png": ("png", "image/png"),
    "data:image/jpeg": ("jpg", "image/jpeg"),
    "data:image/webp": ("webp", "image/webp"),
}


class CliproxyProvider(BaseProvider):
    """Adaptador de CLIProxyAPI al puerto `BaseProvider`."""

    def __init__(self, client: CliproxyClient | None = None) -> None:
        # El cliente se puede inyectar (tests) o se construye en initialize()
        # desde settings — el worker y la API arman su propio registro.
        self._client = client
        self._owns_client = client is None
        self._cancelled: set[UUID] = set()

    # ── Declaración ───────────────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return "cliproxy"

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id="cliproxy",
            supported_job_types=list(_SUPPORTED_JOB_TYPES),
            # Vacío a propósito: el inventario es dinámico y depende de qué
            # cuentas estén conectadas. Fijarlo acá sería mentir en cuanto
            # alguien conecte o desconecte una.
            supported_models=[],
            modality=Modality.TEXT,
            max_concurrent_jobs=_MAX_CONCURRENT_JOBS,
            requires_gpu=False,
            estimated_vram_mb=None,
            metadata={"upstream": "cliproxyapi", "hosted": "cloud"},
        )

    def supports(self, job_type: JobType, model: str | None = None) -> bool:
        """Cualquier modelo: quién puede servirlo lo decide el upstream.

        Rechazar acá por nombre nos obligaría a mantener una lista que envejece
        cada vez que un proveedor renombra un modelo. Si nadie lo cubre, el
        upstream responde `auth_not_found` y eso se propaga como un error claro.
        """
        return job_type in _SUPPORTED_JOB_TYPES

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Idempotente: sólo crea el cliente si no lo inyectaron."""
        if self._client is not None:
            return
        settings = get_settings()
        self._client = CliproxyClient(
            base_url=settings.cliproxy_base_url,
            api_key=settings.cliproxy_api_key,
            timeout_seconds=settings.cliproxy_timeout_s,
            image_timeout_seconds=settings.cliproxy_image_timeout_s,
            catalog_ttl_seconds=settings.cliproxy_catalog_ttl_s,
        )
        log.info("cliproxy_provider_ready", base_url=self._client.base_url)

    async def teardown(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def cancel(self, job_id: UUID) -> bool:
        """No se puede abortar una llamada ya en vuelo río arriba.

        Se marca igual: si el job todavía no arrancó, `execute` lo ve y sale sin
        gastar cuota. Devolver False cuando ya está en vuelo es lo honesto — el
        contrato de `BaseProvider` contempla providers que no cancelan.
        """
        self._cancelled.add(job_id)
        return False

    # ── Ejecución ─────────────────────────────────────────────────────────────

    async def execute(self, context: ExecutionContext) -> ProviderResult:
        """Nunca levanta: todo fallo sale como `ProviderResult(success=False)`."""
        if self._client is None:
            await self.initialize()

        if context.job_id in self._cancelled:
            self._cancelled.discard(context.job_id)
            return ProviderResult(success=False, error_message="Job cancelado antes de empezar")

        await context.on_progress(5.0, "Resolviendo modelo")

        try:
            coro = self._dispatch(context)
            if context.timeout_seconds:
                result = await asyncio.wait_for(coro, timeout=context.timeout_seconds)
            else:
                result = await coro
        except TimeoutError:
            return self._failure(
                "timeout",
                f"El job superó los {context.timeout_seconds}s",
                retryable=True,
            )
        except CliproxyError as exc:
            return self._failure(
                type(exc).__name__,
                exc.message,
                retryable=isinstance(exc, CliproxyRetryableError | CliproxyTransportError),
                no_credential=isinstance(exc, CliproxyNoCredentialError),
            )
        except Exception as exc:  # noqa: BLE001 — el contrato prohíbe propagar
            log.exception("cliproxy_provider_unexpected", job_id=str(context.job_id))
            return self._failure("unexpected", str(exc), retryable=False)
        finally:
            self._cancelled.discard(context.job_id)

        await context.on_progress(90.0, "Guardando resultados")
        artifact_keys = await self._store_artifacts(context, result)
        await context.on_progress(100.0, "Listo")

        return ProviderResult(
            success=True,
            result_summary={
                "provider": self.provider_id,
                "model": result.model,
                "text": result.text,
                "searched": result.searched,
                "sources": [{"uri": s.uri, "title": s.title} for s in result.sources],
                "image_count": len(result.images),
            },
            artifact_keys=artifact_keys,
            execution_metadata={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            },
        )

    async def _dispatch(self, context: ExecutionContext) -> LLMResult:
        assert self._client is not None
        payload = context.input_payload
        model = context.model or get_settings().cliproxy_default_model

        if context.job_type is JobType.IMAGE_GENERATION:
            await context.on_progress(20.0, "Generando imagen")
            return await self._client.image(
                _require_prompt(payload),
                model=model,
                size=payload.get("size"),
                quality=payload.get("quality"),
            )

        messages = _messages_from(payload)
        max_tokens = int(payload.get("max_tokens") or 4096)
        if payload.get("websearch"):
            await context.on_progress(20.0, "Buscando en la web")
            return await self._client.search(messages, model=model, max_tokens=max_tokens)

        await context.on_progress(20.0, "Generando texto")
        return await self._client.chat(messages, model=model, max_tokens=max_tokens)

    # ── Artefactos ────────────────────────────────────────────────────────────

    async def _store_artifacts(self, context: ExecutionContext, result: LLMResult) -> list[str]:
        """Sube las imágenes a MinIO y las anuncia.

        El texto no se sube: vive en `result_summary`, que es donde el cliente lo
        busca. Duplicarlo como archivo sólo agregaría un objeto que nadie lee.
        """
        keys: list[str] = []
        for index, data_uri in enumerate(result.images):
            decoded = _decode_data_uri(data_uri)
            if decoded is None:
                log.warning(
                    "cliproxy_artifact_skipped",
                    job_id=str(context.job_id),
                    reason="data URI ilegible",
                )
                continue
            payload, extension, mime = decoded
            key = f"jobs/{context.job_id}/outputs/image_{index}.{extension}"
            await storage.upload(key, payload, mime)
            # on_artifact espera el objeto ya subido: consulta su tamaño.
            await context.on_artifact(key, "image", mime)
            keys.append(key)
        return keys

    # ── Salud ─────────────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Cuenta modelos anunciados. No confirma que respondan — eso lo prueba
        el watchdog, no una lista."""
        if self._client is None:
            return {"provider": self.provider_id, "status": "not_initialized"}
        try:
            models = await self._client.models()
        except CliproxyError as exc:
            return {"provider": self.provider_id, "status": "error", "error": exc.message}
        return {
            "provider": self.provider_id,
            "status": "ok",
            "base_url": self._client.base_url,
            "advertised_models": len(models),
        }

    # ── Interno ───────────────────────────────────────────────────────────────

    def _failure(
        self,
        kind: str,
        message: str,
        *,
        retryable: bool,
        no_credential: bool = False,
    ) -> ProviderResult:
        return ProviderResult(
            success=False,
            error_message=message,
            error_detail={
                "kind": kind,
                # Lo que el routing (F4) necesita para decidir: reintentar el
                # mismo modelo, cambiar de modelo, o rendirse.
                "retryable": retryable,
                "no_credential": no_credential,
            },
        )


def _messages_from(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Acepta `messages` (forma OpenAI) o `prompt` (atajo de un solo turno)."""
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt}]
    raise ValueError("El job necesita 'messages' o 'prompt' en input_payload")


def _require_prompt(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Un job de imagen necesita 'prompt' en input_payload")
    return prompt


def _decode_data_uri(data_uri: str) -> tuple[bytes, str, str] | None:
    """`data:image/png;base64,XXXX` → (bytes, extensión, mime).

    Devuelve None en vez de levantar: una imagen ilegible entre varias no
    debería tumbar un job que produjo las otras bien.
    """
    if not data_uri.startswith("data:") or "," not in data_uri:
        return None
    header, _, encoded = data_uri.partition(",")
    extension, mime = next(
        (value for prefix, value in _MIME_BY_PREFIX.items() if header.startswith(prefix)),
        ("bin", "application/octet-stream"),
    )
    try:
        return base64.b64decode(encoded, validate=True), extension, mime
    except (binascii.Error, ValueError):
        return None
