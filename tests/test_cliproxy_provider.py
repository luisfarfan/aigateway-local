"""
`CliproxyProvider` sobre el puerto `BaseProvider`, sin red ni MinIO.

Lo que se verifica acá es el cumplimiento del contrato de `BaseProvider`, que es
lo que el worker asume y no puede comprobar por sí mismo: que `execute` nunca
propaga una excepción, que el progreso llega a 100 en el camino feliz, y que un
fallo trae la información que el routing (F4) necesita para decidir si reintenta.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any
from uuid import uuid4

import pytest

from src.core.domain import JobType
from src.modules.providers.base import ExecutionContext
from src.modules.providers.cliproxy import provider as provider_module
from src.modules.providers.cliproxy.errors import (
    CliproxyNoCredentialError,
    CliproxyRetryableError,
)
from src.modules.providers.cliproxy.provider import CliproxyProvider
from src.modules.providers.cliproxy.translate import LLMResult, Source

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"
PNG_DATA_URI = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


class FakeClient:
    def __init__(self, result: LLMResult | None = None, raises: Exception | None = None):
        self._result = result or LLMResult(text="hola", model="gemini-3-flash")
        self._raises = raises
        self.calls: list[str] = []

    async def _answer(self, name: str) -> LLMResult:
        self.calls.append(name)
        if self._raises:
            raise self._raises
        return self._result

    async def chat(self, messages: Any, **_: Any) -> LLMResult:
        return await self._answer("chat")

    async def search(self, messages: Any, **_: Any) -> LLMResult:
        return await self._answer("search")

    async def image(self, prompt: str, **_: Any) -> LLMResult:
        return await self._answer("image")

    async def models(self, **_: Any) -> list[dict[str, Any]]:
        return [{"id": "gemini-3-flash", "owned_by": "antigravity"}]

    async def aclose(self) -> None:
        pass

    @property
    def base_url(self) -> str:
        return "http://fake"


class Recorder:
    """Captura las callbacks que el contrato obliga a llamar."""

    def __init__(self) -> None:
        self.progress: list[tuple[float, str | None]] = []
        self.artifacts: list[tuple[str, str, str]] = []

    async def on_progress(self, percent: float, step: str | None = None) -> None:
        self.progress.append((percent, step))

    async def on_artifact(self, key: str, artifact_type: str, mime: str) -> None:
        self.artifacts.append((key, artifact_type, mime))


def make_context(
    recorder: Recorder,
    *,
    job_type: JobType = JobType.TEXT_GENERATION,
    payload: dict[str, Any] | None = None,
    job_id: Any = None,
    timeout: int | None = None,
) -> ExecutionContext:
    return ExecutionContext(
        job_id=job_id or uuid4(),
        job_type=job_type,
        provider_id="cliproxy",
        model="gemini-3-flash",
        input_payload=payload if payload is not None else {"prompt": "hola"},
        priority="normal",
        timeout_seconds=timeout,
        worker_id="test",
        on_progress=recorder.on_progress,
        on_artifact=recorder.on_artifact,
    )


@pytest.fixture
def no_storage(monkeypatch):
    """MinIO fuera: se registra qué se habría subido."""
    uploaded: list[tuple[str, bytes, str]] = []

    class FakeStorage:
        async def upload(self, key: str, data: bytes, content_type: str) -> None:
            uploaded.append((key, data, content_type))

    monkeypatch.setattr(provider_module, "storage", FakeStorage())
    return uploaded


# ─── Declaración ──────────────────────────────────────────────────────────────


def test_solo_declara_lo_que_puede_hacer():
    p = CliproxyProvider(client=FakeClient())
    assert p.supports(JobType.TEXT_GENERATION)
    assert p.supports(JobType.IMAGE_GENERATION)
    assert not p.supports(JobType.TEXT_TO_SPEECH)
    assert not p.supports(JobType.VIDEO_ASSEMBLY)


def test_no_reclama_gpu():
    """El scheduler reparte slots para proteger la VRAM. Este provider llama a
    una API remota: pedir un slot de GPU serializaría trabajo que no compite
    por nada."""
    capability = CliproxyProvider(client=FakeClient()).capability
    assert capability.requires_gpu is False
    assert capability.estimated_vram_mb is None
    assert capability.max_concurrent_jobs > 1


def test_no_fija_una_lista_de_modelos():
    """El inventario depende de qué cuentas estén conectadas en cada momento.
    Una lista fija mentiría en cuanto alguien conecte o desconecte una."""
    assert CliproxyProvider(client=FakeClient()).capability.supported_models == []


# ─── Camino feliz ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_texto_reporta_progreso_hasta_cien(no_storage):
    recorder = Recorder()
    result = await CliproxyProvider(client=FakeClient()).execute(make_context(recorder))

    assert result.success
    assert result.result_summary["text"] == "hola"
    # El contrato exige llegar a 100 antes de devolver.
    assert recorder.progress[-1][0] == 100.0
    assert [p for p, _ in recorder.progress] == sorted(p for p, _ in recorder.progress)


@pytest.mark.asyncio
async def test_websearch_enruta_a_search_y_conserva_las_fuentes(no_storage):
    fake = FakeClient(
        LLMResult(
            text="Argentina",
            model="gemini-3-flash",
            sources=[Source(uri="https://fifa.com", title="fifa.com")],
            searched=True,
        )
    )
    recorder = Recorder()
    result = await CliproxyProvider(client=fake).execute(
        make_context(recorder, payload={"prompt": "quién ganó", "websearch": True})
    )

    assert fake.calls == ["search"]
    assert result.result_summary["searched"] is True
    assert result.result_summary["sources"] == [{"uri": "https://fifa.com", "title": "fifa.com"}]


@pytest.mark.asyncio
async def test_acepta_messages_o_prompt(no_storage):
    fake = FakeClient()
    recorder = Recorder()
    result = await CliproxyProvider(client=fake).execute(
        make_context(recorder, payload={"messages": [{"role": "user", "content": "hola"}]})
    )
    assert result.success
    assert fake.calls == ["chat"]


@pytest.mark.asyncio
async def test_los_tokens_van_a_execution_metadata(no_storage):
    """De ahí los toma el costo en F3."""
    fake = FakeClient(LLMResult(text="x", model="m", prompt_tokens=7, completion_tokens=3))
    result = await CliproxyProvider(client=fake).execute(make_context(Recorder()))
    assert result.execution_metadata["total_tokens"] == 10


# ─── Artefactos ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_la_imagen_se_sube_y_se_anuncia(no_storage):
    fake = FakeClient(LLMResult(text="", model="gemini-3.1-flash-image", images=[PNG_DATA_URI]))
    recorder = Recorder()
    job_id = uuid4()

    result = await CliproxyProvider(client=fake).execute(
        make_context(
            recorder,
            job_type=JobType.IMAGE_GENERATION,
            payload={"prompt": "un cubo"},
            job_id=job_id,
        )
    )

    key = f"jobs/{job_id}/outputs/image_0.png"
    # Subida primero: on_artifact consulta el tamaño del objeto ya guardado.
    assert no_storage == [(key, PNG_BYTES, "image/png")]
    assert recorder.artifacts == [(key, "image", "image/png")]
    assert result.artifact_keys == [key]


@pytest.mark.asyncio
async def test_una_imagen_ilegible_no_tumba_las_demas(no_storage):
    """Descartar una y entregar el resto es mejor que perder el job entero."""
    fake = FakeClient(LLMResult(text="", model="m", images=["no-es-un-data-uri", PNG_DATA_URI]))
    recorder = Recorder()
    result = await CliproxyProvider(client=fake).execute(
        make_context(recorder, job_type=JobType.IMAGE_GENERATION, payload={"prompt": "x"})
    )

    assert result.success
    assert len(result.artifact_keys) == 1
    assert result.artifact_keys[0].endswith("image_1.png")


# ─── Fallos ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "retryable", "no_credential"),
    [
        (CliproxyRetryableError("429"), True, False),
        (CliproxyNoCredentialError("auth_not_found"), False, True),
    ],
)
async def test_los_errores_salen_como_resultado_no_como_excepcion(
    error: Exception, retryable: bool, no_credential: bool
):
    """El worker asume que `execute` no propaga. Y el detalle importa: el
    routing decide distinto ante 'esperá y reintentá' que ante 'ninguna
    credencial cubre este modelo'."""
    result = await CliproxyProvider(client=FakeClient(raises=error)).execute(
        make_context(Recorder())
    )

    assert result.success is False
    assert result.error_detail["retryable"] is retryable
    assert result.error_detail["no_credential"] is no_credential


@pytest.mark.asyncio
async def test_un_payload_invalido_no_propaga():
    """Falta `prompt` y `messages`. Sale como fallo, no como ValueError suelto."""
    result = await CliproxyProvider(client=FakeClient()).execute(
        make_context(Recorder(), payload={})
    )
    assert result.success is False
    assert "messages" in (result.error_message or "")


@pytest.mark.asyncio
async def test_el_timeout_se_respeta_y_es_reintentable():
    class SlowClient(FakeClient):
        async def chat(self, messages: Any, **_: Any) -> LLMResult:
            await asyncio.sleep(5)
            return LLMResult(text="tarde", model="m")

    result = await CliproxyProvider(client=SlowClient()).execute(
        make_context(Recorder(), timeout=1)
    )
    assert result.success is False
    assert result.error_detail["retryable"] is True


@pytest.mark.asyncio
async def test_cancelar_antes_de_empezar_no_gasta_cuota():
    """No se puede abortar una llamada en vuelo, pero sí evitar la que no salió."""
    fake = FakeClient()
    p = CliproxyProvider(client=fake)
    job_id = uuid4()

    assert await p.cancel(job_id) is False  # honesto: no cancela en vuelo
    result = await p.execute(make_context(Recorder(), job_id=job_id))

    assert result.success is False
    assert fake.calls == []
