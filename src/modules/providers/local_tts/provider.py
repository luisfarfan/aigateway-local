"""
Local TTS provider adapter.

Supports multiple TTS engines (configured via LOCAL_TTS_ENGINE env var):
  - "xtts"    — Coqui XTTS v2 (multilingual, voice cloning capable)
  - "kokoro"  — Kokoro TTS (fast, high quality)
  - "piper"   — Piper TTS (very fast, offline)

IMPORTANT — safe to import without TTS engines installed.
All engine imports are inside methods.

Config env vars:
  LOCAL_TTS_ENGINE=xtts
  XTTS_MODEL_PATH=/data/models/xtts_v2        (optional, uses HF Hub if not set)
  PIPER_MODELS_DIR=/data/models/piper
"""
import asyncio
import io
import os
from typing import Any
from uuid import UUID

import structlog

from src.core.domain import JobType, Modality
from src.core.storage import storage
from src.modules.providers.base import (
    BaseProvider,
    ExecutionContext,
    ProviderCapability,
    ProviderResult,
)

log = structlog.get_logger(__name__)

TTS_ENGINE = os.environ.get("LOCAL_TTS_ENGINE", "xtts")
XTTS_MODEL_PATH = os.environ.get("XTTS_MODEL_PATH", "tts_models/multilingual/multi-dataset/xtts_v2")
PIPER_MODELS_DIR = os.environ.get("PIPER_MODELS_DIR", "/data/models/piper")

# Available voices per engine — extend as you add voice files
XTTS_VOICES: dict[str, str] = {
    # voice_id → path to reference audio file for XTTS voice cloning
    # Add your voice .wav files here:
    # "es_male_01": "/data/voices/es_male_01.wav",
}

PIPER_VOICES: dict[str, str] = {
    # voice_id → path to piper model .onnx file
    # "en_US_amy": "/data/models/piper/en_US-amy-medium.onnx",
}


class LocalTTSProvider(BaseProvider):
    """
    Adapter: runs text-to-speech via local TTS engines.
    Engine is selected via LOCAL_TTS_ENGINE env var.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._initialized = False

    @property
    def provider_id(self) -> str:
        return "local_tts"

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id="local_tts",
            supported_job_types=[JobType.TEXT_TO_SPEECH],
            supported_models=["xtts_v2", "kokoro", "piper"],
            modality=Modality.AUDIO,
            max_concurrent_jobs=2,
            requires_gpu=False,          # TTS can run on CPU, GPU speeds it up
            estimated_vram_mb=2048,
        )

    def supports(self, job_type: JobType, model: str | None = None) -> bool:
        return job_type == JobType.TEXT_TO_SPEECH

    async def initialize(self) -> None:
        if self._initialized:
            return
        log.info("local_tts_initializing", engine=TTS_ENGINE)
        # Model loading deferred to first request to avoid startup time
        self._initialized = True

    async def execute(self, context: ExecutionContext) -> ProviderResult:
        payload = context.input_payload
        text = payload.get("text", "")
        voice = payload.get("voice", "default")
        language = payload.get("language", "en")
        output_format = payload.get("output_format", "wav")

        if not text.strip():
            return ProviderResult(success=False, error_message="Input text is empty.")

        await context.on_progress(0.0, f"Initializing {TTS_ENGINE}")

        if TTS_ENGINE == "xtts":
            return await self._run_xtts(text, voice, language, output_format, context)
        elif TTS_ENGINE == "kokoro":
            return await self._run_kokoro(text, voice, language, output_format, context)
        elif TTS_ENGINE == "piper":
            return await self._run_piper(text, voice, output_format, context)
        else:
            return ProviderResult(
                success=False,
                error_message=f"Unknown TTS engine: '{TTS_ENGINE}'. "
                              f"Set LOCAL_TTS_ENGINE to: xtts | kokoro | piper",
            )

    async def cancel(self, job_id: UUID) -> bool:
        return False  # TTS generation is too fast to cancel meaningfully

    async def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "status": "ok" if self._initialized else "not_initialized",
            "engine": TTS_ENGINE,
            "model_loaded": self._model is not None,
        }

    # ─── XTTS engine ──────────────────────────────────────────────────────────

    async def _run_xtts(
        self, text: str, voice: str, language: str, output_format: str, context: ExecutionContext
    ) -> ProviderResult:
        try:
            from TTS.api import TTS  # coqui-tts package
        except ImportError:
            return ProviderResult(
                success=False,
                error_message="Coqui TTS not installed. Run: pip install TTS",
            )

        loop = asyncio.get_event_loop()
        await context.on_progress(10.0, "Loading XTTS model")

        if self._model is None:
            self._model = await loop.run_in_executor(
                None,
                lambda: TTS(XTTS_MODEL_PATH, gpu=True),
            )

        await context.on_progress(30.0, "Synthesizing speech")

        speaker_wav = XTTS_VOICES.get(voice)
        buf = io.BytesIO()

        def _synthesize() -> bytes:
            kwargs: dict[str, Any] = {
                "text": text,
                "language": language,
                "file_path": buf,
            }
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            self._model.tts_to_file(**kwargs)
            buf.seek(0)
            return buf.read()

        audio_bytes = await loop.run_in_executor(None, _synthesize)
        await context.on_progress(90.0, "Saving audio")

        ext = output_format.lower()
        filename = f"speech.{ext}"
        mime = f"audio/{ext}"
        key = storage.output_key(str(context.job_id), filename)
        await storage.upload(key, audio_bytes, mime)
        await context.on_artifact(key, "audio", mime)
        await context.on_progress(100.0, "Done")

        return ProviderResult(
            success=True,
            result_summary={"engine": "xtts", "voice": voice, "language": language},
            artifact_keys=[key],
        )

    # ─── Kokoro engine ────────────────────────────────────────────────────────

    async def _run_kokoro(
        self, text: str, voice: str, language: str, output_format: str, context: ExecutionContext
    ) -> ProviderResult:
        try:
            import kokoro  # kokoro-onnx package
        except ImportError:
            return ProviderResult(
                success=False,
                error_message="Kokoro not installed. Run: pip install kokoro-onnx",
            )

        loop = asyncio.get_event_loop()
        await context.on_progress(20.0, "Loading Kokoro")

        def _synthesize() -> bytes:
            pipeline = kokoro.KPipeline(lang_code=language[:2])
            samples, sample_rate = pipeline(text, voice=voice)
            buf = io.BytesIO()
            import soundfile as sf
            sf.write(buf, samples, sample_rate, format=output_format.upper())
            buf.seek(0)
            return buf.read()

        await context.on_progress(30.0, "Synthesizing")
        audio_bytes = await loop.run_in_executor(None, _synthesize)

        filename = f"speech.{output_format}"
        key = storage.output_key(str(context.job_id), filename)
        mime = f"audio/{output_format}"
        await storage.upload(key, audio_bytes, mime)
        await context.on_artifact(key, "audio", mime)
        await context.on_progress(100.0, "Done")

        return ProviderResult(
            success=True,
            result_summary={"engine": "kokoro", "voice": voice},
            artifact_keys=[key],
        )

    # ─── Piper engine ─────────────────────────────────────────────────────────

    async def _run_piper(
        self, text: str, voice: str, output_format: str, context: ExecutionContext
    ) -> ProviderResult:
        try:
            from piper import PiperVoice  # piper-tts package
        except ImportError:
            return ProviderResult(
                success=False,
                error_message="Piper TTS not installed. Run: pip install piper-tts",
            )

        model_path = PIPER_VOICES.get(voice)
        if not model_path:
            return ProviderResult(
                success=False,
                error_message=(
                    f"Voice '{voice}' not found for Piper. "
                    f"Add it to PIPER_VOICES in local_tts/provider.py. "
                    f"Available: {list(PIPER_VOICES.keys())}"
                ),
            )

        loop = asyncio.get_event_loop()
        await context.on_progress(20.0, "Loading Piper model")

        def _synthesize() -> bytes:
            piper_voice = PiperVoice.load(model_path)
            buf = io.BytesIO()
            import wave
            with wave.open(buf, "wb") as wav_file:
                piper_voice.synthesize(text, wav_file)
            buf.seek(0)
            return buf.read()

        await context.on_progress(40.0, "Synthesizing")
        audio_bytes = await loop.run_in_executor(None, _synthesize)

        filename = "speech.wav"
        key = storage.output_key(str(context.job_id), filename)
        await storage.upload(key, audio_bytes, "audio/wav")
        await context.on_artifact(key, "audio", "audio/wav")
        await context.on_progress(100.0, "Done")

        return ProviderResult(
            success=True,
            result_summary={"engine": "piper", "voice": voice},
            artifact_keys=[key],
        )
