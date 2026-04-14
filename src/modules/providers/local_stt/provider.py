"""
Local STT (Speech-to-Text) provider adapter.

Primary engine: faster-whisper (CTranslate2 backend — 4x faster than openai-whisper,
same accuracy, runs on CPU and GPU).
Fallback engine: openai-whisper (if faster-whisper not installed).

Supported models (same names for both engines):
  tiny, base, small, medium, large-v2, large-v3, large-v3-turbo

Model path config:
  STT_MODEL_SIZE=large-v3          (which Whisper model to load)
  STT_DEVICE=cuda                  (cuda | cpu | auto)
  STT_COMPUTE_TYPE=float16         (float16 | int8 | float32 — int8 for CPU)
  STT_MODEL_PATH=/data/models/whisper  (optional local path, downloads from HF if not set)

IMPORTANT — safe to import without faster-whisper installed.
All engine imports are inside methods.
"""
import io
import json
import os
import tempfile
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

STT_MODEL_SIZE = os.environ.get("STT_MODEL_SIZE", "base")
STT_DEVICE = os.environ.get("STT_DEVICE", "auto")
STT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "float16")
STT_MODEL_PATH = os.environ.get("STT_MODEL_PATH", None)  # None = download from HF Hub

SUPPORTED_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v2", "large-v3", "large-v3-turbo",
]


class LocalSTTProvider(BaseProvider):
    """
    Adapter: transcribes audio files using local Whisper models.
    Downloads the model on first use if not already cached.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._engine: str | None = None   # "faster_whisper" | "openai_whisper"
        self._loaded_model_size: str | None = None

    @property
    def provider_id(self) -> str:
        return "local_stt"

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id="local_stt",
            supported_job_types=[JobType.SPEECH_TO_TEXT],
            supported_models=SUPPORTED_MODELS,
            modality=Modality.AUDIO,
            max_concurrent_jobs=2,
            requires_gpu=False,         # runs on CPU (slower) or GPU (fast)
            estimated_vram_mb=2048,     # varies: tiny=~390MB, large-v3=~3GB
        )

    def supports(self, job_type: JobType, model: str | None = None) -> bool:
        if job_type != JobType.SPEECH_TO_TEXT:
            return False
        if model and model not in SUPPORTED_MODELS:
            return False
        return True

    async def initialize(self) -> None:
        """Detect available engine. Model loading is deferred to first request."""
        try:
            import faster_whisper  # noqa: F401
            self._engine = "faster_whisper"
            log.info("stt_engine_detected", engine="faster_whisper")
        except ImportError:
            try:
                import whisper  # noqa: F401
                self._engine = "openai_whisper"
                log.info("stt_engine_detected", engine="openai_whisper")
            except ImportError:
                log.warning(
                    "stt_no_engine",
                    hint="Install with: pip install faster-whisper  OR  pip install openai-whisper",
                )

    async def execute(self, context: ExecutionContext) -> ProviderResult:
        if self._engine is None:
            return ProviderResult(
                success=False,
                error_message=(
                    "No STT engine installed. "
                    "Run: pip install faster-whisper  (recommended)"
                ),
            )

        payload = context.input_payload
        audio_key: str | None = payload.get("audio_key")
        if not audio_key:
            return ProviderResult(
                success=False,
                error_message="Missing required field 'audio_key' in input payload.",
            )

        model_size = context.model or STT_MODEL_SIZE
        language = payload.get("language")          # None = auto-detect
        task = payload.get("task", "transcribe")    # transcribe | translate
        word_timestamps = bool(payload.get("word_timestamps", False))

        await context.on_progress(0.0, "Downloading audio from storage")

        # Download audio from MinIO to a temp file
        try:
            audio_bytes = await storage.download(audio_key)
        except Exception as e:
            return ProviderResult(success=False, error_message=f"Failed to download audio: {e}")

        await context.on_progress(10.0, f"Loading Whisper {model_size}")

        # Load model (cached after first call)
        try:
            model = await self._get_model(model_size)
        except Exception as e:
            return ProviderResult(success=False, error_message=f"Model load failed: {e}")

        await context.on_progress(25.0, "Transcribing")

        # Write audio to temp file (Whisper needs a file path)
        suffix = _infer_suffix(audio_key)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            if self._engine == "faster_whisper":
                result_data = await self._transcribe_faster(
                    model, tmp_path, language, task, word_timestamps, context
                )
            else:
                result_data = await self._transcribe_openai(
                    model, tmp_path, language, task, word_timestamps, context
                )
        finally:
            os.unlink(tmp_path)

        if not result_data.get("success"):
            return ProviderResult(
                success=False,
                error_message=result_data.get("error", "Transcription failed"),
            )

        await context.on_progress(90.0, "Saving transcription")

        # Save full transcription JSON as artifact
        transcript_key = storage.output_key(str(context.job_id), "transcription.json")
        await storage.upload(
            transcript_key,
            json.dumps(result_data, ensure_ascii=False, indent=2).encode(),
            "application/json",
        )
        await context.on_artifact(transcript_key, "json", "application/json")

        # Also save plain text transcript
        text_key = storage.output_key(str(context.job_id), "transcription.txt")
        await storage.upload(
            text_key,
            result_data["text"].encode(),
            "text/plain",
        )
        await context.on_artifact(text_key, "text", "text/plain")

        await context.on_progress(100.0, "Done")

        return ProviderResult(
            success=True,
            result_summary={
                "text": result_data["text"],
                "language": result_data.get("language"),
                "duration_s": result_data.get("duration"),
                "segments": len(result_data.get("segments", [])),
                "model": model_size,
                "engine": self._engine,
            },
            artifact_keys=[transcript_key, text_key],
        )

    async def cancel(self, job_id: UUID) -> bool:
        return False  # Whisper inference can't be interrupted mid-run

    async def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "engine": self._engine or "not_detected",
            "loaded_model": self._loaded_model_size,
            "device": STT_DEVICE,
            "compute_type": STT_COMPUTE_TYPE,
        }

    # ─── Model loading ────────────────────────────────────────────────────────

    async def _get_model(self, model_size: str) -> Any:
        """Load model if not already cached. Thread-safe via asyncio executor."""
        if self._model is not None and self._loaded_model_size == model_size:
            return self._model

        import asyncio
        loop = asyncio.get_event_loop()

        if self._engine == "faster_whisper":
            model = await loop.run_in_executor(None, lambda: self._load_faster(model_size))
        else:
            model = await loop.run_in_executor(None, lambda: self._load_openai(model_size))

        self._model = model
        self._loaded_model_size = model_size
        log.info("stt_model_loaded", model=model_size, engine=self._engine)
        return model

    def _load_faster(self, model_size: str) -> Any:
        from faster_whisper import WhisperModel

        device = STT_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = STT_COMPUTE_TYPE
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"  # float16 not supported on CPU

        model_path = STT_MODEL_PATH or model_size
        return WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )

    def _load_openai(self, model_size: str) -> Any:
        import whisper
        model_path = STT_MODEL_PATH or model_size
        return whisper.load_model(model_path)

    # ─── Transcription ────────────────────────────────────────────────────────

    async def _transcribe_faster(
        self, model: Any, audio_path: str,
        language: str | None, task: str,
        word_timestamps: bool, context: ExecutionContext,
    ) -> dict:
        import asyncio
        loop = asyncio.get_event_loop()

        def _run():
            segments_gen, info = model.transcribe(
                audio_path,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                vad_filter=True,            # remove silence
                vad_parameters={"min_silence_duration_ms": 500},
            )
            segments = []
            full_text_parts = []
            for seg in segments_gen:
                segments.append({
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    **({"words": [
                        {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                        for w in seg.words
                    ]} if word_timestamps and seg.words else {}),
                })
                full_text_parts.append(seg.text.strip())

            return {
                "success": True,
                "text": " ".join(full_text_parts),
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "segments": segments,
            }

        result = await loop.run_in_executor(None, _run)

        # Report progress mid-transcription (we only know after it finishes)
        await context.on_progress(80.0, f"Transcribed {result.get('duration', 0):.1f}s of audio")
        return result

    async def _transcribe_openai(
        self, model: Any, audio_path: str,
        language: str | None, task: str,
        word_timestamps: bool, context: ExecutionContext,
    ) -> dict:
        import asyncio
        loop = asyncio.get_event_loop()

        def _run():
            kwargs: dict = {"task": task, "verbose": False}
            if language:
                kwargs["language"] = language
            if word_timestamps:
                kwargs["word_timestamps"] = True

            result = model.transcribe(audio_path, **kwargs)
            segments = [
                {
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ]
            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language"),
                "duration": segments[-1]["end"] if segments else 0,
                "segments": segments,
            }

        result = await loop.run_in_executor(None, _run)
        await context.on_progress(80.0, "Transcription complete")
        return result


def _infer_suffix(audio_key: str) -> str:
    """Extract file extension from MinIO key for temp file naming."""
    parts = audio_key.rsplit(".", 1)
    return f".{parts[-1]}" if len(parts) > 1 else ".wav"
