import asyncio
from uuid import UUID
import structlog
import re
import os
from typing import Any

from src.core.domain import JobType
from src.modules.providers.base import BaseProvider, ExecutionContext, ProviderCapability, ProviderResult, Modality
from src.modules.providers.orchestrator.tools import (
    TextGenerationTool, ImageGenerationTool, TextToSpeechTool, VideoAssemblyTool
)

log = structlog.get_logger(__name__)

class CrewAIOrchestratorProvider(BaseProvider):
    @property
    def provider_id(self) -> str:
        return "orchestrator"

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id=self.provider_id,
            supported_job_types=[JobType.AUTONOMOUS_MISSION],
            supported_models=["crewai-managed"],
            modality=Modality.VIDEO,
            max_concurrent_jobs=2
        )

    def supports(self, job_type: JobType, model: str | None = None) -> bool:
        return job_type == JobType.AUTONOMOUS_MISSION

    async def initialize(self) -> None:
        pass

    async def execute(self, context: ExecutionContext) -> ProviderResult:
        try:
            topic = context.input_payload.get("prompt", "un tema interesante")

            # Setup Tools
            async def silent_progress(p, m): pass
            async def silent_artifact(k, t, m): pass

            text_tool = TextGenerationTool(
                registry=context.registry, worker_id=context.worker_id,
                on_progress=silent_progress, on_artifact=silent_artifact
            )
            tts_tool = TextToSpeechTool(
                registry=context.registry, worker_id=context.worker_id,
                on_progress=silent_progress, on_artifact=silent_artifact
            )

            # --- STEP 1: GENERATE SCRIPT (Direct Call) ---
            await context.on_progress(20.0, "Generando guion informativo...")
            script = await text_tool._arun(f"Escribe un guion breve y profesional sobre: {topic}")
            log.info("script_generated", length=len(script))

            # --- STEP 2: GENERATE AUDIO (Direct Call) ---
            await context.on_progress(60.0, "Sintetizando audio...")
            audio_res = await tts_tool._arun(script)
            log.info("audio_generated", result=audio_res)

            # Final extraction of the storage key
            audio_key = None
            match = re.search(r'(jobs/[a-f0-9-]+/outputs/[\w.-]+)', audio_res)
            if match:
                audio_key = match.group(1)
            else:
                match_v2 = re.search(r'(jobs/[a-f0-9-]+)', audio_res)
                if match_v2:
                    audio_key = f"{match_v2.group(1)}/outputs/speech.wav"

            if audio_key:
                await context.on_artifact(audio_key, "audio", "audio/mpeg")
                await context.on_progress(100.0, "Completado")
                return ProviderResult(
                    success=True,
                    artifact_keys=[audio_key],
                    result_summary={"audio_key": audio_key, "script": script[:100] + "..."}
                )
            else:
                return ProviderResult(
                    success=False,
                    error_message=f"Failed to extract audio key from: {audio_res}"
                )

        except Exception as e:
            log.exception("orchestrator_failed", error=str(e))
            return ProviderResult(success=False, error_message=f"Orchestration failed: {e}")

    async def cancel(self, job_id: UUID) -> bool:
        return True
