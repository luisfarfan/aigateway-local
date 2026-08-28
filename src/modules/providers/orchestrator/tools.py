from typing import Any, Type
from pydantic import BaseModel, Field, model_validator
from crewai.tools import BaseTool
import uuid

from src.core.domain import JobType
from src.modules.providers.registry import ProviderRegistry
from src.modules.providers.base import ExecutionContext

class AIGatewayTool(BaseTool):
    """Base class for tools that call AIGateway providers."""
    registry: ProviderRegistry
    worker_id: str
    on_progress: Any
    on_artifact: Any

    def _resolve_provider(self, preferred_id: str, job_type: JobType) -> Any:
        """Resolves preferred provider or falls back to stub if available."""
        if preferred_id in self.registry:
            return self.registry.resolve(preferred_id, job_type)
        
        # Fallback to stub if preferred is missing (helpful in dev/stub environments)
        if "stub" in self.registry:
            return self.registry.resolve("stub", job_type)
            
        # If no fallback possible, raise original error
        return self.registry.resolve(preferred_id, job_type)

class TextGenerationTool(AIGatewayTool):
    name: str = "text_generation"
    description: str = "Generates text or scripts using a local LLM. Input is a prompt."

    class ToolInput(BaseModel):
        prompt: str = Field(..., description="The prompt to generate text from.")
        system_prompt: str | None = Field(None, description="Optional system instruction.")

    args_schema: Type[BaseModel] = ToolInput

    def _run(self, prompt: str, system_prompt: str | None = None) -> str:
        # In CrewAI, _run is usually sync. We bridge to async.
        import asyncio
        return asyncio.run(self._arun(prompt, system_prompt))

    async def _arun(self, prompt: str, system_prompt: str | None = None) -> str:
        provider = self._resolve_provider("local_llm", JobType.TEXT_GENERATION)
        job_id = uuid.uuid4()
        
        context = ExecutionContext(
            job_id=job_id,
            job_type=JobType.TEXT_GENERATION,
            provider_id="local_llm",
            model=None,
            input_payload={"prompt": prompt, "system_prompt": system_prompt},
            priority="normal",
            timeout_seconds=300,
            worker_id=self.worker_id,
            registry=self.registry,
            on_progress=self.on_progress,
            on_artifact=self.on_artifact
        )
        
        result = await provider.execute(context)
        if result.success:
            # Result summary contains response preview or we can fetch artifact.
            # LocalLLM saves to response.txt
            from src.core.storage import storage
            data = await storage.download(result.artifact_keys[0])
            return data.decode()
        return f"Error: {result.error_message}"

class ImageGenerationTool(AIGatewayTool):
    name: str = "image_generation"
    description: str = "Generates an image from a prompt. Returns the storage_key of the generated image."

    class ToolInput(BaseModel):
        prompt: str = Field(..., description="Visual description for the image.")

    args_schema: Type[BaseModel] = ToolInput

    def _run(self, prompt: str) -> str:
        import asyncio
        return asyncio.run(self._arun(prompt))

    async def _arun(self, prompt: str) -> str:
        provider = self._resolve_provider("diffusers", JobType.IMAGE_GENERATION)
        job_id = uuid.uuid4()
        
        context = ExecutionContext(
            job_id=job_id,
            job_type=JobType.IMAGE_GENERATION,
            provider_id="diffusers",
            model=None,
            input_payload={"prompt": prompt},
            priority="normal",
            timeout_seconds=600,
            worker_id=self.worker_id,
            registry=self.registry,
            on_progress=self.on_progress,
            on_artifact=self.on_artifact
        )
        
        result = await provider.execute(context)
        if result.success and result.artifact_keys:
            return f"LLAVE_REAL: {result.artifact_keys[0]}"
        return f"Error: {result.error_message}"

class TextToSpeechTool(AIGatewayTool):
    name: str = "text_to_speech"
    description: str = "Converts text to an audio file. Returns the storage_key of the audio."

    class ToolInput(BaseModel):
        text: str = Field(..., description="Text to speak.")

    args_schema: Type[BaseModel] = ToolInput

    def _run(self, text: str) -> str:
        import asyncio
        return asyncio.run(self._arun(text))

    async def _arun(self, text: str) -> str:
        provider = self._resolve_provider("local_tts", JobType.TEXT_TO_SPEECH)
        job_id = uuid.uuid4()
        
        context = ExecutionContext(
            job_id=job_id,
            job_type=JobType.TEXT_TO_SPEECH,
            provider_id="local_tts",
            model=None,
            input_payload={"text": text},
            priority="normal",
            timeout_seconds=300,
            worker_id=self.worker_id,
            registry=self.registry,
            on_progress=self.on_progress,
            on_artifact=self.on_artifact
        )
        
        result = await provider.execute(context)
        if result.success and result.artifact_keys:
            return f"LLAVE_REAL: {result.artifact_keys[0]}"
        return f"Error: {result.error_message}"

class VideoAssemblyTool(AIGatewayTool):
    name: str = "video_assembly"
    description: str = "Stitches images and audio into a final video with subtitles. Input is a JSON list of scenes with keys: image_key, audio_key, text."

    class ToolInput(BaseModel):
        scenes: list[dict[str, Any]] = Field(..., description="List of dicts with image_key, audio_key, and text.")

        @model_validator(mode='before')
        @classmethod
        def validate_scenes_json(cls, data: Any) -> Any:
            # If data is a string (raw list from LLM), wrap it in a dict
            if isinstance(data, str):
                data = {"scenes": data}
                
            if not isinstance(data, dict):
                return data
                
            scenes = data.get("scenes")
            if isinstance(scenes, str):
                import json
                try:
                    data["scenes"] = json.loads(scenes)
                except Exception:
                    try:
                        import ast
                        data["scenes"] = ast.literal_eval(scenes)
                    except Exception:
                        pass
            
            # Self-healing: if LLM truncated keys to just "jobs/UUID", fix them
            if isinstance(data.get("scenes"), list):
                for scene in data["scenes"]:
                    for key in ["image_key", "audio_key"]:
                        val = str(scene.get(key, ""))
                        # Remove the "LLAVE_REAL: " prefix if present
                        if "LLAVE_REAL:" in val:
                            val = val.split("LLAVE_REAL:")[-1].strip()
                            scene[key] = val
                            
                        # If it only has "jobs/UUID", append the standard suffix
                        parts = val.split("/")
                        if len(parts) == 2 and parts[0] == "jobs":
                            suffix = "generated_image.png" if key == "image_key" else "generated_audio.mp3"
                            scene[key] = f"{val}/outputs/{suffix}"
            return data

    args_schema: Type[BaseModel] = ToolInput

    def _run(self, scenes: list[dict[str, Any]]) -> str:
        import asyncio
        return asyncio.run(self._arun(scenes))

    async def _arun(self, scenes: list[dict[str, Any]]) -> str:
        provider = self._resolve_provider("video_editor", JobType.VIDEO_ASSEMBLY)
        job_id = uuid.uuid4()
        
        context = ExecutionContext(
            job_id=job_id,
            job_type=JobType.VIDEO_ASSEMBLY,
            provider_id="video_editor",
            model=None,
            input_payload={"scenes": scenes},
            priority="normal",
            timeout_seconds=1200,
            worker_id=self.worker_id,
            registry=self.registry,
            on_progress=self.on_progress,
            on_artifact=self.on_artifact
        )
        
        result = await provider.execute(context)
        if result.success and result.artifact_keys:
            return result.artifact_keys[0]
        return f"Error: {result.error_message}"
