import asyncio
import io
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


class VideoAssemblerProvider(BaseProvider):
    """
    Adapter: assembles videos from images and audio clips.
    Supports hardcoded subtitles.
    """

    @property
    def provider_id(self) -> str:
        return "video_editor"

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id="video_editor",
            supported_job_types=[JobType.VIDEO_ASSEMBLY],
            supported_models=["moviepy-v1"],
            modality=Modality.VIDEO,
            max_concurrent_jobs=2,
            requires_gpu=False,
        )

    def supports(self, job_type: JobType, model: str | None = None) -> bool:
        return job_type == JobType.VIDEO_ASSEMBLY

    async def initialize(self) -> None:
        """Verify moviepy is installed."""
        try:
            import moviepy # noqa
        except ImportError:
            log.warning("moviepy_not_installed", hint="Run: pip install moviepy pysrt")

    async def execute(self, context: ExecutionContext) -> ProviderResult:
        payload = context.input_payload
        scenes = payload.get("scenes", [])
        fps = int(payload.get("fps", 24))

        if not scenes:
            return ProviderResult(success=False, error_message="No scenes provided")

        await context.on_progress(5.0, "Preparing assets")

        # Working in a temporary directory to handle local file processing needed by MoviePy
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                clips = []
                for i, scene in enumerate(scenes):
                    image_key = scene.get("image_key")
                    audio_key = scene.get("audio_key")
                    text = scene.get("text", "")

                    if not image_key or not audio_key:
                        continue

                    # Download assets
                    img_path = os.path.join(tmp_dir, f"img_{i}.png")
                    aud_path = os.path.join(tmp_dir, f"aud_{i}.wav")

                    img_data = await storage.download(image_key)
                    with open(img_path, "wb") as f:
                        f.write(img_data)

                    aud_data = await storage.download(audio_key)
                    with open(aud_path, "wb") as f:
                        f.write(aud_data)

                    # Create MoviePy clips
                    from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip

                    audio_clip = AudioFileClip(aud_path)
                    duration = audio_clip.duration

                    video_clip = ImageClip(img_path).set_duration(duration)
                    video_clip = video_clip.set_audio(audio_clip)

                    # Add subtitles (hardcoded)
                    if text:
                        txt_clip = TextClip(
                            text,
                            fontsize=payload.get("fontsize", 40),
                            color=payload.get("font_color", "white"),
                            font=payload.get("font", "Arial"),
                            bg_color="black",
                            method="caption",
                            size=(video_clip.w * 0.9, None),
                        ).set_duration(duration).set_position(("center", "bottom"))
                        
                        video_clip = CompositeVideoClip([video_clip, txt_clip])

                    clips.append(video_clip)
                    await context.on_progress(5.0 + (i + 1) / len(scenes) * 40.0, f"Processed scene {i+1}")

                if not clips:
                    return ProviderResult(success=False, error_message="Failed to build any clips")

                await context.on_progress(50.0, "Rendering final video")

                from moviepy import concatenate_videoclips
                final_video = concatenate_videoclips(clips, method="compose")

                output_path = os.path.join(tmp_dir, "output.mp4")
                
                # Run render in thread to not block event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: final_video.write_videofile(
                        output_path,
                        fps=fps,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None
                    )
                )

                # Upload result
                await context.on_progress(90.0, "Uploading result")
                artifact_key = storage.output_key(str(context.job_id), "final_video.mp4")
                with open(output_path, "rb") as f:
                    await storage.upload(artifact_key, f.read(), "video/mp4")

                await context.on_artifact(artifact_key, "video", "video/mp4")
                await context.on_progress(100.0, "Done")

                return ProviderResult(
                    success=True,
                    result_summary={"scenes": len(clips), "duration": final_video.duration},
                    artifact_keys=[artifact_key],
                )

            except Exception as e:
                log.exception("video_assembly_failed", error=str(e))
                return ProviderResult(success=False, error_message=f"Video assembly failed: {e}")

    async def cancel(self, job_id: UUID) -> bool:
        return False
