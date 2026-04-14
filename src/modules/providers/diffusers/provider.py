"""
Diffusers provider adapter.

Wraps HuggingFace Diffusers pipelines as a LocalAIGateway provider.
Supports: image_generation, image_edit, video_generation.

IMPORTANT — this file is safe to import on machines WITHOUT Diffusers installed.
All diffusers/torch imports are inside methods and guarded with clear error messages.
When cloned to the Ubuntu machine (where diffusers is installed), everything works.

Model loading strategy:
  - Models are loaded lazily on first request (avoids loading all models at startup).
  - A loaded pipeline is cached in self._pipelines — subsequent requests reuse it.
  - Loading is protected by an asyncio.Lock per model to prevent concurrent double-loads.

Progress reporting:
  - Diffusers callbacks are synchronous but our on_progress is async.
  - We bridge this by scheduling async callbacks on the event loop from the sync thread.
  - Generation runs in run_in_executor (thread pool) to avoid blocking the event loop.
"""
import asyncio
import io
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
from src.modules.providers.diffusers.config import SUPPORTED_MODELS, DiffusersModelSpec

log = structlog.get_logger(__name__)


class DiffusersProvider(BaseProvider):
    """
    Adapter: wraps Diffusers pipelines as a gateway provider.
    One instance handles all supported models (loaded lazily per model).
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}          # model_id → loaded pipeline
        self._locks: dict[str, asyncio.Lock] = {}      # model_id → loading lock
        self._device: str = "cpu"                      # set in initialize()
        self._initialized = False

    # ─── Port implementation ──────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return "diffusers"

    @property
    def capability(self) -> ProviderCapability:
        all_supported_types: set[JobType] = set()
        for spec in SUPPORTED_MODELS.values():
            all_supported_types.update(spec.supported_job_types)

        return ProviderCapability(
            provider_id="diffusers",
            supported_job_types=list(all_supported_types),
            supported_models=list(SUPPORTED_MODELS.keys()),
            modality=Modality.IMAGE,
            max_concurrent_jobs=1,      # GPU can only handle one at a time
            requires_gpu=True,
            estimated_vram_mb=8192,     # conservative default
        )

    def supports(self, job_type: JobType, model: str | None = None) -> bool:
        if model and model not in SUPPORTED_MODELS:
            return False
        spec = SUPPORTED_MODELS.get(model) if model else None
        if spec:
            return job_type in spec.supported_job_types
        # No model specified — check if any model supports this job type
        return any(job_type in s.supported_job_types for s in SUPPORTED_MODELS.values())

    async def initialize(self) -> None:
        """Detect device (CUDA/MPS/CPU). Does NOT load models yet — lazy loading."""
        if self._initialized:
            return
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
                log.warning("diffusers_no_gpu", message="Running on CPU — expect slow inference")
        except ImportError:
            raise RuntimeError(
                "PyTorch is not installed. Install it on the Ubuntu machine: "
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )
        log.info("diffusers_initialized", device=self._device)
        self._initialized = True

    async def execute(self, context: ExecutionContext) -> ProviderResult:
        model_id = context.model or "stable-diffusion-xl"
        spec = SUPPORTED_MODELS.get(model_id)
        if spec is None:
            return ProviderResult(
                success=False,
                error_message=f"Unknown model '{model_id}'. Available: {list(SUPPORTED_MODELS)}",
            )

        try:
            pipeline = await self._get_pipeline(spec)
        except Exception as e:
            return ProviderResult(success=False, error_message=f"Failed to load model: {e}")

        job_type = context.job_type
        if job_type == JobType.IMAGE_GENERATION:
            return await self._run_txt2img(pipeline, spec, context)
        elif job_type == JobType.IMAGE_EDIT:
            return await self._run_img2img(pipeline, spec, context)
        elif job_type == JobType.VIDEO_GENERATION:
            return await self._run_txt2video(pipeline, spec, context)
        else:
            return ProviderResult(
                success=False,
                error_message=f"DiffusersProvider does not handle job_type '{job_type}'",
            )

    async def cancel(self, job_id: UUID) -> bool:
        # Diffusers pipelines don't support mid-inference cancellation.
        # The job will complete, but its result will be discarded.
        log.warning("diffusers_cancel_not_supported", job_id=str(job_id))
        return False

    async def teardown(self) -> None:
        """Unload all pipelines from GPU memory."""
        try:
            import torch
            for model_id, pipe in self._pipelines.items():
                del pipe
                log.info("diffusers_model_unloaded", model_id=model_id)
            self._pipelines.clear()
            if self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def health_check(self) -> dict[str, Any]:
        try:
            import torch
            vram_free = None
            if torch.cuda.is_available():
                vram_free = torch.cuda.mem_get_info()[0] // (1024 ** 2)
        except ImportError:
            vram_free = None

        return {
            "provider": self.provider_id,
            "status": "ok" if self._initialized else "not_initialized",
            "device": self._device,
            "loaded_models": list(self._pipelines.keys()),
            "vram_free_mb": vram_free,
        }

    # ─── Internal: model loading ──────────────────────────────────────────────

    async def _get_pipeline(self, spec: DiffusersModelSpec) -> Any:
        """Returns the loaded pipeline for the spec, loading it if needed."""
        model_id = spec.model_id

        if model_id in self._pipelines:
            return self._pipelines[model_id]

        # One lock per model prevents duplicate concurrent loads
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()

        async with self._locks[model_id]:
            # Double-check after acquiring lock
            if model_id in self._pipelines:
                return self._pipelines[model_id]

            log.info("diffusers_loading_model", model_id=model_id, path=spec.hf_repo_or_path)
            pipeline = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._load_pipeline_sync(spec)
            )
            self._pipelines[model_id] = pipeline
            log.info("diffusers_model_loaded", model_id=model_id, device=self._device)
            return pipeline

    def _load_pipeline_sync(self, spec: DiffusersModelSpec) -> Any:
        """Synchronous model loading — runs in thread pool executor."""
        try:
            import torch
            import diffusers
        except ImportError as e:
            raise RuntimeError(
                f"Diffusers not installed: {e}. "
                "Run: pip install diffusers transformers accelerate torch"
            ) from e

        pipeline_class = getattr(diffusers, spec.pipeline_class, None)
        if pipeline_class is None:
            raise ValueError(f"Unknown pipeline class: {spec.pipeline_class}")

        torch_dtype = getattr(torch, spec.torch_dtype, torch.float16)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "use_safetensors": spec.use_safetensors,
        }
        if spec.variant:
            load_kwargs["variant"] = spec.variant

        pipeline = pipeline_class.from_pretrained(
            spec.hf_repo_or_path,
            **load_kwargs,
        )

        # Memory optimizations
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()

        pipeline = pipeline.to(self._device)
        return pipeline

    # ─── Internal: inference runs ─────────────────────────────────────────────

    async def _run_txt2img(
        self,
        pipeline: Any,
        spec: DiffusersModelSpec,
        context: ExecutionContext,
    ) -> ProviderResult:
        payload = context.input_payload
        steps = int(payload.get("steps", 30))
        loop = asyncio.get_event_loop()

        # Progress bridge: sync callback → async on_progress
        async def report(step: int, total: int) -> None:
            pct = round((step / total) * 100, 1)
            await context.on_progress(pct, f"Denoising step {step}/{total}")

        def step_callback(pipe: Any, step: int, timestep: int, kwargs: dict) -> dict:
            asyncio.run_coroutine_threadsafe(report(step, steps), loop)
            return kwargs

        await context.on_progress(0.0, "Starting inference")

        def _generate() -> Any:
            kwargs: dict[str, Any] = {
                "prompt": payload["prompt"],
                "num_inference_steps": steps,
                "guidance_scale": float(payload.get("guidance_scale", 7.5)),
                "width": int(payload.get("width", 1024)),
                "height": int(payload.get("height", 1024)),
                "num_images_per_prompt": int(payload.get("num_images", 1)),
                "callback_on_step_end": step_callback,
                "callback_on_step_end_tensor_inputs": ["latents"],
            }
            if payload.get("negative_prompt"):
                kwargs["negative_prompt"] = payload["negative_prompt"]
            if payload.get("seed") is not None:
                import torch
                kwargs["generator"] = torch.Generator(device=self._device).manual_seed(
                    int(payload["seed"])
                )
            return pipeline(**kwargs)

        output = await loop.run_in_executor(None, _generate)
        return await self._save_images(output.images, context)

    async def _run_img2img(
        self,
        pipeline: Any,
        spec: DiffusersModelSpec,
        context: ExecutionContext,
    ) -> ProviderResult:
        from PIL import Image
        payload = context.input_payload
        steps = int(payload.get("steps", 30))
        loop = asyncio.get_event_loop()

        # Download source image from MinIO
        image_bytes = await storage.download(payload["image_key"])
        source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        mask_image = None
        if payload.get("mask_key"):
            mask_bytes = await storage.download(payload["mask_key"])
            mask_image = Image.open(io.BytesIO(mask_bytes)).convert("RGB")

        async def report(step: int, total: int) -> None:
            pct = round((step / total) * 100, 1)
            await context.on_progress(pct, f"Denoising step {step}/{total}")

        def step_callback(pipe: Any, step: int, timestep: int, kwargs: dict) -> dict:
            asyncio.run_coroutine_threadsafe(report(step, steps), loop)
            return kwargs

        await context.on_progress(0.0, "Starting img2img inference")

        def _generate() -> Any:
            kwargs: dict[str, Any] = {
                "prompt": payload["prompt"],
                "image": source_image,
                "strength": float(payload.get("strength", 0.8)),
                "num_inference_steps": steps,
                "guidance_scale": float(payload.get("guidance_scale", 7.5)),
                "callback_on_step_end": step_callback,
                "callback_on_step_end_tensor_inputs": ["latents"],
            }
            if mask_image:
                kwargs["mask_image"] = mask_image
            if payload.get("negative_prompt"):
                kwargs["negative_prompt"] = payload["negative_prompt"]
            if payload.get("seed") is not None:
                import torch
                kwargs["generator"] = torch.Generator(device=self._device).manual_seed(
                    int(payload["seed"])
                )
            return pipeline(**kwargs)

        output = await loop.run_in_executor(None, _generate)
        return await self._save_images(output.images, context)

    async def _run_txt2video(
        self,
        pipeline: Any,
        spec: DiffusersModelSpec,
        context: ExecutionContext,
    ) -> ProviderResult:
        payload = context.input_payload
        loop = asyncio.get_event_loop()

        await context.on_progress(0.0, "Starting video generation")

        def _generate() -> Any:
            kwargs: dict[str, Any] = {
                "prompt": payload["prompt"],
                "num_frames": int(payload.get("num_frames", 16)),
                "width": int(payload.get("width", 576)),
                "height": int(payload.get("height", 320)),
            }
            if payload.get("negative_prompt"):
                kwargs["negative_prompt"] = payload["negative_prompt"]
            return pipeline(**kwargs)

        await context.on_progress(10.0, "Running pipeline")
        output = await loop.run_in_executor(None, _generate)
        await context.on_progress(80.0, "Exporting video frames")

        return await self._save_video(output.frames, context, fps=int(payload.get("fps", 8)))

    # ─── Internal: artifact saving ────────────────────────────────────────────

    async def _save_images(self, images: list[Any], context: ExecutionContext) -> ProviderResult:
        import torch
        artifact_keys: list[str] = []

        for i, image in enumerate(images):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)

            filename = f"image_{i:02d}.png"
            key = storage.output_key(str(context.job_id), filename)
            await storage.upload(key, buf.read(), "image/png")
            artifact_keys.append(key)
            await context.on_artifact(key, "image", "image/png")

        await context.on_progress(100.0, "Done")

        vram_used = None
        try:
            import torch
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() // (1024 ** 2)
        except Exception:
            pass

        return ProviderResult(
            success=True,
            result_summary={
                "images_generated": len(images),
                "model": context.model,
            },
            artifact_keys=artifact_keys,
            execution_metadata={"vram_allocated_mb": vram_used, "device": self._device},
        )

    async def _save_video(
        self, frames: Any, context: ExecutionContext, fps: int
    ) -> ProviderResult:
        try:
            import imageio
            import numpy as np
        except ImportError:
            return ProviderResult(
                success=False,
                error_message="imageio not installed. Run: pip install imageio[ffmpeg]",
            )

        buf = io.BytesIO()
        frame_array = [np.array(f) for f in frames[0]]
        imageio.mimsave(buf, frame_array, format="mp4", fps=fps)
        buf.seek(0)

        filename = "output.mp4"
        key = storage.output_key(str(context.job_id), filename)
        await storage.upload(key, buf.read(), "video/mp4")
        await context.on_artifact(key, "video", "video/mp4")
        await context.on_progress(100.0, "Done")

        return ProviderResult(
            success=True,
            result_summary={"frames": len(frame_array), "fps": fps, "model": context.model},
            artifact_keys=[key],
        )
