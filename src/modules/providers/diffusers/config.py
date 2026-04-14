"""
Diffusers provider configuration — model registry and pipeline specs.

Each entry describes a model that can be loaded via Diffusers:
  - which pipeline class to use
  - where the model lives (HF Hub ID or local path)
  - what job types it supports
  - what resources it needs

MODEL_PATH env vars override the default HF Hub ID with a local path.
Example: DIFFUSERS_MODEL_PATH_SDXL=/data/models/stable-diffusion-xl-base-1.0
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.config import get_settings
from src.core.domain import JobType

settings = get_settings()


@dataclass
class DiffusersModelSpec:
    """Spec for one loadable Diffusers model."""
    model_id: str                       # canonical ID used in API requests
    hf_repo_or_path: str               # HuggingFace Hub ID or absolute local path
    pipeline_class: str                 # e.g. "StableDiffusionXLPipeline"
    supported_job_types: list[JobType]
    torch_dtype: str = "float16"       # float16 (GPU) | float32 (CPU)
    use_safetensors: bool = True
    variant: str | None = "fp16"       # HF variant (fp16, bf16, None)
    vram_mb: int = 8192                # estimated VRAM requirement
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def _resolve_path(model_id: str, default_hf_repo: str) -> str:
    """
    If a local path env var is set for this model, use it.
    Otherwise fall back to the HuggingFace Hub repo ID.

    Env var convention: DIFFUSERS_MODEL_PATH_{MODEL_ID_UPPERCASE_UNDERSCORED}
    Example: DIFFUSERS_MODEL_PATH_SDXL=/data/models/sdxl
    """
    env_key = f"DIFFUSERS_MODEL_PATH_{model_id.upper().replace('-', '_')}"
    import os
    return os.environ.get(env_key, default_hf_repo)


# ─── Supported model registry ─────────────────────────────────────────────────
# Add new models here. The model_id is what clients specify in the API request.

SUPPORTED_MODELS: dict[str, DiffusersModelSpec] = {

    "stable-diffusion-xl": DiffusersModelSpec(
        model_id="stable-diffusion-xl",
        hf_repo_or_path=_resolve_path(
            "stable-diffusion-xl",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ),
        pipeline_class="StableDiffusionXLPipeline",
        supported_job_types=[JobType.IMAGE_GENERATION],
        torch_dtype="float16",
        vram_mb=8192,
    ),

    "stable-diffusion-xl-refiner": DiffusersModelSpec(
        model_id="stable-diffusion-xl-refiner",
        hf_repo_or_path=_resolve_path(
            "stable-diffusion-xl-refiner",
            "stabilityai/stable-diffusion-xl-refiner-1.0",
        ),
        pipeline_class="StableDiffusionXLImg2ImgPipeline",
        supported_job_types=[JobType.IMAGE_EDIT],
        torch_dtype="float16",
        vram_mb=8192,
    ),

    "stable-diffusion-v1-5": DiffusersModelSpec(
        model_id="stable-diffusion-v1-5",
        hf_repo_or_path=_resolve_path(
            "stable-diffusion-v1-5",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ),
        pipeline_class="StableDiffusionPipeline",
        supported_job_types=[JobType.IMAGE_GENERATION],
        torch_dtype="float16",
        vram_mb=4096,
    ),

    "stable-diffusion-inpaint": DiffusersModelSpec(
        model_id="stable-diffusion-inpaint",
        hf_repo_or_path=_resolve_path(
            "stable-diffusion-inpaint",
            "runwayml/stable-diffusion-inpainting",
        ),
        pipeline_class="StableDiffusionInpaintPipeline",
        supported_job_types=[JobType.IMAGE_EDIT],
        torch_dtype="float16",
        vram_mb=4096,
    ),

    "sdxl-turbo": DiffusersModelSpec(
        model_id="sdxl-turbo",
        hf_repo_or_path=_resolve_path(
            "sdxl-turbo",
            "stabilityai/sdxl-turbo",
        ),
        pipeline_class="AutoPipelineForText2Image",
        supported_job_types=[JobType.IMAGE_GENERATION],
        torch_dtype="float16",
        variant=None,
        vram_mb=6144,
        extra_kwargs={"guidance_scale": 0.0, "num_inference_steps": 1},
    ),

    "zeroscope-v2": DiffusersModelSpec(
        model_id="zeroscope-v2",
        hf_repo_or_path=_resolve_path(
            "zeroscope-v2",
            "cerspense/zeroscope_v2_576w",
        ),
        pipeline_class="TextToVideoSDPipeline",
        supported_job_types=[JobType.VIDEO_GENERATION],
        torch_dtype="float16",
        variant=None,
        vram_mb=16384,
    ),

    "stable-video-diffusion": DiffusersModelSpec(
        model_id="stable-video-diffusion",
        hf_repo_or_path=_resolve_path(
            "stable-video-diffusion",
            "stabilityai/stable-video-diffusion-img2vid-xt",
        ),
        pipeline_class="StableVideoDiffusionPipeline",
        supported_job_types=[JobType.VIDEO_GENERATION],
        torch_dtype="float16",
        vram_mb=20480,
    ),
}
