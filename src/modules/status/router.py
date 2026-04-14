"""
System status endpoint.

GET /api/v1/status — providers loaded, job counts by state, GPU/VRAM info.

Useful for a quick health overview from the Mac or from a monitoring script,
without needing to open Grafana.
"""
import subprocess
import sys
from typing import Any

import structlog
from fastapi import APIRouter, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import AsyncSessionLocal
from src.modules.jobs.models import Job

log = structlog.get_logger(__name__)
router = APIRouter(tags=["Observability"])


@router.get(
    "/status",
    summary="System and worker status",
    response_model=None,
)
async def get_status(request: Request) -> dict[str, Any]:
    """
    Returns a snapshot of the gateway state:

    - **providers**: registered adapters and their capabilities
    - **jobs**: row counts grouped by status (queued, running, completed, failed …)
    - **gpu**: VRAM usage from torch.cuda or nvidia-smi (if available)
    - **system**: Python version, OS
    """
    # ── Providers from API registry ───────────────────────────────────────────
    registry = getattr(request.app.state, "provider_registry", None)
    providers = []
    if registry:
        for cap in registry.list_capabilities():
            providers.append({
                "provider_id": cap.provider_id,
                "modality": str(cap.modality),
                "supported_job_types": [str(jt) for jt in cap.supported_job_types],
                "max_concurrent_jobs": cap.max_concurrent_jobs,
                "requires_gpu": cap.requires_gpu,
                "estimated_vram_mb": getattr(cap, "estimated_vram_mb", None),
            })

    # ── Job counts from DB ────────────────────────────────────────────────────
    job_counts: dict[str, int] = {}
    try:
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Job.status, func.count(Job.id).label("cnt"))
                .group_by(Job.status)
            )
            rows = (await session.execute(stmt)).fetchall()
            job_counts = {str(row[0].value): row[1] for row in rows}
    except Exception:
        log.exception("status_job_count_failed")

    # ── GPU / VRAM ────────────────────────────────────────────────────────────
    gpu_info = _gpu_info()

    # ── System ────────────────────────────────────────────────────────────────
    import platform
    system_info = {
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
    }

    return {
        "providers": providers,
        "jobs": job_counts,
        "gpu": gpu_info,
        "system": system_info,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gpu_info() -> dict[str, Any]:
    """
    Returns VRAM stats.

    Priority:
      1. torch.cuda — most accurate (allocated vs reserved vs total)
      2. nvidia-smi — fallback when torch isn't installed on the API process
    """
    # ── torch.cuda ────────────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(idx)
            allocated = torch.cuda.memory_allocated(idx)
            return {
                "available": True,
                "backend": "torch.cuda",
                "device_name": torch.cuda.get_device_name(idx),
                "vram_total_mb": round(total / 1024 ** 2),
                "vram_allocated_mb": round(allocated / 1024 ** 2),
                "vram_reserved_mb": round(reserved / 1024 ** 2),
                "vram_free_mb": round((total - reserved) / 1024 ** 2),
            }
    except ImportError:
        pass
    except Exception:
        log.exception("gpu_info_torch_failed")

    # ── nvidia-smi ────────────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    gpus.append({
                        "device_name": parts[0],
                        "vram_total_mb": int(parts[1]),
                        "vram_used_mb": int(parts[2]),
                        "vram_free_mb": int(parts[3]),
                    })
            if gpus:
                return {"available": True, "backend": "nvidia-smi", "gpus": gpus}
    except FileNotFoundError:
        pass   # nvidia-smi not installed — expected on Mac / CPU-only machines
    except Exception:
        log.exception("gpu_info_nvidiasmi_failed")

    return {"available": False}
