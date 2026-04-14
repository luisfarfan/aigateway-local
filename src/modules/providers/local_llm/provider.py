"""
Local LLM provider adapter.

Supports two backends (configured via LOCAL_LLM_BACKEND env var):
  - "ollama"       — calls Ollama's HTTP API (streaming)
  - "transformers" — loads models directly via HuggingFace Transformers

Both backends stream token-by-token and report progress via SSE.

IMPORTANT — safe to import without Ollama/Transformers installed.
All backend imports are inside methods. Will fail gracefully at runtime
with a clear error message if the backend isn't available.
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
from src.modules.providers.local_llm.config import (
    OLLAMA_BASE_URL,
    SUPPORTED_MODELS,
    LLMModelSpec,
)

log = structlog.get_logger(__name__)


class LocalLLMProvider(BaseProvider):
    """
    Adapter: runs text generation against local LLMs.
    Routes to Ollama or HuggingFace Transformers based on model spec.
    """

    def __init__(self) -> None:
        self._transformers_models: dict[str, Any] = {}  # hf models loaded in memory
        self._transformers_locks: dict[str, asyncio.Lock] = {}

    @property
    def provider_id(self) -> str:
        return "local_llm"

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id="local_llm",
            supported_job_types=[JobType.TEXT_GENERATION, JobType.TEXT_EMBEDDING],
            supported_models=list(SUPPORTED_MODELS.keys()),
            modality=Modality.TEXT,
            max_concurrent_jobs=2,   # text is less VRAM-intensive
            requires_gpu=False,      # can run CPU, GPU preferred
            estimated_vram_mb=None,  # varies per model
        )

    def supports(self, job_type: JobType, model: str | None = None) -> bool:
        if job_type not in (JobType.TEXT_GENERATION, JobType.TEXT_EMBEDDING):
            return False
        if model and model not in SUPPORTED_MODELS:
            return False
        return True

    async def initialize(self) -> None:
        """Verify Ollama is reachable (if using ollama backend)."""
        from src.modules.providers.local_llm.config import BACKEND
        if BACKEND == "ollama":
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
                    models = [m["name"] for m in r.json().get("models", [])]
                    log.info("ollama_ready", url=OLLAMA_BASE_URL, models=models)
            except Exception as e:
                log.warning(
                    "ollama_unreachable",
                    url=OLLAMA_BASE_URL,
                    error=str(e),
                    hint="Start Ollama with: ollama serve",
                )

    async def execute(self, context: ExecutionContext) -> ProviderResult:
        model_id = context.model or "llama3.2"
        spec = SUPPORTED_MODELS.get(model_id)

        if spec is None:
            return ProviderResult(
                success=False,
                error_message=(
                    f"Unknown model '{model_id}'. "
                    f"Available: {list(SUPPORTED_MODELS.keys())}"
                ),
            )

        if context.job_type == JobType.TEXT_GENERATION:
            if spec.backend == "ollama":
                return await self._generate_ollama(spec, context)
            else:
                return await self._generate_transformers(spec, context)

        elif context.job_type == JobType.TEXT_EMBEDDING:
            return await self._embed_ollama(spec, context)

        return ProviderResult(
            success=False,
            error_message=f"Unsupported job_type: {context.job_type}",
        )

    async def cancel(self, job_id: UUID) -> bool:
        # Streaming generation can be cancelled by closing the HTTP connection.
        # For now, we signal cancellation and the stream will naturally stop.
        return True

    async def health_check(self) -> dict[str, Any]:
        status = "ok"
        ollama_models: list[str] = []
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
                ollama_models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            status = "ollama_unreachable"

        return {
            "provider": self.provider_id,
            "status": status,
            "ollama_url": OLLAMA_BASE_URL,
            "ollama_models": ollama_models,
            "hf_models_loaded": list(self._transformers_models.keys()),
        }

    # ─── Ollama backend ───────────────────────────────────────────────────────

    async def _generate_ollama(
        self, spec: LLMModelSpec, context: ExecutionContext
    ) -> ProviderResult:
        """Streams generation from Ollama's /api/chat endpoint."""
        try:
            import httpx
        except ImportError:
            return ProviderResult(
                success=False,
                error_message="httpx not installed. Run: pip install httpx",
            )

        payload = context.input_payload
        messages = []
        if payload.get("system_prompt"):
            messages.append({"role": "system", "content": payload["system_prompt"]})
        messages.append({"role": "user", "content": payload["prompt"]})

        await context.on_progress(0.0, "Connecting to Ollama")

        full_text = ""
        token_count = 0

        try:
            async with httpx.AsyncClient(timeout=context.timeout_seconds or 600) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": spec.ollama_model,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "temperature": float(payload.get("temperature", 0.7)),
                            "top_p": float(payload.get("top_p", 0.9)),
                            "num_predict": int(payload.get("max_tokens", 2048)),
                        },
                    },
                ) as response:
                    response.raise_for_status()
                    await context.on_progress(5.0, "Generating")

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        import json
                        chunk = json.loads(line)
                        if chunk.get("message", {}).get("content"):
                            token = chunk["message"]["content"]
                            full_text += token
                            token_count += 1

                            # Report progress every 50 tokens
                            if token_count % 50 == 0:
                                # Progress is approximate — we don't know total tokens ahead of time
                                max_tokens = int(payload.get("max_tokens", 2048))
                                pct = min(95.0, 5.0 + (token_count / max_tokens) * 90.0)
                                await context.on_progress(pct, f"{token_count} tokens")

                        if chunk.get("done"):
                            break

        except httpx.ConnectError:
            return ProviderResult(
                success=False,
                error_message=(
                    f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                    "Is it running? Try: ollama serve"
                ),
            )
        except Exception as e:
            return ProviderResult(success=False, error_message=f"Ollama error: {e}")

        # Save output as text artifact
        artifact_key = storage.output_key(str(context.job_id), "response.txt")
        await storage.upload(artifact_key, full_text.encode(), "text/plain")
        await context.on_artifact(artifact_key, "text", "text/plain")
        await context.on_progress(100.0, "Done")

        return ProviderResult(
            success=True,
            result_summary={
                "model": spec.ollama_model,
                "tokens_generated": token_count,
                "response_preview": full_text[:200] + "..." if len(full_text) > 200 else full_text,
            },
            artifact_keys=[artifact_key],
            execution_metadata={"backend": "ollama", "token_count": token_count},
        )

    async def _embed_ollama(
        self, spec: LLMModelSpec, context: ExecutionContext
    ) -> ProviderResult:
        """Generates embeddings via Ollama's /api/embed endpoint."""
        try:
            import httpx
            import json
        except ImportError:
            return ProviderResult(success=False, error_message="httpx not installed")

        payload = context.input_payload
        texts = payload.get("texts", [payload.get("prompt", "")])

        await context.on_progress(0.0, "Generating embeddings")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": spec.ollama_model, "input": texts},
                )
                r.raise_for_status()
                embeddings = r.json().get("embeddings", [])
        except Exception as e:
            return ProviderResult(success=False, error_message=f"Embedding error: {e}")

        artifact_key = storage.output_key(str(context.job_id), "embeddings.json")
        await storage.upload(
            artifact_key,
            json.dumps({"embeddings": embeddings, "model": spec.ollama_model}).encode(),
            "application/json",
        )
        await context.on_artifact(artifact_key, "json", "application/json")
        await context.on_progress(100.0, "Done")

        return ProviderResult(
            success=True,
            result_summary={
                "texts_embedded": len(texts),
                "dimensions": len(embeddings[0]) if embeddings else 0,
            },
            artifact_keys=[artifact_key],
        )

    # ─── Transformers backend ─────────────────────────────────────────────────

    async def _generate_transformers(
        self, spec: LLMModelSpec, context: ExecutionContext
    ) -> ProviderResult:
        """Loads and runs a HuggingFace Transformers model directly."""
        try:
            model, tokenizer = await self._get_transformers_model(spec)
        except Exception as e:
            return ProviderResult(success=False, error_message=f"Model load failed: {e}")

        payload = context.input_payload
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", 2048))

        await context.on_progress(5.0, "Tokenizing")
        loop = asyncio.get_event_loop()

        def _generate_sync() -> str:
            import torch
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=float(payload.get("temperature", 0.7)),
                    do_sample=True,
                    top_p=float(payload.get("top_p", 0.9)),
                )
            return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        await context.on_progress(10.0, "Generating")
        text = await loop.run_in_executor(None, _generate_sync)
        await context.on_progress(95.0, "Saving output")

        artifact_key = storage.output_key(str(context.job_id), "response.txt")
        await storage.upload(artifact_key, text.encode(), "text/plain")
        await context.on_artifact(artifact_key, "text", "text/plain")
        await context.on_progress(100.0, "Done")

        return ProviderResult(
            success=True,
            result_summary={"model": spec.model_id, "response_length": len(text)},
            artifact_keys=[artifact_key],
            execution_metadata={"backend": "transformers"},
        )

    async def _get_transformers_model(self, spec: LLMModelSpec) -> tuple[Any, Any]:
        """Loads and caches a HuggingFace model + tokenizer."""
        model_id = spec.model_id
        if model_id in self._transformers_models:
            return self._transformers_models[model_id]

        if model_id not in self._transformers_locks:
            self._transformers_locks[model_id] = asyncio.Lock()

        async with self._transformers_locks[model_id]:
            if model_id in self._transformers_models:
                return self._transformers_models[model_id]

            loop = asyncio.get_event_loop()
            model, tokenizer = await loop.run_in_executor(
                None, lambda: self._load_transformers_sync(spec)
            )
            self._transformers_models[model_id] = (model, tokenizer)
            return model, tokenizer

    def _load_transformers_sync(self, spec: LLMModelSpec) -> tuple[Any, Any]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError(
                f"transformers not installed: {e}. "
                "Run: pip install transformers accelerate bitsandbytes"
            ) from e

        import torch
        load_kwargs: dict[str, Any] = {"device_map": "auto"}

        if spec.quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif spec.quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.float16

        path = spec.hf_repo_or_path
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        log.info("transformers_model_loaded", model_id=spec.model_id, path=path)
        return model, tokenizer
