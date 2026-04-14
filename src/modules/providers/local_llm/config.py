"""
Local LLM provider configuration.

Supports two backends, configured via env vars:
  LOCAL_LLM_BACKEND=ollama        (default) — REST API, easiest to use
  LOCAL_LLM_BACKEND=transformers  — direct HuggingFace Transformers loading

Ollama backend:
  - Ollama must be running: systemctl start ollama (or ollama serve)
  - Models pulled with: ollama pull llama3.2, ollama pull mistral, etc.
  - OLLAMA_BASE_URL defaults to http://localhost:11434

Transformers backend:
  - Models loaded directly from HF Hub or local paths
  - Env var per model: LOCAL_LLM_PATH_{MODEL_ID}=/data/models/llama3
  - Uses bitsandbytes for quantization if installed

Model registry:
  Keys are the model_id strings clients send in API requests.
  Values configure how the model is loaded/called.
"""
import os
from dataclasses import dataclass, field


@dataclass
class LLMModelSpec:
    model_id: str                       # what the client sends: e.g. "llama3.2"
    backend: str                        # "ollama" | "transformers"
    ollama_model: str | None = None    # name in ollama: e.g. "llama3.2:8b"
    hf_repo_or_path: str | None = None # HF Hub ID or local path
    context_length: int = 4096
    quantization: str | None = None    # "4bit" | "8bit" | None (transformers only)
    vram_mb: int = 8192


def _local_path(model_id: str, default: str) -> str:
    env_key = f"LOCAL_LLM_PATH_{model_id.upper().replace('-', '_').replace('.', '_')}"
    return os.environ.get(env_key, default)


BACKEND = os.environ.get("LOCAL_LLM_BACKEND", "ollama")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Model registry ───────────────────────────────────────────────────────────
# Add your installed models here.
# For Ollama: match the name you used with `ollama pull`.
# For Transformers: set LOCAL_LLM_PATH_{ID} to the local model directory.

SUPPORTED_MODELS: dict[str, LLMModelSpec] = {

    "llama3.2": LLMModelSpec(
        model_id="llama3.2",
        backend="ollama",
        ollama_model="llama3.2",
        context_length=8192,
        vram_mb=6144,
    ),

    "llama3.2:3b": LLMModelSpec(
        model_id="llama3.2:3b",
        backend="ollama",
        ollama_model="llama3.2:3b",
        context_length=8192,
        vram_mb=3072,
    ),

    "mistral": LLMModelSpec(
        model_id="mistral",
        backend="ollama",
        ollama_model="mistral",
        context_length=8192,
        vram_mb=5120,
    ),

    "mixtral": LLMModelSpec(
        model_id="mixtral",
        backend="ollama",
        ollama_model="mixtral",
        context_length=32768,
        vram_mb=24576,
    ),

    "qwen2.5": LLMModelSpec(
        model_id="qwen2.5",
        backend="ollama",
        ollama_model="qwen2.5",
        context_length=32768,
        vram_mb=8192,
    ),

    "deepseek-r1": LLMModelSpec(
        model_id="deepseek-r1",
        backend="ollama",
        ollama_model="deepseek-r1",
        context_length=65536,
        vram_mb=16384,
    ),

    "codellama": LLMModelSpec(
        model_id="codellama",
        backend="ollama",
        ollama_model="codellama",
        context_length=16384,
        vram_mb=7168,
    ),

    # Transformers backend example — uses local path if env var is set
    "llama3-hf": LLMModelSpec(
        model_id="llama3-hf",
        backend="transformers",
        hf_repo_or_path=_local_path("llama3-hf", "meta-llama/Meta-Llama-3-8B-Instruct"),
        context_length=8192,
        quantization="4bit",
        vram_mb=6144,
    ),
}
