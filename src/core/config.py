"""
Application configuration — single source of truth for all settings.

Loaded from environment variables (or .env file).
All settings are typed and validated by Pydantic at startup.
"""
from functools import lru_cache
from typing import Annotated

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Environment ──────────────────────────────────────────────────────────
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = False          # structured JSON logs (use true in production)

    # ─── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Comma-separated API keys: "key1,key2,key3"
    api_keys: str = Field(default="", description="Comma-separated valid API keys")
    rate_limit_per_minute: int = 120

    @computed_field
    @property
    def valid_api_keys(self) -> frozenset[str]:
        """Parsed set of valid API keys for O(1) lookup."""
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    # ─── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "local_ai_gateway"
    postgres_user: str = "gateway"
    postgres_password: str = "gateway_secret"

    @computed_field
    @property
    def database_url(self) -> str:
        """Async DSN for SQLAlchemy + asyncpg."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """Sync DSN used only by Alembic migrations (not at runtime)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ─── Redis ────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_channel_ttl: int = 3600   # SSE pub/sub channel lifetime (seconds)

    @computed_field
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ─── MinIO / Object Storage ───────────────────────────────────────────────
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "local-ai-gateway"
    minio_region: str = "us-east-1"
    minio_presigned_expiry: int = 86400  # 24h

    # ─── Queue / concurrency ──────────────────────────────────────────────────
    queue_global_concurrency: int = 4
    queue_text_concurrency: int = 4
    queue_audio_concurrency: int = 2
    queue_image_concurrency: int = 1
    queue_video_concurrency: int = 1
    queue_pipeline_concurrency: int = 1

    job_default_timeout: int = 3600     # seconds
    arq_result_ttl: int = 86400         # seconds
    worker_heartbeat_interval: int = 15  # seconds

    # ─── Worker ───────────────────────────────────────────────────────────────
    worker_modalities: str = "text,audio,image,video,pipeline"
    worker_id: str = ""     # auto-generated at startup if empty

    @computed_field
    @property
    def enabled_modalities(self) -> list[str]:
        return [m.strip() for m in self.worker_modalities.split(",") if m.strip()]

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Returns the cached Settings singleton.
    Use this throughout the app: `from src.core.config import get_settings`
    Cached after first call — safe to call anywhere without performance cost.
    """
    return Settings()
