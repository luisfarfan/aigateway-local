"""
Application configuration — single source of truth for all settings.

Loaded from environment variables (or .env file).
All settings are typed and validated by Pydantic at startup.
"""
from functools import lru_cache
from typing import Annotated

from pydantic import Field, computed_field, model_validator
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

    # Crear las tablas desde los modelos al arrancar. Cómodo en desarrollo,
    # PELIGROSO en producción: `create_all` crea lo que falta pero no altera lo
    # que ya existe, así que un campo renombrado en los modelos se queda con el
    # nombre viejo en la base y el desajuste no se nota hasta que un INSERT
    # falla. En producción se apaga y el arranque sólo comprueba que la base
    # esté en la última revisión de Alembic.
    db_auto_create_tables: bool = True

    @model_validator(mode="after")
    def _no_crear_tablas_en_produccion(self) -> "Settings":
        """En producción se apaga solo, sin depender de que alguien lo configure.

        Un default peligroso que hay que acordarse de desactivar termina activo
        el día que se despliega con prisa. Si de verdad hace falta —una base
        efímera de pruebas de carga, por ejemplo— se enciende explícito con
        `DB_AUTO_CREATE_TABLES=true`, y entonces es una decisión y no un olvido.
        """
        import os

        if self.environment == "production" and "DB_AUTO_CREATE_TABLES" not in os.environ:
            object.__setattr__(self, "db_auto_create_tables", False)
        return self
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
    
    # ─── Providers ────────────────────────────────────────────────────────────
    enable_provider_stub: bool = True
    enable_provider_diffusers: bool = False
    enable_provider_local_llm: bool = False
    enable_provider_local_tts: bool = False
    enable_provider_local_stt: bool = False
    enable_provider_orchestrator: bool = False
    enable_provider_video_editor: bool = False
    
    enable_provider_cliproxy: bool = True

    # --- CLIProxyAPI (modelos cloud por OAuth) ---
    # La base NO lleva `/v1`: las rutas se construyen con su prefijo completo,
    # y el cliente normaliza si igual viene puesto.
    cliproxy_base_url: str = "http://localhost:8417"
    cliproxy_api_key: str = ""
    cliproxy_timeout_s: float = 120.0
    # La imagen tarda mucho más que el chat: medido, entre 30 s y 148 s para la
    # misma petición según la carga de arriba. Un timeout único cancelaría a
    # mitad y gastaría la cuota sin traer nada.
    cliproxy_image_timeout_s: float = 420.0
    # Cada cuánto se refresca el mapa modelo → owned_by. No es un health check:
    # el catálogo lista modelos muertos igual (ver docs/F0-VANILLA-CAPABILITIES.md).
    cliproxy_catalog_ttl_s: float = 300.0
    cliproxy_default_model: str = "gemini-3-flash"

    # Cache de respuestas. La clave incluye el proyecto (X-Proxima-Project) para
    # que la contabilidad de costos no se mezcle entre consumidores.
    llm_cache_enabled: bool = True
    llm_cache_ttl_s: int = 3600
    llm_default_project: str = "default"
    # `X-Proxima-Project` obligatoria. Sin ella el gasto de todos los sistemas se
    # mezcla en un solo balde y la contabilidad por servicio no existe — medido:
    # el 97,8 % del tráfico caía en `default`. Se puede apagar por env var como
    # salida de emergencia, sin desplegar código.
    llm_require_project: bool = True

    # ─── Observabilidad (F3) ──────────────────────────────────────────────────
    # Langfuse acepta OTLP sobre HTTP con auth Basic. Sin claves, las trazas se
    # generan igual pero no se exportan: el código que abre spans no tiene que
    # saber si alguien escucha.
    langfuse_otlp_endpoint: str = "http://localhost:3777/api/public/otel/v1/traces"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # Escribir el histórico en Postgres. Se apaga en tests y en cualquier
    # despliegue sin base; la petición se sirve igual.
    llm_history_enabled: bool = True

    # ─── Routing (F4) ─────────────────────────────────────────────────────────
    # El watchdog prueba los modelos de las cadenas y abre el circuito de los
    # que no responden. `GET /v1/models` no sirve para esto: lista modelos con
    # la credencial revocada igual que los vivos.
    # Backend local. Es el último recurso de las cadenas: más lento, pero no
    # depende de cuota ni de internet.
    enable_backend_ollama: bool = True
    ollama_timeout_s: float = 300.0

    watchdog_enabled: bool = True
    watchdog_interval_s: int = 900

    # Último recurso de la ruta de imagen: la app web de Gemini por cookie de
    # sesión. Apagado por defecto — depende de una dependencia AGPL opcional y
    # de una credencial que es la sesión entera de una cuenta de Google, así que
    # se enciende a propósito, nunca por omisión.
    enable_backend_gemini_web: bool = False
    gemini_web_secure_1psid: str = ""
    gemini_web_secure_1psidts: str = ""
    gemini_web_timeout_s: float = 300.0
    # Dónde guarda la librería las cookies ya rotadas. Su default es /tmp, que
    # systemd vacía al arrancar: con eso, cada reinicio vuelve a la semilla del
    # `.env` —vieja— y la sesión se muere sola. Persistirlo rompe ese ciclo.
    gemini_web_cookie_cache_dir: str = "~/.proxima-gateway/gemini-web-cache"
    # Cada cuánto se comprueba que la cookie siga viva. 30 min es holgado a
    # propósito: la sonda habla con Google, y una sesión que expiró va a seguir
    # expirada — no hay nada que ganar preguntando cada minuto.
    gemini_web_session_check_interval_s: int = 1800

    # Aviso al operador cuando algo necesita una mano humana (hoy: la sesión de
    # la app web de Gemini). Sin token configurado no se manda nada y se
    # registra en el log; el gateway funciona igual.
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- Local Engine Settings ---
    ollama_base_url: str = "http://localhost:11434"
    local_llm_backend: str = "ollama"
    local_tts_engine: str = "xtts"

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
