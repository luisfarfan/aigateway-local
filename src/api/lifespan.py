"""
FastAPI lifespan — startup and shutdown hooks.

Startup order:
  1. Logging
  2. DB tables (dev) / verify connection (prod)
  3. MinIO bucket
  4. Redis + ARQ pool
  5. Register providers in ProviderRegistry
  6. Attach shared state to app

Shutdown order (reverse):
  1. ARQ pool
  2. Redis pool
  3. DB engine
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI

from src.core.config import get_settings
from src.core.database import create_all_tables, dispose_engine, verify_schema_is_current
from src.core.logging import configure_logging
from src.core.redis import close_arq_pool, close_redis, get_arq_pool
from src.core.storage import storage
from src.modules.providers.registry import ProviderRegistry
from src.modules.providers.stub.provider import StubProvider

log = structlog.get_logger(__name__)


def _build_provider_registry(settings: Any) -> ProviderRegistry:
    """
    Instantiates and registers all available provider adapters.

    Control which providers load via env vars (set in .env):
      ENABLE_PROVIDER_STUB=true        always on in development
      ENABLE_PROVIDER_DIFFUSERS=true   requires diffusers + torch on Ubuntu
      ENABLE_PROVIDER_LOCAL_LLM=true   requires Ollama running or transformers
      ENABLE_PROVIDER_LOCAL_TTS=true   requires TTS engine installed

    To add a new engine:
      1. Create its adapter in src/modules/providers/{name}/provider.py
      2. Import it here and add its enable flag below.
    """
    registry = ProviderRegistry()
    from src.modules.providers.stub.provider import StubProvider

    # Stub — always enabled in dev, optional in prod
    if settings.enable_provider_stub:
        registry.register(StubProvider(step_delay_seconds=1.5))

    # CLIProxyAPI — modelos cloud por OAuth. No usa GPU: su límite es la cuota
    # de arriba, no la VRAM.
    if settings.enable_provider_cliproxy:
        try:
            from src.modules.providers.cliproxy.provider import CliproxyProvider

            registry.register(CliproxyProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="cliproxy", error=str(e))

    # Diffusers — image/video generation via HuggingFace Diffusers
    if settings.enable_provider_diffusers:
        try:
            from src.modules.providers.diffusers.provider import DiffusersProvider
            registry.register(DiffusersProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="diffusers", error=str(e))

    # Local LLM — text generation via Ollama or HuggingFace Transformers
    if settings.enable_provider_local_llm:
        try:
            from src.modules.providers.local_llm.provider import LocalLLMProvider
            registry.register(LocalLLMProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="local_llm", error=str(e))

    # Local TTS — text-to-speech via XTTS / Kokoro / Piper
    if settings.enable_provider_local_tts:
        try:
            from src.modules.providers.local_tts.provider import LocalTTSProvider
            registry.register(LocalTTSProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="local_tts", error=str(e))

    # Local STT — speech-to-text via faster-whisper or openai-whisper
    if settings.enable_provider_local_stt:
        try:
            from src.modules.providers.local_stt.provider import LocalSTTProvider
            registry.register(LocalSTTProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="local_stt", error=str(e))

    # Video Editor — assembly & stitching
    if settings.enable_provider_video_editor:
        try:
            from src.modules.providers.video_editor.provider import VideoAssemblerProvider
            registry.register(VideoAssemblerProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="video_editor", error=str(e))

    # Orchestrator — CrewAI mission control
    if settings.enable_provider_orchestrator:
        try:
            from src.modules.providers.orchestrator.provider import CrewAIOrchestratorProvider
            registry.register(CrewAIOrchestratorProvider())
        except Exception as e:
            log.error("provider_load_failed", provider="orchestrator", error=str(e))

    registered = registry.list_provider_ids()
    log.info("providers_registered", providers=registered, count=len(registered))
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # ── Startup ──────────────────────────────────────────────────────────────
    configure_logging()
    log.info("gateway_starting", environment=settings.environment)

    # Trazas, lo PRIMERO de todo. OpenTelemetry no deja reemplazar un
    # TracerProvider ya instalado, y algunas dependencias (CrewAI) instalan el
    # suyo al importarse — cosa que pasa al construir el registro de providers,
    # más abajo. Si eso ocurre antes, nuestros spans se crean contra un
    # proveedor que nadie exporta y a Langfuse no llega nada, sin ningún error
    # visible.
    from src.modules.observability.tracing import configure_tracing

    configure_tracing(
        endpoint=settings.langfuse_otlp_endpoint,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        environment=settings.environment,
    )

    # Base de datos. Dos vías, y NUNCA las dos a la vez.
    #
    # En desarrollo se crean las tablas desde los modelos, que es cómodo. En
    # producción no: el esquema lo pone `alembic upgrade head` y el arranque sólo
    # COMPRUEBA que esté al día. Tener las dos activas fue lo que produjo el
    # desajuste que rompió el plano de jobs — `create_all` crea lo que falta pero
    # no altera lo que existe, así que un campo renombrado en los modelos se
    # queda con el nombre viejo en la base, en silencio, hasta que un INSERT
    # falla en caliente.
    if settings.db_auto_create_tables:
        await create_all_tables()
    else:
        await verify_schema_is_current()
    log.info("database_ready", auto_create=settings.db_auto_create_tables)

    # Object storage
    await storage.ensure_bucket()
    log.info("storage_ready", bucket=settings.minio_bucket)

    # Redis + ARQ
    arq_pool = await get_arq_pool()
    log.info("redis_ready")

    # Provider registry
    registry = _build_provider_registry(settings)

    # CLIProxyAPI — cliente único por proceso, reutiliza conexiones. Se crea
    # aunque el upstream esté caído: el plano síncrono responde 503/504 con un
    # error clasificado, que es más útil que no arrancar.
    cliproxy_client = None
    if settings.enable_provider_cliproxy:
        from src.modules.providers.cliproxy.client import CliproxyClient

        cliproxy_client = CliproxyClient(
            base_url=settings.cliproxy_base_url,
            api_key=settings.cliproxy_api_key,
            timeout_seconds=settings.cliproxy_timeout_s,
            image_timeout_seconds=settings.cliproxy_image_timeout_s,
            catalog_ttl_seconds=settings.cliproxy_catalog_ttl_s,
        )
        log.info("cliproxy_ready", base_url=cliproxy_client.base_url)

    # Registro de backends: cloud siempre, local si Ollama está habilitado. El
    # local es opcional a propósito — sin él las cadenas siguen funcionando, sólo
    # pierden su último recurso.
    backends = None
    # Se declara acá arriba y no dentro del `if`: se usa en el apagado, que
    # corre aunque CLIProxyAPI esté deshabilitado.
    gemini_web_backend = None
    if cliproxy_client is not None:
        from src.modules.backends.registry import BackendRegistry

        ollama_backend = None
        if settings.enable_backend_ollama:
            from src.modules.backends.ollama import OllamaBackend

            ollama_backend = OllamaBackend(
                base_url=settings.ollama_base_url,
                timeout_seconds=settings.ollama_timeout_s,
            )
            log.info("ollama_backend_ready", base_url=settings.ollama_base_url)

        # Último recurso de imagen. Sólo si está encendido Y hay cookie: sin
        # credencial el backend no puede hacer nada, y registrarlo vacío haría
        # que la cadena gaste un intento para descubrirlo en cada petición.
        if settings.enable_backend_gemini_web and settings.gemini_web_secure_1psid:
            from src.modules.backends.gemini_web import GeminiWebBackend

            gemini_web_backend = GeminiWebBackend(
                secure_1psid=settings.gemini_web_secure_1psid,
                secure_1psidts=settings.gemini_web_secure_1psidts,
                timeout_s=settings.gemini_web_timeout_s,
                cookie_cache_dir=str(
                    Path(settings.gemini_web_cookie_cache_dir).expanduser()
                ),
            )
            log.info("gemini_web_backend_ready")
        elif settings.enable_backend_gemini_web:
            log.warning("gemini_web_backend_skipped", reason="falta GEMINI_WEB_SECURE_1PSID")

        backends = BackendRegistry(
            cloud=cliproxy_client, local=ollama_backend, gemini_web=gemini_web_backend
        )

    # Vigilante de la sesión de la app web de Gemini. Va aparte del watchdog a
    # propósito: el watchdog abre circuitos —protege la latencia— pero no avisa
    # a nadie, y esta credencial no se recupera sola. Ver
    # `backends/gemini_web_monitor.py`.
    gemini_web_monitor_task = None
    if gemini_web_backend is not None:
        import asyncio

        from src.modules.backends.gemini_web_monitor import run_forever as watch_session
        from src.modules.notifications import NullNotifier, TelegramNotifier

        notifier = (
            TelegramNotifier(
                bot_token=settings.telegram_bot_token, chat_id=settings.telegram_chat_id
            )
            if settings.telegram_bot_token and settings.telegram_chat_id
            else NullNotifier()
        )
        gemini_web_monitor_task = asyncio.create_task(
            watch_session(
                gemini_web_backend,
                notifier,
                interval_s=settings.gemini_web_session_check_interval_s,
            )
        )
        log.info("gemini_web_monitor_started", notifier=notifier.name)

    # Watchdog de modelos. En la API y no en el worker porque basta con un
    # sondeador: el estado vive en Redis y lo comparten todos los procesos.
    watchdog_task = None
    if backends is not None and settings.watchdog_enabled:
        import asyncio

        from src.modules.routing.breaker import CircuitBreaker
        from src.modules.routing.config import load_routing
        from src.modules.routing.watchdog import run_forever

        routing_table = load_routing()
        watchdog_task = asyncio.create_task(
            run_forever(
                backends,
                routing_table,
                CircuitBreaker(routing_table.breaker),
                interval_s=settings.watchdog_interval_s,
            )
        )
        log.info("watchdog_started", interval_s=settings.watchdog_interval_s)

    # Attach shared state — accessible in routers via request.app.state.*
    app.state.provider_registry = registry
    app.state.arq_pool = arq_pool
    app.state.settings = settings
    app.state.cliproxy_client = cliproxy_client
    app.state.backends = backends

    log.info("gateway_started", host=settings.api_host, port=settings.api_port)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("gateway_stopping")
    if watchdog_task is not None:
        watchdog_task.cancel()
    if gemini_web_monitor_task is not None:
        gemini_web_monitor_task.cancel()
    if gemini_web_backend is not None:
        await gemini_web_backend.aclose()
    if cliproxy_client is not None:
        await cliproxy_client.aclose()
    await close_arq_pool()
    await close_redis()
    await dispose_engine()
    log.info("gateway_stopped")
