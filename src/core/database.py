"""
Async PostgreSQL database setup.

Uses SQLAlchemy async engine + asyncpg driver.
SQLModel tables are defined in modules/jobs/models.py — imported here for Alembic discovery.

Usage in FastAPI endpoints (dependency injection):
    async def my_endpoint(session: AsyncSession = Depends(get_session)):
        ...
"""

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from src.core.config import get_settings

log = structlog.get_logger(__name__)

_settings = get_settings()

# Single engine instance — shared across the app lifetime.
# pool_pre_ping: validates connections before use (handles dropped DB connections).
engine = create_async_engine(
    _settings.database_url,
    echo=False,  # keep logs clean
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Session factory — expire_on_commit=False keeps objects usable after commit
# (important for returning data from endpoints after committing).
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.
    Commits on success, rolls back on exception, always closes.

    Usage:
        async def endpoint(session: AsyncSession = Depends(get_session)):
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def _import_models() -> None:
    """Registra todos los modelos en la metadata de SQLModel.

    Un módulo que no se importe acá es invisible tanto para `create_all` como
    para el autogenerate de Alembic. Faltaba `observability`, y por eso una
    baseline autogenerada salió sin `llm_requests` ni `llm_attempts` — sin las
    tablas que sostienen toda la contabilidad de costos.
    """
    import src.modules.jobs.models  # noqa: F401
    import src.modules.observability.models  # noqa: F401


async def create_all_tables() -> None:
    """Crea las tablas desde los modelos. **Sólo para desarrollo.**

    En producción no se llama: el esquema lo pone `alembic upgrade head`. Tener
    las dos vías activas a la vez es lo que produjo el desajuste que rompió el
    plano de jobs — `create_all` crea lo que falta pero NO altera lo que ya
    existe, así que un campo renombrado en los modelos queda con el nombre viejo
    en la base y nadie se entera hasta que un INSERT falla.
    """
    _import_models()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    log.info("database_tables_created")


class SchemaOutOfDate(RuntimeError):
    """La base no está en la última revisión de Alembic."""


async def verify_schema_is_current() -> None:
    """Comprueba que la base esté migrada. Levanta si no, para no arrancar.

    Arrancar contra un esquema viejo no da un error limpio: da fallos parciales
    y tardíos —un INSERT que revienta cuando ya se llamó al proveedor y se subió
    el archivo— y eso se diagnostica mucho peor que no arrancar.

    Si la comprobación misma falla (no se puede leer la configuración de
    Alembic), se avisa y se sigue: un bug en el chequeo no puede dejar el
    servicio en el piso.
    """
    from pathlib import Path

    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import text

    try:
        raiz = Path(__file__).resolve().parents[2]
        script = ScriptDirectory.from_config(Config(str(raiz / "alembic.ini")))
        cabeza = script.get_current_head()
    except Exception as exc:  # noqa: BLE001
        log.warning("schema_check_skipped", error=str(exc)[:200])
        return

    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT version_num FROM alembic_version"))
        actual = result.scalar_one_or_none()

    if actual == cabeza:
        log.info("schema_current", revision=actual)
        return

    raise SchemaOutOfDate(
        f"La base está en la revisión {actual or '(ninguna)'} y el código espera "
        f"{cabeza}. Correr `alembic upgrade head` antes de arrancar. Si la base ya "
        f"tiene el esquema correcto pero nunca se marcó, `alembic stamp head`."
    )


async def dispose_engine() -> None:
    """Called at app shutdown to cleanly close all DB connections."""
    await engine.dispose()
    log.info("database_engine_disposed")
