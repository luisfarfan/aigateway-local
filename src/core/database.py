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
    echo=_settings.debug,           # logs all SQL in debug mode
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


async def create_all_tables() -> None:
    """
    Creates all tables defined in SQLModel metadata.
    Called at app startup in development. In production, use Alembic migrations.

    Importing all models here ensures SQLModel metadata is populated before create_all.
    """
    # These imports must happen before create_all so SQLModel registers the tables.
    import src.modules.jobs.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    log.info("database_tables_created")


async def dispose_engine() -> None:
    """Called at app shutdown to cleanly close all DB connections."""
    await engine.dispose()
    log.info("database_engine_disposed")
