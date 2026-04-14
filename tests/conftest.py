"""
Pytest fixtures shared across all tests.

Test strategy:
  - Unit tests: mock DB/Redis, test logic in isolation
  - Integration tests: use real Postgres + Redis via Docker (testcontainers or docker-compose)

For now, fixtures provide an async test client with:
  - In-memory SQLite for DB (fast, no Docker needed for unit tests)
  - FakeRedis for pub/sub
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from src.api.app import create_app
from src.core.database import get_session
from src.modules.providers.registry import ProviderRegistry
from src.modules.providers.stub.provider import StubProvider


# ─── Database ─────────────────────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        # Import all models so SQLModel knows about them
        import src.modules.jobs.models  # noqa: F401
        await conn.run_sync(SQLModel.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


# ─── App client ───────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    """
    Async HTTP test client with overridden DB session and a stub provider registry.
    No real Redis or MinIO needed for unit tests.
    """
    app = create_app()

    # Override DB dependency
    async def override_session():
        yield db_session

    app.dependency_overrides[get_session] = override_session

    # Inject stub registry and a fake ARQ pool
    registry = ProviderRegistry()
    registry.register(StubProvider(step_delay_seconds=0.01))

    class FakeArqPool:
        async def enqueue_job(self, *args, **kwargs):
            class FakeJob:
                job_id = "fake-arq-id"
            return FakeJob()

    app.state.provider_registry = registry
    app.state.arq_pool = FakeArqPool()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
