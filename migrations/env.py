"""
Alembic environment — async-aware configuration.

Uses the sync DSN (psycopg2) for running migrations, which is how Alembic
handles async engines: migrations run synchronously but read the same models.

To generate a new migration:
    alembic revision --autogenerate -m "describe change"

To apply:
    alembic upgrade head
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

# Import all SQLModel table models so Alembic can detect them for autogenerate.
# Add new models here when created.
import src.modules.jobs.models  # noqa: F401

from src.core.config import get_settings

settings = get_settings()

# Alembic config object (alembic.ini)
config = context.config

# Set DSN dynamically from our Settings (overrides alembic.ini's empty sqlalchemy.url)
config.set_main_option("sqlalchemy.url", settings.database_url_sync)

# Configure stdlib logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SQLModel.metadata contains all table definitions from imported models above
target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL script)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,          # detect column type changes
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
