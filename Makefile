.PHONY: help install dev-install up down logs db-migrate db-upgrade db-downgrade \
        worker api lint format typecheck test clean

# ─── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Local AI Gateway — Dev Commands"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install production dependencies"
	@echo "    make dev-install    Install all dependencies (including dev)"
	@echo "    make cp-env         Copy .env.example → .env"
	@echo ""
	@echo "  Infrastructure:"
	@echo "    make up             Start PostgreSQL + Redis + MinIO"
	@echo "    make down           Stop all services"
	@echo "    make logs           Tail service logs"
	@echo ""
	@echo "  Database:"
	@echo "    make db-migrate m=\"describe change\"   Generate migration"
	@echo "    make db-upgrade     Apply pending migrations"
	@echo "    make db-downgrade   Rollback last migration"
	@echo ""
	@echo "  Run:"
	@echo "    make api            Start the FastAPI server (dev)"
	@echo "    make worker         Start the ARQ worker"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint           Run ruff linter"
	@echo "    make format         Auto-format with ruff"
	@echo "    make typecheck      Run mypy"
	@echo "    make test           Run test suite"
	@echo ""

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

cp-env:
	cp .env.example .env
	@echo ".env created — edit it before running."

# ─── Infrastructure ───────────────────────────────────────────────────────────
up:
	docker compose up -d
	@echo "Services up. MinIO console: http://localhost:9001"

down:
	docker compose down

logs:
	docker compose logs -f

# ─── Database ─────────────────────────────────────────────────────────────────
db-migrate:
	@if [ -z "$(m)" ]; then echo "Usage: make db-migrate m='describe change'"; exit 1; fi
	alembic revision --autogenerate -m "$(m)"

db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-reset:
	alembic downgrade base && alembic upgrade head

# ─── Run ──────────────────────────────────────────────────────────────────────
api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

worker:
	python -m workers.main

# ─── Quality ──────────────────────────────────────────────────────────────────
lint:
	ruff check src/ workers/ tests/

format:
	ruff format src/ workers/ tests/
	ruff check --fix src/ workers/ tests/

typecheck:
	mypy src/ workers/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ─── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
