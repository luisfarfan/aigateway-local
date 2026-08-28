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
	@echo "    make sys-deps       Install system dependencies (ffmpeg, imagemagick)"
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
	@echo "    make clean-data     Clear Redis queues and reset DB tables"
	@echo ""

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

sys-deps:
	sudo apt-get update && sudo apt-get install -y ffmpeg imagemagick
	@echo "Enabling ImageMagick security policy for TextClip..."
	sudo sed -i 's/policy domain="path" rights="none" pattern="@\*"/policy domain="path" rights="read|write" pattern="@\*"/g' /etc/ImageMagick-6/policy.xml || true

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

# Igual que `api` pero sin --reload: para dejarlo sirviendo a la red de casa.
# Escucha en todas las interfaces, así que cualquier dispositivo del WiFi llega
# por http://<ip-de-esta-maquina>:8000. Requiere API_KEYS con una clave real en
# el .env — sin eso el gateway queda abierto a toda la red.
serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1

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

clean-data:
	@echo "Limpiando Redis y Base de Datos..."
	@export PYTHONPATH=$$PYTHONPATH:. ; \
	python scripts/clean_data.py
	@echo "Done. Redis y DB están limpias."
