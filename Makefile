.PHONY: setup setup-frontend setup-backend models dev-frontend dev-backend dev-backend-mock test test-frontend test-backend test-cv lint build check

VENV := backend/.venv
PY := $(VENV)/bin/python

setup: setup-frontend setup-backend

setup-frontend:
	cd frontend && npm install

setup-backend:
	python3 -m venv $(VENV)
	$(PY) -m pip install -e 'backend[dev,cv]'
	$(MAKE) models

# YOLO + TrackNetV3 weights into backend/data/models (git-ignored).
models:
	$(PY) backend/scripts/download_models.py

# Run each in its own terminal.
dev-frontend:
	cd frontend && npm run dev

dev-backend:
	cd backend && .venv/bin/uvicorn app.main:app --reload --reload-dir app --reload-dir rallycv --port 8000

# Simulated analysis data (no models needed), for UI work.
dev-backend-mock:
	cd backend && RALLYREVIEW_ANALYZER=mock .venv/bin/uvicorn app.main:app --reload --port 8000

test: test-frontend test-backend

test-frontend:
	cd frontend && npm test

test-backend:
	cd backend && .venv/bin/pytest

# Slow tests: runs the models on real footage in backend/data/samples.
test-cv:
	cd backend && .venv/bin/pytest -m slow -v

lint:
	cd frontend && npm run lint
	cd backend && .venv/bin/ruff check . && .venv/bin/ruff format --check .

build:
	cd frontend && npm run build

# Everything CI would run.
check: lint test build
