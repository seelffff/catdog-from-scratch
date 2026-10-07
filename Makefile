.PHONY: model install build serve test

model:
	python scripts/download_model.py

install:
	python -m pip install -e '.[training,dev]'
	python -m pip install -e 'backend[dev]'
	cd frontend && npm ci

build:
	cd frontend && VITE_API_MODE=live npm run build

serve: build
	MODEL_PATH=models/v2_best.pt STATIC_DIR=frontend/dist uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000

test:
	python -m pytest tests -q
	cd backend && python -m pytest -q
	cd frontend && npm run typecheck && npm test
