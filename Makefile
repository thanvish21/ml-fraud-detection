.PHONY: data train explain api up down logs clean

PY ?= python3

data:
	$(PY) ml/generate_data.py

train: data
	cd ml && $(PY) train.py

explain:
	cd ml && $(PY) explain.py

api:
	uvicorn api.main:app --reload --port 8000

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=100

clean:
	rm -rf data/*.csv models/*.joblib models/*.png models/metrics.json mlruns
