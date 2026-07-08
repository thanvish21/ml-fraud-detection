# ml-fraud-detection

Production-grade real-time fraud detection: synthetic data → tuned ensemble → low-latency API → Kafka streaming → live React dashboard. Everything orchestrated by Docker Compose.

## Stack
| Layer       | Tech |
|-------------|------|
| Modeling    | XGBoost · LightGBM · IsolationForest ensemble, Optuna tuning, MLflow tracking, SHAP explainability |
| Serving     | FastAPI (`/predict`, `/predict/bulk`) · champion/challenger A/B · Prometheus `/metrics` |
| Streaming   | Kafka producer @ ~1000 tx/sec → consumer → PostgreSQL · dead-letter queue |
| Dashboard   | React + Vite + Recharts · live gauge, timeline, patterns, model panel |
| Observability | Prometheus + Grafana, drift + latency alerts |

## Architecture

See `docs/architecture.md` for the full diagram.

```
Producer → Kafka → Consumer → API → PostgreSQL
                              ↓
                        Prometheus → Grafana
                              ↓
            Stats API ← PostgreSQL → React Dashboard
```

## Quick start

```bash
# 1. Generate data + train models locally
python -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt
python ml/generate_data.py
cd ml && python train.py && python explain.py && cd ..

# 2. (optional) Override dev-only default passwords
cp .env.example .env   # then edit POSTGRES_PASSWORD / GRAFANA_ADMIN_PASSWORD

# 3. Bring up the full stack
docker compose up -d --build
```

Services (after `compose up`):

| URL                          | Purpose |
|------------------------------|---------|
| http://localhost:3000        | React dashboard (live) |
| http://localhost:8000/docs   | FastAPI inference docs |
| http://localhost:8001/health | Stats API |
| http://localhost:9090        | Prometheus |
| http://localhost:3001        | Grafana (admin / `$GRAFANA_ADMIN_PASSWORD`, default `admin`) |
| http://localhost:5000        | MLflow UI |

## ML pipeline (`ml/`)

- `generate_data.py` → 100k synthetic transactions with realistic fraud signal in `amount`, `time_of_day`, `user_velocity`, `geo_distance`.
- `features.py` → scikit-learn ColumnTransformer with interaction features (`log_amount`, `amount_velocity`, `geo_velocity`, `night_tx`).
- `train.py` → Optuna-tuned XGBoost + LightGBM + IsolationForest, weighted ensemble, MLflow runs, joblib persistence.
- `explain.py` → SHAP summary plots for tree models.
- `notebooks/01_eda_and_modeling.ipynb` → end-to-end EDA, training, model comparison.

## Inference API (`api/`)

```bash
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"transaction_id":"tx_1","amount":1899.50,"merchant_category":"electronics","time_of_day":2.5,"user_velocity":14.0,"geo_distance":220.0}'
```

- p99 latency target: **<50 ms** (warm path).
- `?variant=challenger` forces the challenger; default `auto` deterministically hashes `transaction_id` for a 90/10 A/B split.
- `/metrics` exposes `fraud_requests_total`, `fraud_predict_latency_seconds`, `fraud_rate_observed`, `fraud_model_drift_alert`.

## Streaming (`streaming/`)

- `producer.py` simulates ~1000 tx/sec with ~4% baseline fraud-like patterns.
- `consumer.py` calls the inference API and bulk-inserts to PostgreSQL. Failed scores are republished to `transactions.dlq`.
- `stats_api.py` exposes the aggregations powering the dashboard.

## Dashboard (`dashboard/`)

Recharts-driven UI polling every 5s:
- Live fraud-rate gauge (last 5 min)
- Transaction volume timeline (last 30 min, total vs flagged)
- Top fraud patterns by merchant category (last 1 h)
- Model performance panel — request volume, mean latency, mean predicted probability per model variant

## Operations

```bash
make data      # synth dataset
make train     # train all models + ensemble
make explain   # SHAP plots
make up        # docker compose up
make down      # tear down + volumes
make logs      # tail all services
```

## Layout

```
ml-fraud-detection/
├── ml/                 # data gen, features, train, explain
├── notebooks/          # EDA + modeling notebook
├── api/                # FastAPI serving + model registry
├── streaming/          # Kafka producer/consumer + stats API
├── dashboard/          # React + Vite + Recharts
├── docker/             # Prometheus + alert rules
├── docs/architecture.md
├── docker-compose.yml
└── Makefile
```
