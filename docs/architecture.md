# System Architecture

```
                                  ┌──────────────────────────────┐
                                  │       React Dashboard        │
                                  │   (Recharts, polls 5s)       │
                                  └────────────┬─────────────────┘
                                               │ HTTP
                                  ┌────────────▼─────────────────┐
                                  │     Stats API (FastAPI)      │
                                  │  /stats/live /timeline ...   │
                                  └────────────┬─────────────────┘
                                               │ SQL
   ┌─────────────────┐    Kafka     ┌──────────▼────────┐   ┌───────────────┐
   │  Producer       │─ transactions─▶  Consumer (xN)    │──▶│  PostgreSQL   │
   │  1000 tx/sec    │              │  scores + sinks   │   │  predictions  │
   └─────────────────┘              └─────┬─────────────┘   └───────────────┘
                                          │ HTTP                ▲
                                          ▼                     │
                                  ┌───────────────────┐         │
                                  │  Inference API    │─────────┘
                                  │  (FastAPI)        │
                                  │  /predict (<50ms) │──┐
                                  │  /predict/bulk    │  │
                                  └─────────┬─────────┘  │ /metrics
                                            │            ▼
                                            │     ┌──────────────┐
                                            │     │  Prometheus  │──▶ Grafana
                                            │     └──────────────┘
              ┌─────────────────────┐       │
              │  Model Registry     │◀──────┘
              │  champion/challenger│
              │  (joblib + meta)    │
              └─────────┬───────────┘
                        │ loaded at startup
                        ▼
        ┌──────────────────────────────┐
        │   Training Pipeline          │
        │   XGBoost + LightGBM +       │      ┌───────────────┐
        │   IsolationForest ensemble   │─────▶│   MLflow      │
        │   Optuna · SHAP · sklearn    │      │  experiments  │
        └──────────────────────────────┘      └───────────────┘
            ▲
            │
            └── 100k synthetic transactions (data/transactions.csv)
```

## Data flow
1. **Producer** emits ~1000 transactions/sec to Kafka topic `transactions`.
2. **Consumer** (scaled to 2 replicas) POSTs each tx to the **Inference API**.
3. **Inference API** loads the latest preprocessor + champion/challenger models from `models/` and routes 10% of traffic to the challenger via deterministic hashing of `transaction_id`.
4. Predictions land in PostgreSQL. Failed scores go to the `transactions.dlq` topic.
5. The **Stats API** aggregates over 5-minute and 1-hour windows for the React dashboard.
6. **Prometheus** scrapes `/metrics` from the API; alerts fire on `HighFraudRate`, `ModelDrift`, `HighPredictLatency` (p99 > 50 ms).

## Latency budget (`/predict`)
| Step               | Target |
|--------------------|--------|
| JSON parse + validate | <1 ms  |
| Preprocessor + interaction features | <3 ms  |
| Model inference (XGBoost/LightGBM)  | <8 ms  |
| Response serialize | <1 ms  |
| **p99 end-to-end** | **<50 ms** |
