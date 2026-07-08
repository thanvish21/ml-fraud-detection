"""FastAPI fraud-prediction service with versioning, A/B testing, and Prometheus metrics."""
from __future__ import annotations
import os
import time
import random
import zlib
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from starlette.responses import Response

from .registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))

app = FastAPI(title="Fraud Detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

registry = ModelRegistry(MODELS_DIR)

REQUESTS = Counter("fraud_requests_total", "Total prediction requests", ["endpoint", "model"])
LATENCY = Histogram(
    "fraud_predict_latency_seconds", "Prediction latency", ["endpoint", "model"],
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
FRAUD_RATE = Gauge("fraud_rate_observed", "Rolling observed fraud rate", ["model"])
DRIFT_ALERT = Gauge("fraud_model_drift_alert", "1 if model drift detected", ["model"])
PRED_COUNT = Counter("fraud_predictions_total", "Total predictions by class", ["model", "label"])

ROLLING_WINDOW = 1000
_rolling: dict[str, list[int]] = {}


class Transaction(BaseModel):
    transaction_id: str = Field(..., examples=["tx_0001"])
    amount: float = Field(..., gt=0)
    merchant_category: str
    time_of_day: float = Field(..., ge=0, le=24)
    user_velocity: float = Field(..., ge=0)
    geo_distance: float = Field(..., ge=0)
    device_fingerprint: str | None = None


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    model_version: str
    model_name: str
    latency_ms: float


class BulkRequest(BaseModel):
    transactions: list[Transaction]
    variant: Literal["champion", "challenger", "auto"] = "auto"


def _record_rate(model: str, label: int):
    arr = _rolling.setdefault(model, [])
    arr.append(label)
    if len(arr) > ROLLING_WINDOW:
        arr.pop(0)
    rate = sum(arr) / len(arr)
    FRAUD_RATE.labels(model=model).set(rate)
    baseline = 0.018
    DRIFT_ALERT.labels(model=model).set(1 if abs(rate - baseline) > 0.05 else 0)


def _pick_variant(variant: str, user_hint: str | None) -> str:
    if variant in ("champion", "challenger"):
        return variant
    # A/B split deterministic on transaction_id hash → 90/10.
    # crc32, not hash(): str hash() is salted per-process, breaking determinism across workers.
    if user_hint:
        return "challenger" if (zlib.crc32(user_hint.encode()) % 10 == 0) else "champion"
    return "champion" if random.random() > 0.1 else "challenger"


def _to_frame(items: list[Transaction]) -> pd.DataFrame:
    return pd.DataFrame([t.model_dump() for t in items])


@app.on_event("startup")
def _startup():
    registry.load()


@app.get("/health")
def health():
    return {"status": "ok", "models": registry.summary()}


@app.get("/models")
def models():
    return registry.summary()


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction, request: Request, variant: Literal["champion", "challenger", "auto"] = "auto"):
    variant_name = _pick_variant(variant, tx.transaction_id)
    bundle = registry.get(variant_name)
    if bundle is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    df = _to_frame([tx])
    proba = float(bundle.predict_proba(df)[0])
    label = int(proba >= 0.5)
    latency_ms = (time.perf_counter() - t0) * 1000

    REQUESTS.labels(endpoint="/predict", model=bundle.name).inc()
    LATENCY.labels(endpoint="/predict", model=bundle.name).observe(latency_ms / 1000)
    PRED_COUNT.labels(model=bundle.name, label=str(label)).inc()
    _record_rate(bundle.name, label)

    return PredictionResponse(
        transaction_id=tx.transaction_id,
        fraud_probability=proba,
        is_fraud=bool(label),
        model_version=bundle.version,
        model_name=bundle.name,
        latency_ms=latency_ms,
    )


@app.post("/predict/bulk")
def predict_bulk(payload: BulkRequest):
    if not payload.transactions:
        raise HTTPException(400, "Empty batch")
    variant_name = _pick_variant(payload.variant, None)
    bundle = registry.get(variant_name)
    if bundle is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    df = _to_frame(payload.transactions)
    probas = bundle.predict_proba(df)
    labels = (probas >= 0.5).astype(int)
    latency_ms = (time.perf_counter() - t0) * 1000

    REQUESTS.labels(endpoint="/predict/bulk", model=bundle.name).inc()
    LATENCY.labels(endpoint="/predict/bulk", model=bundle.name).observe(latency_ms / 1000)
    for lbl in labels:
        PRED_COUNT.labels(model=bundle.name, label=str(int(lbl))).inc()
        _record_rate(bundle.name, int(lbl))

    return {
        "model_name": bundle.name,
        "model_version": bundle.version,
        "count": len(probas),
        "latency_ms": latency_ms,
        "predictions": [
            {
                "transaction_id": t.transaction_id,
                "fraud_probability": float(p),
                "is_fraud": bool(l),
            }
            for t, p, l in zip(payload.transactions, probas, labels)
        ],
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
