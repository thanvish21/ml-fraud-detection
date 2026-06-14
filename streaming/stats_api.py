"""Tiny aggregation API for the dashboard (reads from PostgreSQL)."""
from __future__ import annotations
import os
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PG_DSN = os.getenv("PG_DSN", "host=postgres dbname=fraud user=fraud password=fraud port=5432")
app = FastAPI(title="Fraud Stats API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _conn():
    return psycopg2.connect(PG_DSN, cursor_factory=psycopg2.extras.RealDictCursor)


@app.get("/stats/live")
def live():
    cutoff = datetime.utcnow() - timedelta(minutes=5)
    with _conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud,
                   AVG(latency_ms) AS avg_latency
            FROM predictions WHERE ts >= %s
        """, (cutoff,))
        row = cur.fetchone()
    total = row["total"] or 0
    fraud = row["fraud"] or 0
    return {
        "window_minutes": 5,
        "total_transactions": total,
        "fraud_transactions": fraud,
        "fraud_rate": (fraud / total) if total else 0,
        "avg_latency_ms": float(row["avg_latency"] or 0),
    }


@app.get("/stats/timeline")
def timeline(minutes: int = 30):
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    with _conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT date_trunc('minute', ts) AS bucket,
                   COUNT(*) AS total,
                   SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud
            FROM predictions WHERE ts >= %s
            GROUP BY 1 ORDER BY 1
        """, (cutoff,))
        rows = cur.fetchall()
    return [{"bucket": r["bucket"].isoformat(), "total": r["total"], "fraud": r["fraud"] or 0} for r in rows]


@app.get("/stats/patterns")
def patterns():
    cutoff = datetime.utcnow() - timedelta(hours=1)
    with _conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT merchant_category,
                   COUNT(*) AS total,
                   SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud,
                   AVG(amount) AS avg_amount
            FROM predictions WHERE ts >= %s
            GROUP BY 1 ORDER BY fraud DESC NULLS LAST LIMIT 10
        """, (cutoff,))
        rows = cur.fetchall()
    return [{
        "merchant_category": r["merchant_category"],
        "total": r["total"],
        "fraud": r["fraud"] or 0,
        "fraud_rate": ((r["fraud"] or 0) / r["total"]) if r["total"] else 0,
        "avg_amount": float(r["avg_amount"] or 0),
    } for r in rows]


@app.get("/stats/model")
def model_perf():
    cutoff = datetime.utcnow() - timedelta(hours=1)
    with _conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT model_name, model_version,
                   COUNT(*) AS total,
                   AVG(latency_ms) AS avg_latency,
                   AVG(fraud_probability) AS avg_proba
            FROM predictions WHERE ts >= %s
            GROUP BY 1, 2 ORDER BY total DESC
        """, (cutoff,))
        rows = cur.fetchall()
    return [{
        "model_name": r["model_name"], "model_version": r["model_version"],
        "total": r["total"], "avg_latency_ms": float(r["avg_latency"] or 0),
        "avg_proba": float(r["avg_proba"] or 0),
    } for r in rows]


@app.get("/health")
def health():
    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}
