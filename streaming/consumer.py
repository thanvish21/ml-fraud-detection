"""Kafka consumer: runs inference and writes results to PostgreSQL. Failed records go to a DLQ topic."""
from __future__ import annotations
import json
import os
import time
import traceback
from datetime import datetime

import psycopg2
import psycopg2.extras
import requests
from confluent_kafka import Consumer, Producer, KafkaException

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
DLQ_TOPIC = os.getenv("DLQ_TOPIC", "transactions.dlq")
GROUP = os.getenv("KAFKA_GROUP", "fraud-consumer")
API_URL = os.getenv("API_URL", "http://api:8000/predict")
PG_DSN = os.getenv("PG_DSN", "host=postgres dbname=fraud user=fraud password=fraud port=5432")

INSERT_SQL = """
INSERT INTO predictions (transaction_id, ts, amount, merchant_category, fraud_probability,
                         is_fraud, model_name, model_version, latency_ms)
VALUES %s
ON CONFLICT (transaction_id) DO NOTHING
"""

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    transaction_id      TEXT PRIMARY KEY,
    ts                  TIMESTAMP NOT NULL DEFAULT NOW(),
    amount              DOUBLE PRECISION,
    merchant_category   TEXT,
    fraud_probability   DOUBLE PRECISION,
    is_fraud            BOOLEAN,
    model_name          TEXT,
    model_version       TEXT,
    latency_ms          DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS predictions_ts_idx ON predictions (ts DESC);
CREATE INDEX IF NOT EXISTS predictions_fraud_idx ON predictions (is_fraud) WHERE is_fraud;
"""


def connect_pg(retries: int = 30):
    for i in range(retries):
        try:
            conn = psycopg2.connect(PG_DSN)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(DDL)
            return conn
        except Exception as exc:
            print(f"  postgres retry {i+1}: {exc}")
            time.sleep(2)
    raise RuntimeError("postgres unavailable")


def score(tx: dict) -> dict:
    r = requests.post(API_URL, json={
        "transaction_id": tx["transaction_id"],
        "amount": tx["amount"],
        "merchant_category": tx["merchant_category"],
        "time_of_day": tx["time_of_day"],
        "user_velocity": tx["user_velocity"],
        "geo_distance": tx["geo_distance"],
        "device_fingerprint": tx.get("device_fingerprint"),
    }, timeout=2.0)
    r.raise_for_status()
    return r.json()


def main():
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP,
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([TOPIC])
    dlq = Producer({"bootstrap.servers": BOOTSTRAP})
    conn = connect_pg()
    print(f"Consuming {TOPIC} → {API_URL} → postgres. DLQ: {DLQ_TOPIC}")

    batch: list[tuple] = []
    last_flush = time.time()

    while True:
        msg = consumer.poll(0.5)
        if msg is None:
            if batch and time.time() - last_flush > 1.0:
                _flush(conn, batch)
                batch.clear()
                last_flush = time.time()
            continue
        if msg.error():
            print(f"  kafka err: {msg.error()}")
            continue
        try:
            tx = json.loads(msg.value())
            pred = score(tx)
            batch.append((
                tx["transaction_id"], datetime.utcnow(), tx["amount"], tx["merchant_category"],
                pred["fraud_probability"], pred["is_fraud"], pred["model_name"], pred["model_version"], pred["latency_ms"],
            ))
            if len(batch) >= 100:
                _flush(conn, batch)
                batch.clear()
                last_flush = time.time()
        except Exception as exc:
            print(f"  dead-letter: {exc}")
            traceback.print_exc()
            dlq.produce(DLQ_TOPIC, key=msg.key(), value=json.dumps({
                "raw": msg.value().decode(errors="replace"),
                "error": str(exc),
            }).encode())
            dlq.poll(0)


def _flush(conn, batch):
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, INSERT_SQL, batch)
        print(f"  inserted {len(batch)} predictions")
    except Exception as exc:
        print(f"  pg insert failed: {exc}")


if __name__ == "__main__":
    main()
