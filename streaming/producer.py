"""Kafka producer that simulates a live transaction stream (~1000 tx/sec)."""
from __future__ import annotations
import json
import os
import random
import time
from datetime import datetime

import numpy as np
from confluent_kafka import Producer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
RATE = int(os.getenv("TX_PER_SEC", "1000"))

MERCHANT_CATEGORIES = [
    "grocery", "gas", "restaurant", "online_retail", "electronics",
    "travel", "entertainment", "healthcare", "utilities", "luxury",
]

rng = np.random.default_rng()


def synth_tx(idx: int) -> dict:
    fraud_like = random.random() < 0.04
    if fraud_like:
        amount = float(np.abs(rng.lognormal(5.0, 1.3)))
        velocity = float(np.abs(rng.normal(10, 5)))
        geo = float(np.abs(rng.exponential(120)))
        hour = float(rng.choice([1.0, 2.5, 3.5, 23.5]))
        category = random.choice(["online_retail", "electronics", "travel", "luxury"])
    else:
        amount = float(np.abs(rng.lognormal(3.2, 1.0)))
        velocity = float(np.abs(rng.normal(2.5, 1.2)))
        geo = float(np.abs(rng.exponential(8)))
        hour = float(np.clip(rng.normal(14, 5), 0, 23.99))
        category = random.choice(MERCHANT_CATEGORIES)
    return {
        "transaction_id": f"live_{idx:010d}",
        "amount": round(amount, 2),
        "merchant_category": category,
        "time_of_day": round(hour, 2),
        "user_velocity": round(velocity, 2),
        "geo_distance": round(geo, 2),
        "device_fingerprint": f"dev_{rng.integers(0, 5000):05d}",
        "ts": datetime.utcnow().isoformat(),
    }


def main():
    producer = Producer({"bootstrap.servers": BOOTSTRAP, "linger.ms": 5, "batch.num.messages": 200})
    interval = 1.0 / RATE
    idx = 0
    print(f"Producing ~{RATE} tx/s to {BOOTSTRAP} topic={TOPIC}")
    next_tick = time.perf_counter()
    while True:
        tx = synth_tx(idx)
        producer.produce(TOPIC, key=tx["transaction_id"], value=json.dumps(tx).encode())
        idx += 1
        if idx % 500 == 0:
            producer.poll(0)
            print(f"  sent {idx:,} (queue~{len(producer)})")
        next_tick += interval
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_tick = time.perf_counter()


if __name__ == "__main__":
    main()
