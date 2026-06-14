"""Synthetic transaction dataset generator with realistic fraud patterns."""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

MERCHANT_CATEGORIES = [
    "grocery", "gas", "restaurant", "online_retail", "electronics",
    "travel", "entertainment", "healthcare", "utilities", "luxury",
]

DEVICE_FINGERPRINTS = [f"dev_{i:05d}" for i in range(5000)]


def generate_transactions(n_rows: int = 100_000, fraud_rate: float = 0.018) -> pd.DataFrame:
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    base_time = datetime(2026, 1, 1)

    legit = pd.DataFrame({
        "amount": np.abs(RNG.lognormal(mean=3.2, sigma=1.1, size=n_legit)),
        "merchant_category": RNG.choice(MERCHANT_CATEGORIES, size=n_legit,
                                        p=[0.18, 0.12, 0.15, 0.20, 0.08, 0.05, 0.08, 0.06, 0.06, 0.02]),
        "time_of_day": np.clip(RNG.normal(14, 5, n_legit), 0, 23.99),
        "user_velocity": np.abs(RNG.normal(2.5, 1.5, n_legit)),
        "geo_distance": np.abs(RNG.exponential(8, n_legit)),
        "device_fingerprint": RNG.choice(DEVICE_FINGERPRINTS, size=n_legit),
        "is_fraud": 0,
    })

    fraud = pd.DataFrame({
        "amount": np.abs(RNG.lognormal(mean=5.5, sigma=1.4, size=n_fraud)),
        "merchant_category": RNG.choice(MERCHANT_CATEGORIES, size=n_fraud,
                                        p=[0.05, 0.03, 0.05, 0.30, 0.20, 0.15, 0.05, 0.02, 0.02, 0.13]),
        "time_of_day": np.clip(RNG.normal(2.5, 3.5, n_fraud) % 24, 0, 23.99),
        "user_velocity": np.abs(RNG.normal(12, 6, n_fraud)),
        "geo_distance": np.abs(RNG.exponential(150, n_fraud)),
        "device_fingerprint": RNG.choice(DEVICE_FINGERPRINTS[:500], size=n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    timestamps = [base_time + timedelta(seconds=int(s)) for s in RNG.integers(0, 60 * 60 * 24 * 90, len(df))]
    df["timestamp"] = sorted(timestamps)
    df["transaction_id"] = [f"tx_{i:08d}" for i in range(len(df))]
    df["user_id"] = [f"u_{i:06d}" for i in RNG.integers(0, 20000, len(df))]

    return df[[
        "transaction_id", "user_id", "timestamp", "amount", "merchant_category",
        "time_of_day", "user_velocity", "geo_distance", "device_fingerprint", "is_fraud",
    ]]


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_transactions()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows ({df.is_fraud.mean():.2%} fraud) -> {out}")
