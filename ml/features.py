"""Feature engineering pipeline for fraud detection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ["amount", "time_of_day", "user_velocity", "geo_distance"]
CATEGORICAL_FEATURES = ["merchant_category"]


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Adds amount*velocity, geo*velocity, and log_amount interaction features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=NUMERIC_FEATURES).copy()
        df["log_amount"] = np.log1p(df["amount"])
        df["amount_velocity"] = df["amount"] * df["user_velocity"]
        df["geo_velocity"] = df["geo_distance"] * df["user_velocity"]
        df["night_tx"] = ((df["time_of_day"] < 6) | (df["time_of_day"] > 22)).astype(float)
        return df.values


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("interact", InteractionFeatures()),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def split_features(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_fraud"].astype(int)
    return X, y
