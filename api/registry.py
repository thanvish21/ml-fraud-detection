"""Versioned model registry with champion/challenger A/B support."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    name: str
    version: str
    preprocessor: object
    model: object

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(df)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        # IsolationForest fallback
        scores = -self.model.score_samples(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._bundles: dict[str, ModelBundle] = {}
        self._version = "v0"

    def load(self):
        meta_path = self.models_dir / "version.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
            "version": "v1.0.0", "champion": "lightgbm", "challenger": "xgboost",
        }
        self._version = meta.get("version", "v1.0.0")
        pre_path = self.models_dir / "preprocessor.joblib"
        if not pre_path.exists():
            # Degraded mode: serve a constant low-probability response
            self._bundles = {}
            return
        pre = joblib.load(pre_path)
        champion_name = meta.get("champion", "lightgbm")
        challenger_name = meta.get("challenger", "xgboost")
        for slot, name in [("champion", champion_name), ("challenger", challenger_name)]:
            mp = self.models_dir / f"{name}.joblib"
            if mp.exists():
                self._bundles[slot] = ModelBundle(name=name, version=self._version, preprocessor=pre, model=joblib.load(mp))

    def get(self, slot: str) -> ModelBundle | None:
        return self._bundles.get(slot) or next(iter(self._bundles.values()), None)

    def summary(self):
        return {slot: {"name": b.name, "version": b.version} for slot, b in self._bundles.items()}
