"""Train XGBoost, LightGBM, and Isolation Forest ensemble with Optuna tuning + MLflow tracking."""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import xgboost as xgb

from features import build_preprocessor, split_features

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA)
    X, y = split_features(df)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def tune_xgb(X_train, y_train, X_val, y_val, n_trials: int = 20):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lgb(X_train, y_train, X_val, y_val, n_trials: int = 20):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def evaluate(name, y_true, y_proba):
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred).tolist()
    print(f"  {name:18s} ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}  CM={cm}")
    return {"roc_auc": auc, "pr_auc": ap, "confusion_matrix": cm}


def main(n_trials: int = 15):
    mlflow.set_tracking_uri(f"file://{ROOT}/mlruns")
    mlflow.set_experiment("fraud-detection")

    X_train_raw, X_val_raw, y_train, y_val = load_data()
    pre = build_preprocessor().fit(X_train_raw)
    X_train = pre.transform(X_train_raw)
    X_val = pre.transform(X_val_raw)
    print(f"Train shape: {X_train.shape} | val: {X_val.shape} | fraud rate: {y_train.mean():.3%}")

    metrics = {}

    with mlflow.start_run(run_name="xgboost"):
        print("Tuning XGBoost...")
        best_xgb = tune_xgb(X_train, y_train, X_val, y_val, n_trials)
        mdl = xgb.XGBClassifier(**best_xgb, eval_metric="aucpr", tree_method="hist", verbosity=0)
        mdl.fit(X_train, y_train)
        proba = mdl.predict_proba(X_val)[:, 1]
        metrics["xgboost"] = evaluate("xgboost", y_val, proba)
        mlflow.log_params(best_xgb)
        mlflow.log_metrics(metrics["xgboost"] | {})
        joblib.dump(mdl, MODELS / "xgboost.joblib")

    with mlflow.start_run(run_name="lightgbm"):
        print("Tuning LightGBM...")
        best_lgb = tune_lgb(X_train, y_train, X_val, y_val, n_trials)
        mdl = lgb.LGBMClassifier(**best_lgb, verbosity=-1)
        mdl.fit(X_train, y_train)
        proba = mdl.predict_proba(X_val)[:, 1]
        metrics["lightgbm"] = evaluate("lightgbm", y_val, proba)
        mlflow.log_params(best_lgb)
        mlflow.log_metrics(metrics["lightgbm"] | {})
        joblib.dump(mdl, MODELS / "lightgbm.joblib")

    with mlflow.start_run(run_name="isolation_forest"):
        print("Fitting Isolation Forest...")
        iso = IsolationForest(n_estimators=200, contamination=float(y_train.mean()), random_state=42, n_jobs=-1)
        iso.fit(X_train)
        proba = -iso.score_samples(X_val)
        proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
        metrics["isolation_forest"] = evaluate("isolation_forest", y_val, proba)
        mlflow.log_metrics({k: v for k, v in metrics["isolation_forest"].items() if isinstance(v, float)})
        joblib.dump(iso, MODELS / "isolation_forest.joblib")

    print("Training ensemble (weighted avg)...")
    xgb_m = joblib.load(MODELS / "xgboost.joblib")
    lgb_m = joblib.load(MODELS / "lightgbm.joblib")
    iso_m = joblib.load(MODELS / "isolation_forest.joblib")
    iso_score = -iso_m.score_samples(X_val)
    iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9)
    ensemble = 0.45 * xgb_m.predict_proba(X_val)[:, 1] + 0.45 * lgb_m.predict_proba(X_val)[:, 1] + 0.10 * iso_score
    metrics["ensemble"] = evaluate("ensemble", y_val, ensemble)

    joblib.dump(pre, MODELS / "preprocessor.joblib")
    with open(MODELS / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(MODELS / "version.json", "w") as fh:
        json.dump({"version": "v1.0.0", "champion": "ensemble", "challenger": "xgboost"}, fh, indent=2)

    print(f"Saved models -> {MODELS}")


if __name__ == "__main__":
    main()
