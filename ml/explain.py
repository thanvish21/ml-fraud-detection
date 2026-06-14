"""SHAP feature-importance plots for the trained models."""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from features import build_preprocessor, split_features

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT / "data" / "transactions.csv").sample(5000, random_state=42)
    X, _ = split_features(df)
    pre = joblib.load(ROOT / "models" / "preprocessor.joblib")
    Xp = pre.transform(X)
    feature_names = (
        [f"num_{c}" for c in ["amount", "time_of_day", "user_velocity", "geo_distance",
                              "log_amount", "amount_velocity", "geo_velocity", "night_tx"]]
        + list(pre.named_transformers_["cat"].get_feature_names_out(["merchant_category"]))
    )

    for name in ("xgboost", "lightgbm"):
        mdl = joblib.load(ROOT / "models" / f"{name}.joblib")
        expl = shap.TreeExplainer(mdl)
        sv = expl.shap_values(Xp)
        if isinstance(sv, list):
            sv = sv[1]
        plt.figure()
        shap.summary_plot(sv, Xp, feature_names=feature_names, show=False, max_display=12)
        out = ROOT / "models" / f"shap_{name}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
