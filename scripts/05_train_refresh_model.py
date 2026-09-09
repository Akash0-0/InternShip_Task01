from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

from ml_utils import (
    PROCESSED_DIR,
    OUTPUT_DIR,
    ensure_dirs,
    MODEL_NUMERIC_FEATURES,
    MODEL_CATEGORICAL_FEATURES,
    precision_at_k,
    display_path,
)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed feature vector not found: {path}\nRun scripts/01_prepare_features.py first.")
    return pd.read_csv(path)


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X_num = df[MODEL_NUMERIC_FEATURES].fillna(0)
    X_cat = pd.get_dummies(df[MODEL_CATEGORICAL_FEATURES].fillna("unknown"), drop_first=True)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    y = df["is_declining_label"].astype(int)
    return X, y


def main() -> None:
    ensure_dirs()
    input_path = PROCESSED_DIR / "refresh_feature_vector.csv"
    df = load_data(input_path)

    X, y = build_matrix(df)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    prob_test = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, prob_test))
    # Score: higher means opportunity = 1 - prob_declining
    opportunity_scores = 1.0 - prob_test

    prec_at_100 = precision_at_k(y_test.tolist(), opportunity_scores.tolist(), k=100)

    # Save model and artifacts
    model_path = OUTPUT_DIR / "refresh_model.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "roc_auc_decline_prob": auc,
        "precision_at_100_opportunity": prec_at_100,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (OUTPUT_DIR / "refresh_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Produce top-ranked recommendations on the entire dataset
    full_prob = model.predict_proba(X)[:, 1]
    df_out = df.copy()
    df_out["decline_prob"] = full_prob
    df_out["opportunity_score"] = 1.0 - df_out["decline_prob"]

    # Simple reason code: high impressions or high score
    df_out["reason"] = df_out.apply(
        lambda r: "high_impressions" if r.get("impressions_90d", 0) >= 1000 else "score_based",
        axis=1,
    )

    top_path = OUTPUT_DIR / "top_ranked_refresh.csv"
    df_out.sort_values("opportunity_score", ascending=False).head(200).to_csv(top_path, index=False)

    print(f"Model saved to: {display_path(model_path)}")
    print(f"Metrics: {json.dumps(metrics)}")
    print(f"Top-ranked candidates written to: {display_path(top_path)}")


if __name__ == "__main__":
    main()
