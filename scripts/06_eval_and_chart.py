from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from ml_utils import PROCESSED_DIR, OUTPUT_DIR, CHART_DIR, MODEL_NUMERIC_FEATURES, MODEL_CATEGORICAL_FEATURES, precision_at_k, ensure_dirs


def build_matrix(df: pd.DataFrame):
    X_num = df[MODEL_NUMERIC_FEATURES].fillna(0)
    X_cat = pd.get_dummies(df[MODEL_CATEGORICAL_FEATURES].fillna("unknown"), drop_first=True)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    y = df["is_declining_label"].astype(int)
    return X, y


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(PROCESSED_DIR / "refresh_feature_vector.csv")
    X, y = build_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_path = OUTPUT_DIR / "refresh_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    model = joblib.load(model_path)

    prob_test = model.predict_proba(X_test)[:, 1]
    opportunity_scores = 1.0 - prob_test
    auc = float(roc_auc_score(y_test, prob_test))
    print(f"ROC AUC (decline prob): {auc:.4f}")

    # ROC chart
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve — decline probability")
    plt.legend()
    roc_path = CHART_DIR / "roc_decline_prob.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {roc_path}")

    # Precision@k
    ks = list(range(10, 501, 10))
    ps = [precision_at_k(y_test.tolist(), opportunity_scores.tolist(), k=k) for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, ps)
    plt.xlabel("k")
    plt.ylabel("Precision@k")
    plt.title("Precision@k — opportunity ranking")
    prec_path = CHART_DIR / "precision_at_k.png"
    plt.savefig(prec_path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {prec_path}")


if __name__ == "__main__":
    main()
