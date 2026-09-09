import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score
import matplotlib.pyplot as plt


def precision_at_k(y_true, scores, k):
    # compute precision at k for ranking by scores
    order = np.argsort(-scores)
    topk = order[:k]
    return y_true.iloc[topk].sum() / float(k)


def run(out_features='work/features_sample.parquet', output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(out_features)
    print('loaded features rows', len(df))

    # drop rows with zero impressions to reduce noise (optional)
    df = df[df['gsc_impressions'] >= 1]
    print('after filter impressions>=1 rows', len(df))

    # features and label
    X = df[['log_impr','inv_pos','ga4_pageviews','ga4_sessions','ga4_engaged_sessions','ga4_total_engagement_sec','sessions_organic','scroll_events','ai_sum']]
    # sanitize
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # clip extreme values to avoid numeric issues
    X = X.clip(lower=-1e6, upper=1e6)
    y = df['label_clicked']

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # baseline: rank by impressions
    baseline_scores = X_test['log_impr']
    # model
    mdl = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    mdl.fit(X_train, y_train)
    proba = mdl.predict_proba(X_test)[:,1]

    # metrics
    auc = roc_auc_score(y_test, proba)
    baseline_auc = roc_auc_score(y_test, baseline_scores)

    print('Model AUC:', auc)
    print('Baseline (log_impr) AUC:', baseline_auc)

    # precision@k curve
    ks = list(range(10, 201, 10))
    model_p = [precision_at_k(y_test.reset_index(drop=True), proba, k) for k in ks]
    base_p = [precision_at_k(y_test.reset_index(drop=True), baseline_scores.values, k) for k in ks]

    plt.figure(figsize=(6,4))
    plt.plot(ks, model_p, label='Model')
    plt.plot(ks, base_p, label='Baseline')
    plt.xlabel('k')
    plt.ylabel('Precision@k')
    plt.legend()
    out_plot = os.path.join(output_dir, 'precision_at_k.png')
    plt.savefig(out_plot, dpi=150)
    print('wrote', out_plot)

    # feature importances
    fi = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi.to_csv(os.path.join(output_dir, 'feature_importances.csv'))
    print('wrote feature_importances.csv')

if __name__ == '__main__':
    run()
