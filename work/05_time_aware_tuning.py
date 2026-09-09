import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def precision_at_k(y_true, scores, k):
    order = np.argsort(-scores)
    topk = order[:k]
    return y_true.iloc[topk].sum() / float(k)


def run(features_path='work/features_sample.parquet', output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(features_path)
    print('loaded rows', len(df))

    # require month col for time-aware split
    if 'month' in df.columns:
        months = sorted(df['month'].dropna().unique())
    else:
        months = []
    print('months found:', months[:10])

    if len(months) > 1:
        # use earliest 70% months as train, latest 30% as test
        split_idx = int(len(months) * 0.7)
        train_months = months[:split_idx]
        test_months = months[split_idx:]
        train_df = df[df['month'].isin(train_months)]
        test_df = df[df['month'].isin(test_months)]
        if len(train_df) == 0 or len(test_df) == 0:
            print('time split produced empty set, falling back to random split')
            train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_clicked'], random_state=42)
    else:
        print('only one month present; using stratified random split')
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_clicked'], random_state=42)

    print('train rows', len(train_df), 'test rows', len(test_df))

    feature_cols = ['log_impr','inv_pos','ga4_pageviews','ga4_sessions','ga4_engaged_sessions','ga4_total_engagement_sec','sessions_organic','scroll_events','ai_sum']
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6,1e6)
    y_train = train_df['label_clicked']
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6,1e6)
    y_test = test_df['label_clicked']

    base_scores = X_test['log_impr']

    # random forest with randomized search
    param_dist = {
        'n_estimators': [50,100,200],
        'max_depth': [5,10,20,None],
        'min_samples_split': [2,5,10],
        'max_features': ['sqrt','log2', None]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rsearch = RandomizedSearchCV(rf, param_dist, n_iter=16, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1, verbose=1)
    rsearch.fit(X_train, y_train)
    print('best params', rsearch.best_params_)

    best = rsearch.best_estimator_
    proba = best.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    base_auc = roc_auc_score(y_test, base_scores)
    print('Best model AUC:', auc)
    print('Baseline AUC:', base_auc)

    # precision@k
    ks = list(range(10,201,10))
    model_p = [precision_at_k(y_test.reset_index(drop=True), proba, k) for k in ks]
    base_p = [precision_at_k(y_test.reset_index(drop=True), base_scores.values, k) for k in ks]

    plt.figure(figsize=(6,4))
    plt.plot(ks, model_p, label='Model')
    plt.plot(ks, base_p, label='Baseline')
    plt.xlabel('k')
    plt.ylabel('Precision@k')
    plt.legend()
    out_plot = os.path.join(output_dir, 'precision_at_k_tuned.png')
    plt.savefig(out_plot, dpi=150)
    print('wrote', out_plot)

    # save best params and results
    res = {
        'best_params': rsearch.best_params_,
        'model_auc': float(auc),
        'baseline_auc': float(base_auc)
    }
    with open(os.path.join(output_dir, 'tuning_summary.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print('wrote tuning_summary.json')

if __name__ == '__main__':
    run()
