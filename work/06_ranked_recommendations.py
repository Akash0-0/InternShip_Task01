import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def load_best_params(path='outputs/tuning_summary.json'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            j = json.load(f)
            return j.get('best_params', None)
    return None

def reason_code(row):
    reasons = []
    # ai referral present
    if row.get('ai_sum', 0) > 0:
        reasons.append('ai_referral')
    # visible but low CTR
    if row.get('gsc_impressions', 0) >= 100 and row.get('ctr', 0) < 0.01:
        reasons.append('visible_low_ctr')
    # high position but low volume
    if row.get('inv_pos', 0) > 0.5 and row.get('gsc_impressions', 0) < 50:
        reasons.append('high_pos_low_volume')
    # low engagement
    if row.get('scroll_events', 0) < 1 and row.get('ga4_engaged_sessions', 0) < 1:
        reasons.append('low_engagement')
    if not reasons:
        reasons = ['monitor']
    return '|'.join(reasons)

def run(features_path='work/features_sample.parquet', output_dir='outputs', top_n=200):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(features_path)
    print('loaded', len(df), 'rows')

    # keep rows with impressions
    df = df.copy()
    df['gsc_impressions'] = df.get('gsc_impressions', 0).fillna(0)
    df = df[df['gsc_impressions'] >= 1]

    feature_cols = ['log_impr','inv_pos','ga4_pageviews','ga4_sessions','ga4_engaged_sessions','ga4_total_engagement_sec','sessions_organic','scroll_events','ai_sum']
    # train model on entire available set using best params
    params = load_best_params()
    if params is None:
        print('No tuning summary found; using default RF params')
        params = {'n_estimators':100, 'max_depth':10, 'min_samples_split':2, 'max_features':'sqrt'}

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6,1e6)
    y = df['label_clicked']
    mdl = RandomForestClassifier(random_state=42, n_jobs=-1, **{k:v for k,v in params.items() if k in RandomForestClassifier().get_params()})
    mdl.fit(X, y)

    # score latest month if available, else whole
    if 'month' in df.columns:
        latest = df['month'].max()
        target_df = df[df['month'] == latest].copy()
        if len(target_df) == 0:
            target_df = df.copy()
    else:
        target_df = df.copy()

    X_target = target_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6,1e6)
    target_df['model_score'] = mdl.predict_proba(X_target)[:,1]
    target_df['baseline_score'] = target_df['log_impr']

    # compute reason codes
    target_df['reason'] = target_df.apply(reason_code, axis=1)

    # select top N by model_score
    out = target_df.sort_values('model_score', ascending=False).head(top_n)
    out_cols = ['report_date','client_hash_id','content_hash_id','month','gsc_impressions','ctr','model_score','baseline_score','reason']
    for c in out_cols:
        if c not in out.columns:
            out[c] = ''

    out_path = os.path.join(output_dir, f'recommendation_queue_top{top_n}.csv')
    out[out_cols].to_csv(out_path, index=False)
    print('wrote', out_path)

if __name__ == '__main__':
    run()
