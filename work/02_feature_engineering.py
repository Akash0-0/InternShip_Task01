import os
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

def build_features(parquet_filename='fact_content_daily_performance_sample.parquet', sample_frac=None, out_path='work/features_sample.parquet'):
    token = os.environ.get('HUGGINGFACE_TOKEN')
    path = hf_hub_download(repo_id='FlyRank/internship-warehouse', filename=parquet_filename, repo_type='dataset', token=token)
    print('Loading (via duckdb) from', path)
    import duckdb

    con = duckdb.connect(database=':memory:')
    # sample via SQL to avoid loading entire parquet into memory
    if sample_frac is not None and 0 < sample_frac < 1.0:
        sql = f"SELECT * FROM parquet_scan('{path}') WHERE random() < {sample_frac}"
    else:
        sql = f"SELECT * FROM parquet_scan('{path}')"
    df = con.execute(sql).df()
    print('sampled rows', len(df))

    # Basic features
    df['gsc_impressions'] = df['gsc_impressions'].fillna(0).astype(float)
    df['gsc_clicks'] = df['gsc_clicks'].fillna(0).astype(float)
    df['gsc_avg_position'] = df['gsc_avg_position'].fillna(999.0).astype(float)
    df['ctr'] = df['gsc_clicks'] / df['gsc_impressions'].replace(0, np.nan)
    df['ctr'] = df['ctr'].fillna(0.0)
    df['label_clicked'] = (df['gsc_clicks'] > 0).astype(int)

    # feature engineering
    df['log_impr'] = np.log1p(df['gsc_impressions'])
    df['inv_pos'] = 1.0 / df['gsc_avg_position'].replace(0, np.nan).replace(np.inf, 0).fillna(0)
    ga4_cols = ['ga4_pageviews','ga4_sessions','ga4_users','ga4_engaged_sessions','ga4_total_engagement_sec']
    for c in ga4_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(float)
        else:
            df[c] = 0.0
    ai_cols = ['ai_chatgpt','ai_perplexity','ai_gemini','ai_copilot','ai_claude','ai_meta','ai_other']
    for c in ai_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(float)
        else:
            df[c] = 0.0
    df['ai_sum'] = df[ai_cols].sum(axis=1)
    df['scroll_events'] = df.get('scroll_events', 0).fillna(0).astype(float)

    # select features and ids
    features = ['report_date','client_hash_id','content_hash_id','month','gsc_impressions','gsc_clicks','gsc_avg_position','ctr','label_clicked','log_impr','inv_pos','ga4_pageviews','ga4_sessions','ga4_engaged_sessions','ga4_total_engagement_sec','sessions_organic','scroll_events','ai_sum']
    features = [c for c in features if c in df.columns or c in ['label_clicked','log_impr','inv_pos','ai_sum']]
    feat_df = df[features].copy()

    if sample_frac is not None and 0 < sample_frac < 1.0:
        feat_df = feat_df.sample(frac=sample_frac, random_state=42)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feat_df.to_parquet(out_path, index=False)
    print('wrote', out_path, 'rows', len(feat_df))

if __name__ == '__main__':
    # default: sample 10% to keep processing quick in this environment
    build_features(sample_frac=0.1)
