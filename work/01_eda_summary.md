# EDA Summary — Ranking Signal Analysis

Dataset: FlyRank/internship-warehouse (sample: fact_content_daily_performance_sample.parquet)
Rows sampled: 11,694,072; Columns: 31

Key columns: report_date, client_hash_id, content_hash_id, gsc_impressions, gsc_clicks, gsc_avg_position, ga4_pageviews, sessions_organic, sessions_ai, ai_* indicators, scroll_events, month

Quick observations:
- Many zero-valued rows for impressions/clicks; skewed distributions (long tail).
- `gsc_avg_position` has many NaNs where `gsc_impressions`==0 (expected).
- AI referral columns present but sparse.
- `month` field present; good for time-aware splits.

Next steps:
- Define target: CTR (gsc_clicks / gsc_impressions) or action score used by starter pipeline.
- Compute baseline rule (e.g., score = impressions * ctr_estimate or use provided `02_baseline_score.py`).
- Build feature set: recent impressions, avg position, GA4 engagement, scroll_events, ai signals.
- Validate with time-aware split (train on earlier months, test on later months) and check leakage.

Files created during this step:
- `work/download_and_preview_sample.py` (script used)
- `work/01_eda.ipynb` (starter notebook)
- `work/01_eda_summary.md` (this summary)

If you want, I'll proceed to implement baseline score and a simple model (RandomForest) and produce evaluation charts next. Reply "go on" and I'll continue. Or I'll proceed automatically in 5s.