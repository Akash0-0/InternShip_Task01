# Baseline vs Model — Quick Results

Commands run:
- `python work/02_feature_engineering.py` (sampled 10% via DuckDB)
- `python work/03_baseline_and_model.py` (trained RandomForest)

Key metrics (sample run):
- Model AUC: 0.8927
- Baseline (log_impr) AUC: 0.8608

Top feature importances (from RandomForest):
- `inv_pos` (inverse avg position)
- `log_impr` (log impressions)
- `sessions_organic`

Outputs written:
- `outputs/precision_at_k.png`
- `outputs/feature_importances.csv`

Next steps:
- Add notebook that documents these steps and visualizes results (`work/02_modeling.ipynb` or capstone notebook).
- Expand features, tune hyperparameters, and compute robust time-aware validation on full data.
