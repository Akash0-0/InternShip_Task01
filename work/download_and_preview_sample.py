import os
from huggingface_hub import hf_hub_download
import pandas as pd

def main():
    token = os.environ.get('HUGGINGFACE_TOKEN')
    try:
        path = hf_hub_download(repo_id='FlyRank/internship-warehouse', filename='fact_content_daily_performance_sample.parquet', repo_type='dataset', token=token)
        print('Downloaded to', path)
        df = pd.read_parquet(path)
        print('shape:', df.shape)
        print(df.head(5).to_string())
        print('\nColumns:', list(df.columns))
        print('\nSample stats:')
        print(df.describe(include='all').transpose())
    except Exception as e:
        print('ERROR', e)

if __name__ == '__main__':
    main()
