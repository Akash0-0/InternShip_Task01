from huggingface_hub import HfApi
import os

def main():
    token = os.environ.get('HUGGINGFACE_TOKEN')
    api = HfApi()
    try:
        info = api.dataset_info('FlyRank/internship-warehouse', token=token)
        print('ID:', info.id)
        print('Files:')
        for i, f in enumerate(info.siblings[:200]):
            print('--- file', i)
            # print available attributes
            for k, v in f.__dict__.items():
                print(f"{k}: {v}")
    except Exception as e:
        print('ERROR', e)

if __name__ == '__main__':
    main()
