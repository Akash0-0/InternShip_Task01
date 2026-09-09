from huggingface_hub import HfApi
import os

def main():
    token = os.environ.get('HUGGINGFACE_TOKEN')
    api = HfApi()
    try:
        repos = api.list_repos(actor='flyrank', token=token)
        print('REPO_COUNT', len(repos))
        for r in repos[:50]:
            print(r.id)
    except Exception as e:
        print('ERROR', str(e))

if __name__ == '__main__':
    main()
