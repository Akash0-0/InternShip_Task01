from huggingface_hub import HfApi
import os

def main():
    token = os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        print('NO_TOKEN')
        return
    api = HfApi()
    try:
        info = api.whoami(token=token)
        # Print minimal info
        print('WHOAMI_OK')
        print('user:', info.get('name') or info.get('username'))
    except Exception as e:
        print('WHOAMI_FAIL')
        print(str(e))

if __name__ == '__main__':
    main()
