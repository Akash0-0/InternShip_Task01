import os
import requests

def main():
    token = os.environ.get('HUGGINGFACE_TOKEN')
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    url = 'https://huggingface.co/api/repos?search=flyrank'
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        print('FOUND', len(data), 'repos')
        for item in data[:50]:
            print(item.get('id'))
    except Exception as e:
        print('ERROR', str(e))

if __name__ == '__main__':
    main()
