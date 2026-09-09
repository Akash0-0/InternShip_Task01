# Setup for Capstone: Ranking Signal Analysis

1. Create and switch to branch locally:

```powershell
cd path\to\YT_VIDEO_Translator
git checkout -b capstone-ranking-signal
git add work/ submission/
git commit -m "capstone: scaffold ranking signal lane"
```

2. Install dependencies (recommend using a venv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Access FlyRank dataset from Hugging Face (set token in env):

```powershell
$env:HUGGINGFACE_TOKEN = 'YOUR_TOKEN_HERE'  # or set in system env vars
# Run the example cells in work/01_eda.ipynb to connect via DuckDB using the token
```

4. Replace `PAPER_URL_HERE` in `submission/paper_url.txt` with your deployed GitHub Pages URL when ready.

Notes:
- Do NOT commit or publish private tokens. Use env vars when running notebooks.
- If you want me to attempt the data download here, reply OK and I'll use the token you provided to run the example EDA locally.
