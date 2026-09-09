# Auto Video Dubber

Auto Video Dubber downloads a YouTube video, transcribes its audio with Whisper, translates Indic-language speech into English with AI4Bharat IndicTrans2, generates English speech with Edge TTS, and remuxes the dubbed audio onto the original video.

## Pipeline

1. Download a YouTube video with `yt-dlp`.
2. Transcribe speech and detect the source language with `faster-whisper`.
3. Translate each timestamped segment with IndicTrans2.
4. Synthesize each translated segment with Microsoft Edge TTS.
5. Time-align the clips and mux the resulting English audio with the original video.

## Requirements

- Python 3.10 or newer
- FFmpeg and FFprobe available on `PATH`
- A Hugging Face token with access to `ai4bharat/indictrans2-indic-en-dist-200M`
- A CUDA-capable PyTorch installation is recommended for practical processing times

Install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script activation, run the commands from an activated Command Prompt environment instead.

## Configuration

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token
USE_CUDA=auto
WHISPER_MODEL_SIZE=medium
CPU_THREADS=2
TRANSLATE_BATCH_SIZE=8
DUB_VOLUME_DB=6
```

`USE_CUDA` accepts `auto`, `true`, or `false`. The default `auto` value uses CUDA when it is available and otherwise falls back to CPU. Do not commit `.env` or put your token directly in source code.

## Usage

Make sure `ffmpeg` and `ffprobe` can be found from the terminal, activate the virtual environment, and run:

```powershell
python app.py
```

Enter a YouTube URL when prompted. Downloaded videos are stored in `videos/`, synthesized clips are stored in a video-specific `*_audio/` folder, and final videos are written to `dubbed_output/`.

## Current limitations

- Translation currently supports the Indic languages listed in `INDIC_TO_FLORES` in `app.py`.
- Non-Indic source languages are detected by Whisper but are not currently routed to a general translation model.
- Processing may require substantial disk space and GPU memory, especially for long videos.
- The generated dub replaces the source audio rather than mixing it with the original speech.

## Project files

- `app.py` contains the LangGraph workflow and audio/video processing stages.
- `tools.py` contains the IndicTrans2 batch translation helper.
- `requirements.txt` lists the Python dependencies.

## FlyRank Capstone — Ranking Signal Analysis

This repository also hosts a FlyRank ML Internship capstone: a reproducible Ranking Signal Analysis built on the FlyRank internship warehouse sample. The capstone artifacts live under the `work/` folder and a short research paper is published under `docs/index.html`.

Quick links:
- Capstone notebook: `work/05_capstone_notebook.ipynb`
- Feature & model scripts: `work/02_feature_engineering.py`, `work/03_baseline_and_model.py`, `work/05_time_aware_tuning.py`
- Recommendation queue: `work/06_ranked_recommendations.py` → `outputs/recommendation_queue_top200.csv`
- Deployed paper (GitHub Pages): see `submission/paper_url.txt`

How to publish the paper via GitHub Pages:

```bash
git add work docs outputs submission
git commit -m "capstone: add model, tuning, recommendations, and docs paper"
git push origin main
```

Then enable Pages in your GitHub repo: Settings → Pages → Source → Branch `main` and folder `/docs`, Save. The paper URL is in `submission/paper_url.txt`.

