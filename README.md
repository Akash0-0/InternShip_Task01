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
