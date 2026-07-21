from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Any, Optional, TypedDict ,cast
from dotenv import load_dotenv
load_dotenv()
import gc
import os
import yt_dlp
import torch
from faster_whisper import WhisperModel
import asyncio
import subprocess
import edge_tts
from tools import translate_batch_indic

# from huggingface_hub import login
# login(os.getenv("HF_TOKEN"))

class SegmentDict(TypedDict):
    start: float
    end: float
    text: str


class WorkFlow(TypedDict):
    url: str
    video_path: str
    segments: list[SegmentDict]            # source-language transcript, with timestamps
    detected_lang: str
    translated_segments: list[SegmentDict] # English text, same timestamps
    tts_clips: list[tuple[float, str]]     # (original_start_time, wav_path)
    output_path: str


def Download_Video(state: WorkFlow):
    """Downloads the video from the URL and saves it to a local path.

    Named after yt-dlp's own video ID (guaranteed filesystem-safe) rather
    than the raw URL, since URLs contain characters like '?' and '/' that
    break on Windows and get misread as path separators.
    """
    url = state.get("url") if isinstance(state, dict) else None
    if not url:
        url = input("Enter the url here:")

    videos_dir = os.path.join(os.getcwd(), "videos")
    os.makedirs(videos_dir, exist_ok=True)

    print("Downloading video...")
    with yt_dlp.YoutubeDL({
        "outtmpl": os.path.join(videos_dir, "%(id)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
    }) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        base, _ = os.path.splitext(path)
        path = base + ".mp4"  # merge_output_format can change the extension

    print(f"Video downloaded to {path}")
    return {"video_path": path, "url": url}


def Transcribe_Video(state: WorkFlow):
    gpu_available = torch.cuda.is_available()
    use_cuda = os.environ.get("USE_CUDA", "auto").lower()
    if use_cuda in ("true", "1", "yes"):
        if not gpu_available:
            raise RuntimeError(
                "USE_CUDA=true was requested but no CUDA-capable GPU is visible to torch. "
                "Install a CUDA-enabled torch build and try again."
            )
        device = "cuda"
    elif use_cuda in ("false", "0", "no"):
        device = "cpu"
    else:
        device = "cuda" if gpu_available else "cpu"

    if device == "cpu" and not gpu_available:
        print("WARNING: torch cuda is unavailable in this environment; using CPU transcription.")
        print("If your machine has a GPU, install a CUDA-enabled PyTorch build and set USE_CUDA=true.")

    model_size = os.environ.get("WHISPER_MODEL_SIZE", "medium" if device == "cuda" else "medium")
    compute_type = "int8" if device == "cuda" else "int8"
    cpu_threads = int(os.environ.get("CPU_THREADS", "2"))

    print(f"Loading Whisper model ({model_size}, device={device}, compute_type={compute_type})...")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
    )

    try:
        segments_iter, info = model.transcribe(
            state["video_path"],
            beam_size=1,
            best_of=1,
            temperature=[0.0],
            vad_filter=True,
            chunk_length=30,
        )
    except RuntimeError as exc:
        if "failed to allocate memory" in str(exc).lower() and model_size != "small":
            next_size = "medium" if model_size == "large-v2" else "small"
            print(f"Memory allocation failed during transcription; retrying with the {next_size} Whisper model...")
            del model
            gc.collect()
            model = WhisperModel(
                next_size,
                device=device,
                compute_type="float16" if device == "cuda" else "int8",
                cpu_threads=cpu_threads,
                num_workers=1,
            )
            segments_iter, info = model.transcribe(
                state["video_path"],
                beam_size=1,
                best_of=1,
                temperature=[0.0],
                vad_filter=True,
                chunk_length=30,
            )
        else:
            raise

    segments = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments_iter
        if s.text.strip()
    ]

    del model
    gc.collect()
    return {"segments": segments, "detected_lang": info.language}


INDIC_TO_FLORES = {
    "hi": "hin_Deva", "bn": "ben_Beng", "gu": "guj_Gujr", "kn": "kan_Knda",
    "ml": "mal_Mlym", "mr": "mar_Deva", "or": "ory_Orya", "pa": "pan_Guru",
    "ta": "tam_Taml", "te": "tel_Telu", "ur": "urd_Arab", "as": "asm_Beng",
}


def Translate_Segments(state: WorkFlow):
    # NOTE: only Indic languages are routed right now. Non-Indic sources
    # like German/French will raise a KeyError here until an NLLB (or
    # similar) fallback branch is added -- fine for testing on Hindi
    # clips, but wire in a fallback before running your actual submission
    # videos if any of them aren't in an Indian language.
    src_lang = INDIC_TO_FLORES[state["detected_lang"]]  # e.g. "hi" -> "hin_Deva"
    texts = [s["text"] for s in state["segments"]]

    translations = translate_batch_indic(texts, src_lang=src_lang, tgt_lang="eng_Latn")

    translated_segments = [
        {"start": s["start"], "end": s["end"], "text": t}
        for s, t in zip(state["segments"], translations)
    ]
    return {"translated_segments": translated_segments}


def TTS_segments(state: WorkFlow):
    """Synthesizes English speech per translated segment and time-stretches
    it to fit the original segment's duration, keeping the dub aligned
    to the source video's timeline.
    """
    video_id = os.path.splitext(os.path.basename(state["video_path"]))[0]
    tts_dir = os.path.join(os.getcwd(), f"{video_id}_audio")
    os.makedirs(tts_dir, exist_ok=True)

    voice = "en-US-AriaNeural"
    segments = state["translated_segments"]
    tts_clips = []

    print(f"\nSynthesizing {len(segments)} segments with edge-tts ({voice})...")
    for i, segment in enumerate(segments):
        text = segment["text"].strip()
        if not text:
            continue

        raw_path = os.path.join(tts_dir, f"tts_{i:04d}_raw.mp3")
        final_path = os.path.join(tts_dir, f"tts_{i:04d}.wav")

        asyncio.run(edge_tts.Communicate(text, voice).save(raw_path))

        target_ms = max((segment["end"] - segment["start"]) * 1000, 1)
        raw_duration_ms = int(_probe_duration_seconds(raw_path) * 1000)
        factor = max(0.5, min(raw_duration_ms / target_ms, 2.0))

        ffmpeg_cmd = ["ffmpeg", "-y", "-i", raw_path]
        if abs(factor - 1.0) > 0.03:
            ffmpeg_cmd += ["-filter:a", f"atempo={factor}"]
        ffmpeg_cmd += [final_path]

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        tts_clips.append((segment["start"], final_path))
        print(f"  [{i + 1}/{len(segments)}] -> {final_path}")

    return {"tts_clips": tts_clips}


def _probe_duration_seconds(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def Mix_Audio(state: WorkFlow):
    """Overlays every synthesized clip onto a silent track spanning the
    full video's duration (so each line lands at its original timestamp),
    then remuxes that track onto the source video. The video stream is
    copied, never re-encoded, so this step is fast and lossless on the
    visual side no matter how long the source video is.
    """
    video_path = state["video_path"]
    tts_clips = state["tts_clips"]

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.getcwd(), "dubbed_output")
    os.makedirs(output_dir, exist_ok=True)

    dubbed_audio_path = os.path.join(output_dir, f"{video_id}_dubbed_audio.wav")
    output_path = os.path.join(output_dir, f"{video_id}_dubbed.mp4")

    print("\nProbing source video duration...")
    total_ms = int(_probe_duration_seconds(video_path) * 1000)

    print(f"Building {total_ms / 1000:.1f}s dubbed audio track from {len(tts_clips)} clips...")
    gain_db = float(os.environ.get("DUB_VOLUME_DB", "6"))
    if not tts_clips:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", str(total_ms / 1000),
                "-c:a", "pcm_s16le",
                dubbed_audio_path,
            ],
            check=True,
            capture_output=True,
        )
    else:
        # Create delayed, full-length copies of each clip so they can be
        # mixed without constructing a large in-memory audio track.
        delayed_files = []
        total_s = total_ms / 1000.0
        for idx, (start_sec, clip_path) in enumerate(tts_clips):
            delay_ms = int(start_sec * 1000)
            delayed = os.path.join(output_dir, f"delayed_{idx:04d}.wav")

            # Use adelay to push the clip to the right start position, and
            # trim/pad to the total length so all inputs are the same duration.
            afilter = f"adelay={delay_ms}|{delay_ms}"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", clip_path,
                    "-af", afilter,
                    "-t", str(total_s),
                    "-c:a", "pcm_s16le",
                    delayed,
                ],
                check=True,
                capture_output=True,
            )
            delayed_files.append(delayed)

        # Mix all delayed files together into the dubbed track and boost volume.
        mix_cmd = ["ffmpeg", "-y"]
        for fpath in delayed_files:
            mix_cmd.extend(["-i", fpath])

        inputs = "".join(f"[{i}:a]" for i in range(len(delayed_files)))
        filter_complex = (
            f"{inputs}amix=inputs={len(delayed_files)}:duration=longest:dropout_transition=0,"
            f"volume={gain_db}dB[aout]"
        )
        mix_cmd.extend(["-filter_complex", filter_complex, "-map", "[aout]", "-c:a", "pcm_s16le", dubbed_audio_path])
        subprocess.run(mix_cmd, check=True, capture_output=True)

        # Cleanup intermediate delayed files to save disk space.
        for fpath in delayed_files:
            try:
                os.remove(fpath)
            except OSError:
                pass

    print(f"  -> {dubbed_audio_path}")

    print("Remuxing with original video (video stream copied, not re-encoded)...")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", dubbed_audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ],
        check=True, capture_output=True,
    )
    print(f"Done -> {output_path}")

    return {"output_path": output_path}


# --- Graph wiring -----------------------------------------------------
# Straight-line chain for now: download -> transcribe -> translate ->
# synthesize -> mix. This is *not* the conditional-edge design discussed
# earlier (routing Indic vs. non-Indic languages to different translate
# nodes) -- that still needs an NLLB fallback node before it can branch,
# since Translate_Segments only has the Indic path built. Once that
# exists, swap the "translate" add_edge below for:
#
#   def route_translation(state: WorkFlow) -> str:
#       return "translate" if state["detected_lang"] in INDIC_TO_FLORES else "translate_general"
#   graph.add_conditional_edges("transcribe", route_translation, ["translate", "translate_general"])
#
# MemorySaver below is in-memory only -- fine for a single run, but for
# the 2-hour test video you may want SqliteSaver instead, so a crash in
# `mix` lets you resume from there instead of re-running transcription
# and translation from scratch.
if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver

    graph = StateGraph(WorkFlow)
    graph.add_node("download", Download_Video)
    graph.add_node("transcribe", Transcribe_Video)
    graph.add_node("translate", Translate_Segments)
    graph.add_node("synthesize", TTS_segments)
    graph.add_node("mix", Mix_Audio)

    graph.add_edge(START, "download")
    graph.add_edge("download", "transcribe")
    graph.add_edge("transcribe", "translate")
    graph.add_edge("translate", "synthesize")
    graph.add_edge("synthesize", "mix")
    graph.add_edge("mix", END)

    app = graph.compile(checkpointer=MemorySaver())

    url = input("Enter YouTube URL: ")
    #config = {"configurable": {"thread_id": url}}
    result = app.invoke(cast(WorkFlow, {"url": url}), config={"configurable": {"thread_id": url}})
    print(f"\nFinal dubbed video: {result['output_path']}")