from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tool_node
from typing import List ,Dict,Any,Optional,TypedDict 
from dotenv import load_dotenv
load_dotenv()
import os 
import yt_dlp
from faster_whisper import WhisperModel
from IndicTransToolkit import IndicProcessor
import torch
import gc
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM


def translate_batch_indic(sentences, src_lang, tgt_lang="eng_Latn"):
    """Translates a list of strings using AI4Bharat IndicTrans2 model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is required to access ai4bharat/indictrans2-indic-en-dist-200M. "
            "Set HF_TOKEN in your environment with a token that has access."
        )

    model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
    print(f"\nLoading IndicTrans2 Model ({src_lang} -> {tgt_lang}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=hf_token,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=hf_token,
    ).to(device)
    ip = IndicProcessor(inference=True)

    print("Preprocessing & translating sentences...")
    translated_sentences = []
    batch_size = int(os.environ.get("TRANSLATE_BATCH_SIZE", "8"))
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        preprocessed_batch = ip.preprocess_batch(chunk, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(
            preprocessed_batch,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                num_beams=4,
                num_return_sequences=1,
                max_length=256,
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_sentences.extend(ip.postprocess_batch(decoded_outputs, lang=tgt_lang))

        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return translated_sentences