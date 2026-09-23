"""Hugging Face Whisper transcription for browser-recorded audio."""

from __future__ import annotations

import os

from dotenv import load_dotenv

MODEL_NAME = os.getenv("TRUTHBRIDGE_ASR_MODEL", "openai/whisper-large-v3")
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def transcribe_audio(audio: bytes, filename: str = "recording.webm") -> str:
    if not audio:
        raise ValueError("The recording is empty")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("The recording is too large")

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not configured")

    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=token)
    result = client.automatic_speech_recognition(
        audio=audio,
        model=MODEL_NAME,
    )
    text = str(getattr(result, "text", "") or "").strip()
    if not text:
        raise ValueError("Hugging Face returned no transcription")
    return text
