"""
HealthMax — Layer 1: ASR (Automatic Speech Recognition)
Transcribes audio bytes to Bangla text using Whisper fine-tuned on Bangla.

Model: asif00/whisper-bangla (HuggingFace)
Fallback: If confidence is below CONFIDENCE_THRESHOLD, returns a structured
          prompt asking the user to speak again or switch to text/yes-no mode.

Collaborator instructions:
    - Load the Whisper model once at startup (see load_model()).
    - The main function to implement is transcribe_audio(audio_bytes) -> str.
    - Add dialect identification if time permits (FLEX feature).
    - Test with at least 10 sentences per dialect before integration.
"""

import io
import logging
import tempfile
import os
from typing import Optional

import torch

logger = logging.getLogger("healthmax.asr")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ASR_MODEL_ID = "asif00/whisper-bangla"
CONFIDENCE_THRESHOLD = -1.0          # avg_logprob threshold; below this → low confidence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000                  # Whisper expects 16kHz audio

# Bangla fallback message when ASR confidence is low
LOW_CONFIDENCE_MESSAGE = (
    "আমি আপনার কথা স্পষ্টভাবে বুঝতে পারিনি। "
    "অনুগ্রহ করে ধীরে ধীরে বলুন অথবা লিখে জানান।"
)

# ---------------------------------------------------------------------------
# Model loading (called once at startup)
# ---------------------------------------------------------------------------

_pipeline = None  # Lazy-loaded whisper pipeline


def load_model():
    """
    Load the Whisper ASR pipeline from HuggingFace.
    Call this once during app startup to avoid cold-start latency.

    TODO (collaborator):
        - Use transformers.pipeline("automatic-speech-recognition", ...)
        - Set device=DEVICE, chunk_length_s=30, return_timestamps=False
        - Cache model to /tmp or S3-backed local path to avoid re-download
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading Whisper ASR model: %s on %s", ASR_MODEL_ID, DEVICE)

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        _pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_ID,
            device=0 if DEVICE == "cuda" else -1,
            chunk_length_s=30,
        )
        logger.info("ASR model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load ASR model: %s", e)
        _pipeline = None

    return _pipeline


# ---------------------------------------------------------------------------
# Core ASR function
# ---------------------------------------------------------------------------

def transcribe_audio(audio_bytes: bytes, file_format: str = "wav") -> Optional[str]:
    """
    Transcribe raw audio bytes to Bangla text.

    Args:
        audio_bytes: Raw audio data (WAV, M4A, OGG, WEBM).
        file_format: File format hint for ffmpeg/librosa decoding.

    Returns:
        Bangla transcript string, or LOW_CONFIDENCE_MESSAGE if confidence
        is too low, or None if transcription completely fails.

    TODO (collaborator):
        1. Write audio_bytes to a temp file.
        2. Load as 16kHz mono waveform using librosa or soundfile.
        3. Pass numpy array to _pipeline().
        4. Parse the 'chunks' output to get avg_logprob for confidence check.
        5. If avg_logprob < CONFIDENCE_THRESHOLD → return LOW_CONFIDENCE_MESSAGE.
        6. Otherwise return the transcribed text stripped of leading/trailing whitespace.
    """
    model = load_model()
    if model is None:
        logger.error("ASR model not loaded; cannot transcribe.")
        return None

    try:
        # Write bytes to a temporary file for librosa/ffmpeg to read
        with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # TODO: Load waveform at SAMPLE_RATE using librosa.load(tmp_path, sr=SAMPLE_RATE)
        # TODO: Run model({"array": waveform, "sampling_rate": SAMPLE_RATE})
        # TODO: Extract transcript and confidence from result

        os.unlink(tmp_path)

        # Placeholder return — replace with actual transcription
        raise NotImplementedError("transcribe_audio() not yet implemented. See TODOs above.")

    except NotImplementedError:
        raise
    except Exception as e:
        logger.exception("ASR transcription failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Dialect identification (FLEX feature)
# ---------------------------------------------------------------------------

def identify_dialect(bangla_text: str) -> str:
    """
    Identify the Bangla dialect of the input text.

    Returns one of: 'dhaka', 'chittagong', 'sylheti', 'noakhali', 'barishal', 'unknown'

    TODO (collaborator — FLEX):
        - Train a lightweight text classifier on BanglaDialecto corpus.
        - Use dialect label to adjust downstream prompting strategy.
        - For high-WER dialects (Chittagong, Noakhali), trigger yes/no fallback.
    """
    # Placeholder
    return "unknown"


def should_use_structured_fallback(dialect: str, avg_logprob: float) -> bool:
    """
    Decide whether to fall back to structured yes/no prompting.

    Fallback is triggered if:
        - Dialect is Chittagong or Noakhali (high WER dialects), OR
        - ASR confidence is below threshold regardless of dialect.

    Args:
        dialect: Identified dialect string.
        avg_logprob: Average log probability from Whisper output.

    Returns:
        True if structured yes/no fallback should be used.
    """
    high_wer_dialects = {"chittagong", "noakhali"}
    return dialect in high_wer_dialects or avg_logprob < CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Dev test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("ASR module loaded. Run with an audio file for testing.")
    print(f"Device: {DEVICE}")
    print(f"Model: {ASR_MODEL_ID}")
