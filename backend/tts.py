"""
HealthMax — TTS (Text-to-Speech)
Converts Bangla triage response text to audio using Google Cloud TTS.

Voice: Bangla WaveNet (bn-BD-Wavenet-A or B)
Output: MP3 bytes returned to the caller (browser or WhatsApp media attachment)

Collaborator instructions:
    - Set GOOGLE_APPLICATION_CREDENTIALS env var to the service account JSON key path.
    - Implement synthesize_bangla() — the primary function called from the frontend route.
    - This is a FLEX feature; implement after core pipeline is working.
    - For the demo, TTS playback is a "wow" feature — it impresses judges significantly.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("healthmax.tts")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GCP_LANGUAGE_CODE = "bn-BD"
GCP_VOICE_NAME = "bn-BD-Wavenet-A"   # Female voice; use bn-BD-Wavenet-B for male
GCP_AUDIO_ENCODING = "MP3"
MAX_CHARS = 500  # Google TTS limit per request is 5000; keep responses short

# ---------------------------------------------------------------------------
# Core TTS function
# ---------------------------------------------------------------------------

def synthesize_bangla(text: str) -> Optional[bytes]:
    """
    Convert a Bangla text string to MP3 audio bytes using Google Cloud TTS.

    Args:
        text: Bangla text to synthesize (max MAX_CHARS characters).
              Long texts will be truncated with a notice.

    Returns:
        MP3 audio bytes, or None if synthesis fails.

    TODO (collaborator):
        1. pip install google-cloud-texttospeech
        2. from google.cloud import texttospeech
        3. client = texttospeech.TextToSpeechClient()
        4. synthesis_input = texttospeech.SynthesisInput(text=text[:MAX_CHARS])
        5. voice = texttospeech.VoiceSelectionParams(
               language_code=GCP_LANGUAGE_CODE,
               name=GCP_VOICE_NAME)
        6. audio_config = texttospeech.AudioConfig(
               audio_encoding=texttospeech.AudioEncoding.MP3)
        7. response = client.synthesize_speech(
               input=synthesis_input, voice=voice, audio_config=audio_config)
        8. Return response.audio_content (bytes)
    """
    if not text.strip():
        return None

    try:
        from google.cloud import texttospeech  # type: ignore

        client = texttospeech.TextToSpeechClient()

        truncated_text = text[:MAX_CHARS]
        if len(text) > MAX_CHARS:
            logger.info("TTS text truncated from %d to %d chars.", len(text), MAX_CHARS)

        synthesis_input = texttospeech.SynthesisInput(text=truncated_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=GCP_LANGUAGE_CODE,
            name=GCP_VOICE_NAME,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.9,   # Slightly slower for medical clarity
        )
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        logger.info("TTS synthesis successful: %d bytes", len(response.audio_content))
        return response.audio_content

    except ImportError:
        logger.warning("google-cloud-texttospeech not installed. TTS unavailable.")
        return None
    except Exception as e:
        logger.exception("TTS synthesis failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# FastAPI route helper (for use in main.py)
# ---------------------------------------------------------------------------

def tts_endpoint_handler(text: str):
    """
    Returns a FastAPI Response with MP3 audio content, or 503 if TTS fails.
    Import and register this in main.py as a /api/tts POST route.

    TODO (collaborator): Wire this into main.py as:
        @app.post("/api/tts")
        async def tts_route(request: TriageRequest):
            return tts_endpoint_handler(request.text)
    """
    from fastapi.responses import Response  # type: ignore
    audio_bytes = synthesize_bangla(text)
    if audio_bytes is None:
        from fastapi import HTTPException  # type: ignore
        raise HTTPException(status_code=503, detail="TTS service unavailable.")
    return Response(content=audio_bytes, media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# Dev test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "আপনার সম্ভাব্য রোগ ডেঙ্গু। অনুগ্রহ করে উপজেলা স্বাস্থ্য কমপ্লেক্সে যান।"
    audio = synthesize_bangla(sample)
    if audio:
        with open("test_tts_output.mp3", "wb") as f:
            f.write(audio)
        print("TTS output saved to test_tts_output.mp3")
    else:
        print("TTS failed — check credentials.")
