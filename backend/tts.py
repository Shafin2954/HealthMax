import os
from typing import Optional


def text_to_speech_bangla(text: str) -> Optional[bytes]:
    """
    Convert Bangla text to speech using Google Cloud TTS (WaveNet).
    Returns audio bytes (MP3) or None if TTS is unavailable.
    """
    gcp_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not gcp_creds or not os.path.exists(gcp_creds):
        print("[TTS] GCP credentials not found. TTS disabled.")
        return None

    try:
        from google.cloud import texttospeech

        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="bn-BD",
            name="bn-IN-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        return response.audio_content

    except Exception as e:
        print(f"[TTS] Failed: {e}")
        return None
