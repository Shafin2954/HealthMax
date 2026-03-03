import io
import os
import tempfile
import numpy as np
from typing import Tuple

_asr_model = None
_asr_processor = None


def _load_asr_model():
    global _asr_model, _asr_processor
    if _asr_model is None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        model_name = "asif00/whisper-bangla"
        print(f"[ASR] Loading model: {model_name}")
        _asr_processor = WhisperProcessor.from_pretrained(model_name)
        _asr_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        _asr_model.eval()
        print("[ASR] Model loaded successfully.")
    return _asr_model, _asr_processor


def transcribe_audio(audio_bytes: bytes) -> Tuple[str, float]:
    """
    Transcribe audio bytes to Bangla text using Whisper.
    Returns: (transcript_text, confidence_score)
    confidence_score is 0.0 to 1.0 (derived from avg_logprob)
    """
    import torch
    import soundfile as sf

    model, processor = _load_asr_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio_array, sample_rate = sf.read(tmp_path)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16000 Hz if needed
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_features"],
                language="bn",
                return_dict_in_generate=True,
                output_scores=True
            )

        transcript = processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        # Estimate confidence from token scores
        if outputs.scores:
            import torch.nn.functional as F
            avg_log_prob = np.mean([
                F.log_softmax(score, dim=-1).max().item()
                for score in outputs.scores
            ])
            confidence = float(np.clip(np.exp(avg_log_prob), 0.0, 1.0))
        else:
            confidence = 0.7

        return transcript, confidence

    finally:
        os.unlink(tmp_path)


def apply_dialect_normalization(text: str) -> str:
    """
    Normalize common Bangla dialect variations to standard form.
    Covers Chittagong, Sylheti, Noakhali common substitutions.
    """
    dialect_map = {
        "মাতা": "মাথা",
        "গরম জ্বর": "উচ্চ জ্বর",
        "পেডে ব্যথা": "পেটে ব্যথা",
        "ওষুদ": "ওষুধ",
        "হাউনি": "হচ্ছে না",
        "খাইতে পারি না": "খেতে পারছি না",
        "বুইক্কা ব্যথা": "বুকে ব্যথা",
    }
    for dialect_form, standard_form in dialect_map.items():
        text = text.replace(dialect_form, standard_form)
    return text
