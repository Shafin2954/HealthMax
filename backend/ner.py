"""
HealthMax — Layer 2: NER (Named Entity Recognition)
Extracts symptom, disease, and medicine entities from Bangla text.

Model: sagorsarker/bangla-bert-base fine-tuned on BanglaHealthNER + MedER datasets.
       Alternatively: bangla-speechprocessing/BanglaNER (pre-fine-tuned fallback).

BIO Tag Labels:
    B-SYMPTOM, I-SYMPTOM
    B-DISEASE, I-DISEASE
    B-MEDICINE, I-MEDICINE
    O (outside any entity)

Collaborator instructions:
    - Fine-tune the model in notebooks/banglabert_finetune.ipynb on Google Colab.
    - Upload the fine-tuned checkpoint to S3 and set MODEL_PATH to the local cache path.
    - Implement extract_symptoms() — this is the primary function called by main.py.
    - Return a dict with keys: 'symptoms', 'diseases', 'medicines' (each a list of strings).
"""

import logging
from typing import Optional

import torch
import os

logger = logging.getLogger("healthmax.ner")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to fine-tuned NER model (local cache or S3-synced directory)
MODEL_PATH = os.environ.get("NER_MODEL_PATH", "sagorsarker/bangla-bert-base")

# Fallback model if fine-tuned weights are not available
FALLBACK_MODEL_PATH = "csebuetnlp/banglabert"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BIO label mapping — must match the training label set exactly
LABEL_MAP = {
    0: "O",
    1: "B-SYMPTOM",
    2: "I-SYMPTOM",
    3: "B-DISEASE",
    4: "I-DISEASE",
    5: "B-MEDICINE",
    6: "I-MEDICINE",
}

import os


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_ner_pipeline = None


def load_model():
    """
    Load the BanglaBERT NER pipeline.
    Called once at application startup.

    TODO (collaborator):
        - Use transformers.pipeline("ner", model=MODEL_PATH, tokenizer=MODEL_PATH,
          aggregation_strategy="simple", device=...)
        - If MODEL_PATH fails to load, fall back to FALLBACK_MODEL_PATH with a warning.
        - Log which model was loaded so we can verify in CloudWatch.
    """
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline

    logger.info("Loading NER model: %s on %s", MODEL_PATH, DEVICE)

    try:
        from transformers import pipeline as hf_pipeline # type: ignore

        _ner_pipeline = hf_pipeline(
            "token-classification",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            aggregation_strategy="simple",
            device=0 if DEVICE == "cuda" else -1,
        )
        logger.info("NER model loaded: %s", MODEL_PATH)
    except Exception as e:
        logger.warning("Primary NER model failed (%s); trying fallback.", e)
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
            _ner_pipeline = hf_pipeline(
                "token-classification",
                model=FALLBACK_MODEL_PATH,
                aggregation_strategy="simple",
                device=0 if DEVICE == "cuda" else -1,
            )
            logger.info("Fallback NER model loaded: %s", FALLBACK_MODEL_PATH)
        except Exception as e2:
            logger.error("Both NER models failed to load: %s", e2)
            _ner_pipeline = None

    return _ner_pipeline


# ---------------------------------------------------------------------------
# Core NER extraction
# ---------------------------------------------------------------------------

def extract_symptoms(bangla_text: str) -> dict:
    """
    Extract medical entities from a Bangla symptom description.

    Args:
        bangla_text: Raw Bangla text from the user (post-ASR or direct input).

    Returns:
        dict with keys:
            'symptoms'  : list[str] — extracted symptom phrases
            'diseases'  : list[str] — extracted disease mentions
            'medicines' : list[str] — extracted medicine mentions
        Example:
            {
                'symptoms':  ['জ্বর', 'মাথাব্যথা', 'চোখ লাল'],
                'diseases':  ['ডেঙ্গু'],
                'medicines': []
            }

    TODO (collaborator):
        1. Load the NER pipeline via load_model().
        2. Run _ner_pipeline(bangla_text).
        3. Group consecutive BIO tokens into full entity spans.
        4. Separate by entity_group into symptoms / diseases / medicines.
        5. Deduplicate and strip whitespace from each entity.
        6. Return the structured dict.
    """
    model = load_model()

    result = {"symptoms": [], "diseases": [], "medicines": []}

    if model is None:
        logger.error("NER model not available; returning empty entities.")
        return result

    try:
        raw_entities = model(bangla_text)
        # TODO: Parse raw_entities list from HuggingFace pipeline output.
        # Each item has keys: entity_group, score, word, start, end
        # Map entity_group → result key:
        #   'SYMPTOM'  → result['symptoms']
        #   'DISEASE'  → result['diseases']
        #   'MEDICINE' → result['medicines']
        raise NotImplementedError("Entity parsing not yet implemented. See TODOs above.")
    except NotImplementedError:
        raise
    except Exception as e:
        logger.exception("NER extraction failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def bio_tags_to_spans(tokens: list, tags: list) -> list:
    """
    Convert a list of (token, BIO-tag) pairs into entity spans.

    Args:
        tokens: List of word-piece tokens.
        tags:   Corresponding BIO tags (strings).

    Returns:
        List of dicts: [{'entity': str, 'tokens': list[str]}, ...]

    TODO (collaborator): Implement the BIO span merging logic here.
    """
    spans = []
    current_entity = None
    current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_entity:
                spans.append({"entity": current_entity, "tokens": current_tokens})
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_entity == tag[2:]:
            current_tokens.append(token)
        else:
            if current_entity:
                spans.append({"entity": current_entity, "tokens": current_tokens})
            current_entity = None
            current_tokens = []

    if current_entity:
        spans.append({"entity": current_entity, "tokens": current_tokens})

    return spans


# ---------------------------------------------------------------------------
# Dev test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "তিন দিন ধরে জ্বর, মাথাব্যথা, চোখ লাল এবং গা ব্যথা।"
    print("Input:", sample)
    print("Entities:", extract_symptoms(sample))
