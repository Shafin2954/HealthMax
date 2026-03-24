import json
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_classifier_model: Optional[Any] = None
_label_encoder: Optional[List[str]] = None
_symptom_list: Optional[List[str]] = None

MODEL_PATH = "models/disease_classifier.json"
LABEL_ENCODER_PATH = "models/label_encoder.json"
SYMPTOM_LIST_PATH = "models/symptom_list.json"

SYMPTOM_ALIASES = {
    "মাথাব্যথা": "মাথা ব্যথা",
    "মাথা ব্যাথা": "মাথা ব্যথা",
    "গা ব্যথা": "শরীর ব্যথা",
    "গায়ে ব্যথা": "শরীর ব্যথা",
    "গায়ে ব্যথা": "শরীর ব্যথা",
    "শরীর ব্যথা": "শরীর ব্যথা",
    "বুক ব্যথা": "বুকে ব্যথা",
    "বুকে ব্যথা": "বুকে ব্যথা",
    "পেটব্যথা": "পেট ব্যথা",
    "পেটে ব্যথা": "পেট ব্যথা",
    "মাথা ঘোরা": "মাথা ঘোরা",
    "শ্বাস নিতে কষ্ট": "শ্বাসকষ্ট",
    "শ্বাস কষ্ট": "শ্বাসকষ্ট",
    "বমি বমি": "বমি বমি ভাব",
    "বমি বমি লাগছে": "বমি বমি ভাব",
    "ঠান্ডা": "ঠান্ডা",
}


def _normalize_symptom(symptom: str) -> str:
    """Normalize symptom text so extracted entities match training features better."""
    before_paren = symptom.split("(", 1)[0]
    normalized = unicodedata.normalize("NFKC", before_paren)
    normalized = normalized.replace("\u200c", "").replace("\u200d", "")
    normalized = normalized.replace("_", " ").replace("-", " ").strip()
    normalized = " ".join(normalized.split()).casefold()
    return SYMPTOM_ALIASES.get(normalized, normalized)


def _resolve_input_symptom(input_symptom: str, symptom_lookup: Dict[str, int]) -> Optional[int]:
    normalized_input = _normalize_symptom(input_symptom)
    exact_match = symptom_lookup.get(normalized_input)
    if exact_match is not None:
        return exact_match

    try:
        from rapidfuzz import fuzz, process

        best_match = process.extractOne(
            normalized_input,
            list(symptom_lookup.keys()),
            scorer=fuzz.WRatio,
        )
        if best_match and best_match[1] >= 88:
            return symptom_lookup[best_match[0]]
    except Exception:
        return None

    return None


def _load_classifier() -> Tuple[Any, List[str], List[str]]:
    global _classifier_model, _label_encoder, _symptom_list

    if _classifier_model is None or _label_encoder is None or _symptom_list is None:
        import xgboost as xgb

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Classifier model not found at {MODEL_PATH}. "
                "Run: python data/process_datasets.py"
            )

        print("[Classifier] Loading XGBoost model...")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)

        with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
            label_classes = json.load(f)
        with open(SYMPTOM_LIST_PATH, "r", encoding="utf-8") as f:
            symptom_list = json.load(f)

        _classifier_model = model
        _label_encoder = [str(label) for label in label_classes]
        _symptom_list = [str(symptom) for symptom in symptom_list]
        print("[Classifier] Model metadata loaded.")

    assert _classifier_model is not None
    assert _label_encoder is not None
    assert _symptom_list is not None
    return _classifier_model, _label_encoder, _symptom_list


def predict_diseases(symptoms: List[str], top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Given a list of extracted symptom strings, predict top-N diseases.
    Returns: [{"disease": str, "probability": float}, ...]
    """
    model, label_classes, symptom_list = _load_classifier()
    symptom_lookup = {_normalize_symptom(symptom): i for i, symptom in enumerate(symptom_list)}

    # Binarize input symptoms
    feature_vector = np.zeros(len(symptom_list), dtype=np.float32)
    for input_symptom in symptoms:
        matched_index = _resolve_input_symptom(input_symptom, symptom_lookup)
        if matched_index is not None:
            feature_vector[matched_index] = 1.0

    # If no symptoms matched, return empty
    if feature_vector.sum() == 0:
        return []

    proba = model.predict_proba([feature_vector])[0]
    top_indices = np.argsort(proba)[::-1][:top_n]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        if proba[idx] > 0.05:  # Filter very low probability predictions
            results.append({
                "disease": label_classes[idx],
                "probability": float(proba[idx])
            })

    return results
