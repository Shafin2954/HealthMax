"""
HealthMax — Layer 4: XGBoost Disease Classifier
Predicts the top-3 most probable diseases from a list of extracted symptom strings.

Model: XGBoost classifier trained on the Bangla Symptoms-Disease dataset
       (85 diseases, 172 unique symptoms, 758 symptom-disease relations).
Saved model: models/disease_classifier.json

Collaborator instructions:
    - Train the model using data/process_datasets.py or the notebook.
    - Implement predict_diseases() — the primary function called by main.py.
    - Target Macro F1 > 0.80. Record the actual score in tests/eval_classifier.py.
    - The vocabulary (symptom binarizer) must be saved alongside the model.
"""

import logging
import os
import json
from typing import Optional

import numpy as np

logger = logging.getLogger("healthmax.classifier")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "CLASSIFIER_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "disease_classifier.json"),
)
BINARIZER_PATH = os.environ.get(
    "BINARIZER_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "symptom_binarizer.json"),
)
LABEL_ENCODER_PATH = os.environ.get(
    "LABEL_ENCODER_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "disease_label_encoder.json"),
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_symptom_vocab: list = []     # Ordered list of 172 symptom names (Bangla)
_disease_labels: list = []    # Ordered list of 85 disease names (Bangla/English)


def load_model():
    """
    Load the trained XGBoost classifier and associated vocabularies.

    TODO (collaborator):
        1. import xgboost as xgb
        2. _model = xgb.Booster(); _model.load_model(MODEL_PATH)
        3. Load _symptom_vocab from BINARIZER_PATH (JSON list of symptom strings)
        4. Load _disease_labels from LABEL_ENCODER_PATH (JSON list of disease names)
        5. Validate that len(_disease_labels) matches _model.num_boosted_rounds or num_class param
    """
    global _model, _symptom_vocab, _disease_labels

    if _model is not None:
        return _model

    logger.info("Loading XGBoost classifier from %s", MODEL_PATH)

    try:
        import xgboost as xgb  # type: ignore

        if not os.path.exists(MODEL_PATH):
            logger.warning("Classifier model not found at %s. Train first.", MODEL_PATH)
            return None

        _model = xgb.Booster()
        _model.load_model(MODEL_PATH)
        logger.info("XGBoost model loaded.")
    except Exception as e:
        logger.error("Failed to load XGBoost model: %s", e)
        return None

    try:
        with open(BINARIZER_PATH, "r", encoding="utf-8") as f:
            _symptom_vocab = json.load(f)
        with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
            _disease_labels = json.load(f)
        logger.info("Vocabularies loaded: %d symptoms, %d diseases", len(_symptom_vocab), len(_disease_labels))
    except Exception as e:
        logger.error("Failed to load vocabularies: %s", e)

    return _model


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def symptoms_to_feature_vector(symptoms: list) -> np.ndarray:
    """
    Convert a list of extracted symptom strings to a binary feature vector.

    Args:
        symptoms: List of symptom strings extracted by the NER layer.
                  Must be in the same language/form as _symptom_vocab entries.

    Returns:
        1-D numpy float32 array of shape (172,) with 1.0 where symptom is present.

    TODO (collaborator):
        - Normalize symptom strings before lookup (lowercase, strip, etc.)
        - Use fuzzy matching (rapidfuzz) if exact match fails, with threshold 80.
    """
    load_model()
    if not _symptom_vocab:
        return np.zeros(1, dtype=np.float32)

    symptom_set = set(s.strip() for s in symptoms)
    vector = np.array(
        [1.0 if sym in symptom_set else 0.0 for sym in _symptom_vocab],
        dtype=np.float32,
    )
    return vector


def predict_diseases(symptoms: list, top_n: int = 3) -> list:
    """
    Predict the top-n most probable diseases from a list of symptom strings.

    Args:
        symptoms: List of symptom strings from the NER layer.
        top_n:    Number of top diseases to return (default 3).

    Returns:
        List of dicts, sorted by probability (highest first):
            [
                {'name': 'ডেঙ্গু', 'name_en': 'Dengue', 'probability': 0.87},
                {'name': 'টাইফয়েড', 'name_en': 'Typhoid', 'probability': 0.62},
                ...
            ]
        Returns [] if model not loaded or symptoms list is empty.

    TODO (collaborator):
        1. Call symptoms_to_feature_vector(symptoms) → feature_vector.
        2. Wrap in xgb.DMatrix for inference.
        3. Call _model.predict(dmatrix) → probabilities array of shape (num_classes,).
        4. Get top_n indices via np.argsort(probs)[::-1][:top_n].
        5. Map indices to _disease_labels.
        6. Return structured list.
    """
    model = load_model()

    if model is None or not symptoms:
        logger.warning("Classifier not ready or no symptoms provided.")
        return []

    try:
        import xgboost as xgb  # type: ignore

        feature_vector = symptoms_to_feature_vector(symptoms)
        dmatrix = xgb.DMatrix(feature_vector.reshape(1, -1))
        probs = model.predict(dmatrix)[0]  # shape: (num_classes,)

        top_indices = np.argsort(probs)[::-1][:top_n]
        results = []
        for idx in top_indices:
            disease = _disease_labels[idx] if idx < len(_disease_labels) else f"Disease_{idx}"
            results.append({
                "name": disease,
                "name_en": disease,       # TODO: add Bangla label mapping
                "probability": float(probs[idx]),
            })
        return results
    except NotImplementedError:
        raise
    except Exception as e:
        logger.exception("Disease prediction failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Dev test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_symptoms = ["জ্বর", "মাথাব্যথা", "চোখ লাল", "গা ব্যথা"]
    print("Input symptoms:", test_symptoms)
    predictions = predict_diseases(test_symptoms)
    for p in predictions:
        print(f"  {p['name']} — {p['probability']:.2%}")
