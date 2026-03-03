import os
import json
import numpy as np
from typing import List, Dict

_classifier_model = None
_label_encoder = None
_symptom_binarizer = None

MODEL_PATH = "models/disease_classifier.json"
LABEL_ENCODER_PATH = "models/label_encoder.json"
SYMPTOM_LIST_PATH = "models/symptom_list.json"


def _load_classifier():
    global _classifier_model, _label_encoder, _symptom_binarizer
    if _classifier_model is None:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import MultiLabelBinarizer

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Classifier model not found at {MODEL_PATH}. "
                "Run: python data/process_datasets.py"
            )

        print("[Classifier] Loading XGBoost model...")
        _classifier_model = xgb.XGBClassifier()
        _classifier_model.load_model(MODEL_PATH)

        with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
            label_classes = json.load(f)
        _label_encoder = label_classes

        with open(SYMPTOM_LIST_PATH, "r", encoding="utf-8") as f:
            symptom_list = json.load(f)
        _symptom_binarizer = symptom_list
        print("[Classifier] Mse dicts with: disease, symptoms, urgency, speciaier_model, _label_encoder, _symptom_binarizer


def predict_diseases(symptoms: List[str], top_n: int = 3) -> List[Dict]:
    """
    Given a list of extracted symptom strings, predict top-N diseases.
    Returns: [{"disease": str, "probability": float}, ...]
    """
    model, label_classes, symptom_list = _load_classifier()

    # Binarize input symptoms
    feature_vector = np.zeros(len(symptom_list), dtype=np.float32)
    for i, symptom in enumerate(symptom_list):
        if symptom in symptoms:
            feature_vector[i] = 1.0

    # If no symptoms matched, return empty
    if feature_vector.sum() == 0:
        return []

    proba = model.predict_proba([feature_vector])[0]
    top_indices = np.argsort(proba)[::-1][:top_n]

    results = []
    for idx in top_indices:
        if proba[idx] > 0.05:  # Filter very low probability predictions
            results.append({
                "disease": label_classes[idx],
                "probability": float(proba[idx])
            })

    return results
