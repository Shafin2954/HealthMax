import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
PUBLIC_MODEL_DIR = REPO_ROOT / "healthmax-ai-assistant" / "public" / "model"
PUBLIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_PATH = MODELS_DIR / "disease_classifier.json"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.json"
SYMPTOM_LIST_PATH = MODELS_DIR / "symptom_list.json"
RAG_VECTORIZER_PATH = MODELS_DIR / "disease_rag_vectorizer.pkl"
DISEASE_RECORDS_PATH = MODELS_DIR / "disease_records.json"
MEDICINE_CSV_PATH = REPO_ROOT / "assets" / "medicine.csv"

PRICE_PATTERN = re.compile(r"৳\s*([0-9]+(?:\.[0-9]+)?)")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))


def _extract_tree_scores(model_json: Dict[str, Any], features: np.ndarray) -> np.ndarray:
    trees = model_json["learner"]["gradient_booster"]["model"]["trees"]
    tree_info = model_json["learner"]["gradient_booster"]["model"]["tree_info"]
    num_class = int(model_json["learner"]["learner_model_param"]["num_class"])
    scores = np.zeros(num_class, dtype=np.float64)

    for tree, class_index in zip(trees, tree_info):
        node_index = 0
        left_children = tree["left_children"]
        right_children = tree["right_children"]
        split_indices = tree["split_indices"]
        split_conditions = tree["split_conditions"]
        default_left = tree["default_left"]
        leaf_weights = tree["base_weights"]

        while left_children[node_index] != -1:
            feature_index = split_indices[node_index]
            threshold = split_conditions[node_index]
            feature_value = float(features[feature_index])

            if math.isnan(feature_value):
                go_left = bool(default_left[node_index])
            else:
                go_left = feature_value < threshold

            node_index = left_children[node_index] if go_left else right_children[node_index]

        scores[int(class_index)] += float(leaf_weights[node_index])

    return scores


def _build_classifier_artifact() -> Dict[str, Any]:
    model_json = _read_json(CLASSIFIER_PATH)
    label_encoder = _read_json(LABEL_ENCODER_PATH)
    symptom_list = _read_json(SYMPTOM_LIST_PATH)

    booster = xgb.XGBClassifier()
    booster.load_model(CLASSIFIER_PATH)

    num_feature = int(model_json["learner"]["learner_model_param"]["num_feature"])
    zero_vector = np.zeros((1, num_feature), dtype=np.float32)
    raw_margin = booster.get_booster().predict(xgb.DMatrix(zero_vector), output_margin=True)[0]
    tree_only_scores = _extract_tree_scores(model_json, zero_vector[0])
    class_biases = (raw_margin - tree_only_scores).tolist()

    trees = []
    for tree in model_json["learner"]["gradient_booster"]["model"]["trees"]:
        trees.append(
            {
                "leftChildren": tree["left_children"],
                "rightChildren": tree["right_children"],
                "splitIndices": tree["split_indices"],
                "splitConditions": tree["split_conditions"],
                "defaultLeft": tree["default_left"],
                "leafWeights": tree["base_weights"],
            }
        )

    return {
        "numClass": int(model_json["learner"]["learner_model_param"]["num_class"]),
        "numFeature": num_feature,
        "treeInfo": model_json["learner"]["gradient_booster"]["model"]["tree_info"],
        "trees": trees,
        "classBiases": class_biases,
        "labels": label_encoder,
        "symptomList": symptom_list,
    }


def _sparse_row(row: np.ndarray) -> Dict[str, List[float] | List[int]]:
    indices = np.nonzero(row)[0]
    values = row[indices]
    return {
        "indices": indices.astype(int).tolist(),
        "values": values.astype(float).tolist(),
    }


def _build_rag_artifact() -> Dict[str, Any]:
    vectorizer = pickle.load(RAG_VECTORIZER_PATH.open("rb"))
    disease_records = _read_json(DISEASE_RECORDS_PATH)

    texts = [str(record.get("text_representation", "")) for record in disease_records]
    dense_matrix = vectorizer.transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_matrix = dense_matrix / norms

    records_with_vectors = []
    for record, row in zip(disease_records, normalized_matrix):
        normalized_record = dict(record)
        normalized_record["vector"] = _sparse_row(row)
        records_with_vectors.append(normalized_record)

    return {
        "vocabulary": {str(term): int(index) for term, index in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.astype(float).tolist(),
        "ngramRange": list(vectorizer.ngram_range),
        "lowercase": bool(vectorizer.lowercase),
        "records": records_with_vectors,
    }


def _extract_price_bdt(*values: object) -> float:
    for value in values:
        text = str(value or "")
        match = PRICE_PATTERN.search(text)
        if match:
            return float(match.group(1))
    return float("nan")


def _build_medicine_artifact() -> Dict[str, Any]:
    raw_df = pd.read_csv(MEDICINE_CSV_PATH)
    normalized = raw_df.copy()
    normalized.columns = [str(column).lower().strip().replace(" ", "_") for column in normalized.columns]

    if not {"brand_name", "generic_name"}.issubset(normalized.columns):
        canonical_columns = [
            "id",
            "brand_name",
            "product_type",
            "slug",
            "dosage_form",
            "generic_name",
            "strength",
            "manufacturer",
            "price_info",
            "pack_info",
        ]
        normalized = normalized.iloc[:, : len(canonical_columns)].copy()
        normalized.columns = canonical_columns

    text_columns = ["brand_name", "generic_name", "dosage_form", "manufacturer", "price_info", "pack_info"]
    for column in text_columns:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    normalized["price_bdt"] = normalized.apply(
        lambda row: _extract_price_bdt(row.get("price_info"), row.get("pack_info")),
        axis=1,
    )
    normalized["search_text"] = (
        normalized["brand_name"].str.casefold()
        + " "
        + normalized["generic_name"].str.casefold()
        + " "
        + normalized["dosage_form"].str.casefold()
    )

    records = []
    for _, row in normalized.iterrows():
        price = row.get("price_bdt")
        price_value = float(price) if pd.notna(price) else None
        records.append(
            {
                "brand_name": row.get("brand_name", ""),
                "generic_name": row.get("generic_name", ""),
                "dosage_form": row.get("dosage_form", ""),
                "manufacturer": row.get("manufacturer", ""),
                "price_bdt": price_value,
                "search_text": row.get("search_text", ""),
            }
        )

    return {"records": records}


def main() -> None:
    classifier_artifact = _build_classifier_artifact()
    rag_artifact = _build_rag_artifact()
    medicine_artifact = _build_medicine_artifact()

    _write_json(PUBLIC_MODEL_DIR / "classifier.json", classifier_artifact)
    _write_json(PUBLIC_MODEL_DIR / "rag.json", rag_artifact)
    _write_json(PUBLIC_MODEL_DIR / "medicines.json", medicine_artifact)

    _write_json(
        PUBLIC_MODEL_DIR / "manifest.json",
        {
            "classifier": "classifier.json",
            "rag": "rag.json",
            "medicines": "medicines.json",
        },
    )

    print(f"Exported browser artifacts to {PUBLIC_MODEL_DIR}")


if __name__ == "__main__":
    main()
