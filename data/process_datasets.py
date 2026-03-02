"""
HealthMax — Dataset Processing & FAISS Index Builder
Cleans, encodes, and indexes all datasets for the HealthMax pipeline.

Run this script once before starting the backend server:
    python data/process_datasets.py

Outputs:
    models/disease_rag.index         — FAISS vector index
    models/disease_records.json      — Parallel metadata list
    models/symptom_binarizer.json    — Ordered symptom vocabulary (172 symptoms)
    models/disease_label_encoder.json— Ordered disease label list (85 diseases)
    models/disease_classifier.json   — Trained XGBoost model
    data/dgda_medicines_clean.csv    — Cleaned DGDA drug registry

Collaborator instructions:
    - Download all Priority 1 & 2 datasets to data/raw/ before running.
    - Update FILE_PATHS to point to the correct filenames in data/raw/.
    - Run section by section using the --step flag for partial runs.
    - Target XGBoost Macro F1 > 0.80 — print it at the end of training.
"""

import os
import json
import logging
import argparse

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("healthmax.data")

# ---------------------------------------------------------------------------
# File paths — update these to match your downloaded filenames in data/raw/
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FILE_PATHS = {
    "symptoms_disease":  os.path.join(RAW_DIR, "bangla_symptoms_disease.csv"),
    "ner_dataset":       os.path.join(RAW_DIR, "bangla_health_ner.csv"),
    "meder_dataset":     os.path.join(RAW_DIR, "bangla_meder.csv"),
    "dgda_medicines":    os.path.join(RAW_DIR, "dgda_medicine_registry.csv"),
}

OUTPUT_PATHS = {
    "faiss_index":      os.path.join(MODELS_DIR, "disease_rag.index"),
    "disease_records":  os.path.join(MODELS_DIR, "disease_records.json"),
    "symptom_binarizer":os.path.join(MODELS_DIR, "symptom_binarizer.json"),
    "label_encoder":    os.path.join(MODELS_DIR, "disease_label_encoder.json"),
    "xgb_model":        os.path.join(MODELS_DIR, "disease_classifier.json"),
    "dgda_clean":       os.path.join(os.path.dirname(__file__), "dgda_medicines_clean.csv"),
}

EMBEDDING_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Step 1: Load & clean Symptoms-Disease dataset
# ---------------------------------------------------------------------------

def load_symptoms_disease(filepath: str) -> pd.DataFrame:
    """
    Load and clean the Bangla Symptoms-Disease dataset.

    Expected columns (adapt if CSV schema differs):
        disease, symptom_1, symptom_2, ..., symptom_n

    Returns:
        DataFrame with columns: ['disease', 'symptoms'] where 'symptoms' is a list.

    TODO (collaborator):
        1. Read CSV. Inspect columns — dataset has 85 diseases, 172 symptoms.
        2. Melt wide symptom columns into a tidy long format.
        3. Group by disease to get a list of symptoms per disease.
        4. Normalize disease and symptom names (strip, lowercase for matching).
        5. Return the cleaned DataFrame.
    """
    logger.info("Loading Symptoms-Disease dataset from %s", filepath)
    if not os.path.exists(filepath):
        logger.error("File not found: %s", filepath)
        return pd.DataFrame()

    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info("Raw shape: %s", df.shape)

    # TODO: Implement cleaning and normalization
    raise NotImplementedError("load_symptoms_disease() not yet implemented. See TODOs.")


# ---------------------------------------------------------------------------
# Step 2: Build symptom vocabulary & disease label encoder
# ---------------------------------------------------------------------------

def build_vocabularies(df: pd.DataFrame) -> tuple:
    """
    Build the symptom vocabulary and disease label list.

    Args:
        df: Cleaned symptoms-disease DataFrame.

    Returns:
        (symptom_vocab: list, disease_labels: list)
        symptom_vocab  — sorted list of all unique symptom strings (172 items)
        disease_labels — sorted list of all unique disease strings (85 items)

    TODO (collaborator):
        1. Collect all unique symptoms from df['symptoms'] (flattened list).
        2. Sort alphabetically for reproducibility.
        3. Collect all unique disease names from df['disease'].
        4. Sort alphabetically.
        5. Save both as JSON to OUTPUT_PATHS['symptom_binarizer'] and ['label_encoder'].
    """
    raise NotImplementedError("build_vocabularies() not yet implemented.")


# ---------------------------------------------------------------------------
# Step 3: Train XGBoost classifier
# ---------------------------------------------------------------------------

def train_xgboost_classifier(df: pd.DataFrame, symptom_vocab: list, disease_labels: list):
    """
    Train the XGBoost disease classifier.

    Args:
        df:             Cleaned symptoms-disease DataFrame.
        symptom_vocab:  Ordered symptom vocabulary.
        disease_labels: Ordered disease label list.

    Training params:
        n_estimators=200, max_depth=6, use_label_encoder=False,
        eval_metric='mlogloss', random_state=42, 80/20 train-test split.

    Saves model to OUTPUT_PATHS['xgb_model'].
    Prints Macro F1 to console.

    TODO (collaborator):
        1. Use MultiLabelBinarizer or manual vectorization to create feature matrix X.
        2. Encode disease labels to integers for y.
        3. Split into train/test (80/20, stratified).
        4. Train XGBClassifier.
        5. Evaluate: print classification_report with macro F1.
        6. Assert macro F1 > 0.80 (raise ValueError if below target).
        7. Save model via booster.save_model(OUTPUT_PATHS['xgb_model']).
    """
    raise NotImplementedError("train_xgboost_classifier() not yet implemented.")


# ---------------------------------------------------------------------------
# Step 4: Build FAISS RAG index
# ---------------------------------------------------------------------------

def build_faiss_index(df: pd.DataFrame):
    """
    Encode disease records and build a FAISS vector index for RAG retrieval.

    For each disease, create a text representation:
        "{disease_name}: {comma-joined symptom list}"
    Encode with paraphrase-multilingual-MiniLM.
    Build a FAISS IndexFlatIP (inner product = cosine on normalized vectors).

    Saves:
        - FAISS index to OUTPUT_PATHS['faiss_index']
        - Metadata JSON to OUTPUT_PATHS['disease_records']

    TODO (collaborator):
        1. from sentence_transformers import SentenceTransformer
        2. model = SentenceTransformer(EMBEDDING_MODEL_ID)
        3. Create text strings for each disease record.
        4. embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        5. import faiss; index = faiss.IndexFlatIP(embedding_dim)
        6. index.add(embeddings.astype(np.float32))
        7. faiss.write_index(index, OUTPUT_PATHS['faiss_index'])
        8. Save disease_records (list of dicts with disease_name, symptoms, urgency, specialist).
    """
    raise NotImplementedError("build_faiss_index() not yet implemented.")


# ---------------------------------------------------------------------------
# Step 5: Clean DGDA drug registry
# ---------------------------------------------------------------------------

def clean_dgda_dataset(filepath: str):
    """
    Clean and normalize the DGDA medicine registry CSV.

    Expected input columns (adapt to actual schema):
        brand_name, generic_name, indication, price, unit, manufacturer

    Cleaning steps:
        1. Normalize column names to snake_case.
        2. Rename 'price' to 'price_bdt'.
        3. Convert price_bdt to float; drop rows with invalid prices.
        4. Strip and lowercase text columns.
        5. Drop rows with missing generic_name or indication.
        6. Save to OUTPUT_PATHS['dgda_clean'].

    TODO (collaborator): Adapt column mapping to match the actual DGDA CSV schema.
    """
    logger.info("Cleaning DGDA dataset from %s", filepath)
    if not os.path.exists(filepath):
        logger.error("DGDA file not found: %s", filepath)
        return

    raise NotImplementedError("clean_dgda_dataset() not yet implemented.")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main(step: str = "all"):
    """
    Run all or a specific data processing step.

    Args:
        step: 'all' | 'symptoms' | 'xgb' | 'faiss' | 'dgda'
    """
    logger.info("=== HealthMax Data Processing Pipeline ===")
    logger.info("Step: %s", step)

    if step in ("all", "symptoms", "xgb", "faiss"):
        df = load_symptoms_disease(FILE_PATHS["symptoms_disease"])
        if df.empty:
            logger.error("Symptoms dataset failed to load. Aborting.")
            return
        symptom_vocab, disease_labels = build_vocabularies(df)

    if step in ("all", "xgb"):
        train_xgboost_classifier(df, symptom_vocab, disease_labels)

    if step in ("all", "faiss"):
        build_faiss_index(df)

    if step in ("all", "dgda"):
        clean_dgda_dataset(FILE_PATHS["dgda_medicines"])

    logger.info("=== Data processing complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HealthMax data processor")
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "symptoms", "xgb", "faiss", "dgda"],
        help="Which processing step to run.",
    )
    args = parser.parse_args()
    main(step=args.step)
