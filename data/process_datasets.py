"""
HealthMax Dataset Processor
Run this script before starting the backend.
It builds the FAISS index and trains the XGBoost classifier.

Usage:
    python data/process_datasets.py
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RAG_VECTORIZER_PATH = MODELS_DIR / "disease_rag_vectorizer.pkl"
RAG_CONFIG_PATH = MODELS_DIR / "rag_config.json"
TRAINING_SUMMARY_PATH = MODELS_DIR / "training_summary.json"
DEFAULT_RAG_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _configure_output_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except ValueError:
                # Some hosts expose non-reconfigurable wrapped streams.
                pass


_configure_output_streams()


def _get_runtime_acceleration() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_available": False,
        "cuda_device_name": None,
        "xgboost_device": "cpu",
        "faiss_device": "cpu",
    }

    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["xgboost_device"] = "cuda"
    except Exception as error:
        print(f"  [WARN] CUDA detection failed. Falling back to CPU. ({error})")

    return info


def _prepare_transformers_backend() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def _find_dataset_path() -> Path | None:
    for candidate in (DATA_DIR / "Symptoms.csv", DATA_DIR / "symptoms_disease_bangla.csv"):
        if candidate.exists():
            return candidate
    return None


def _normalize_feature_name(symptom_name: str) -> str:
    return " ".join(str(symptom_name).strip().split())


def _infer_metadata(disease_name: str) -> Dict[str, str]:
    lowered = disease_name.casefold()

    if any(keyword in lowered for keyword in ("heart attack", "স্ট্রোক", "stroke", "cholera", "কলেরা")):
        return {
            "urgency": "EMERGENCY",
            "specialist": "General Physician",
            "facility": "District Hospital",
        }

    return {
        "urgency": "URGENT",
        "specialist": "General Physician",
        "facility": "Upazila Health Complex",
    }


def _is_active(value: Any) -> bool:
    try:
        return int(float(value)) == 1
    except (TypeError, ValueError):
        return False


def _extract_symptoms(symptoms_text: str) -> List[str]:
    return [part.strip() for part in symptoms_text.split(",") if part.strip()]


def _matrix_to_long_form(df: pd.DataFrame) -> pd.DataFrame:
    symptom_columns = [column for column in df.columns if column != "prognosis"]
    records: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        disease_name = str(row.get("prognosis", "")).strip()
        active_symptoms = [
            _normalize_feature_name(column)
            for column in symptom_columns
            if _is_active(row.get(column))
        ]
        metadata = _infer_metadata(disease_name)
        records.append(
            {
                "disease": disease_name,
                "symptoms": ",".join(active_symptoms),
                "urgency": metadata["urgency"],
                "specialist": metadata["specialist"],
                "facility": metadata["facility"],
            }
        )

    return pd.DataFrame(records)


def _load_disease_dataframe() -> pd.DataFrame:
    csv_path = _find_dataset_path()
    if csv_path is None:
        print("  [!] No real dataset found in data/raw. Using mock data.")
        return _create_mock_disease_data()

    print(f"  Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    if "prognosis" in df.columns:
        return _matrix_to_long_form(df)

    if {"disease", "symptoms"}.issubset(df.columns):
        standardized = df.copy()
        standardized["disease"] = standardized["disease"].fillna("Unknown").astype(str)
        standardized["symptoms"] = standardized["symptoms"].fillna("").astype(str)

        if "urgency" not in standardized.columns:
            standardized["urgency"] = standardized["disease"].map(
                lambda disease_name: _infer_metadata(str(disease_name))["urgency"]
            )
        else:
            standardized["urgency"] = standardized["urgency"].fillna("URGENT").astype(str)

        if "specialist" not in standardized.columns:
            standardized["specialist"] = "General Physician"
        else:
            standardized["specialist"] = standardized["specialist"].fillna("General Physician").astype(str)

        if "facility" not in standardized.columns:
            standardized["facility"] = standardized["urgency"].map(
                lambda urgency: "District Hospital" if str(urgency).upper() == "EMERGENCY" else "Upazila Health Complex"
            )
        else:
            standardized["facility"] = standardized["facility"].fillna("Upazila Health Complex").astype(str)

        return standardized[["disease", "symptoms", "urgency", "specialist", "facility"]]

    print("  [!] Dataset format not recognized. Using mock data.")
    return _create_mock_disease_data()


def _build_retrieval_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        disease_name = str(row.get("disease", "Unknown")).strip()
        disease_entry = grouped.setdefault(
            disease_name,
            {
                "disease": disease_name,
                "symptoms": set(),
                "urgency": str(row.get("urgency", "URGENT")),
                "specialist": str(row.get("specialist", "General Physician")),
                "facility": str(row.get("facility", "Upazila Health Complex")),
            },
        )
        disease_entry["symptoms"].update(_extract_symptoms(str(row.get("symptoms", ""))))

    records: List[Dict[str, Any]] = []
    for disease_name, record in grouped.items():
        symptoms = sorted(cast(set[str], record["symptoms"]))
        record["symptoms"] = symptoms
        record["text_representation"] = f"Disease: {disease_name}. Symptoms: {', '.join(symptoms)}"
        records.append(record)

    return records


def _l2_normalize_dense(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _save_training_summary(summary: Dict[str, Any]) -> None:
    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────
# STEP 1: Build FAISS Index from disease-symptom dataset
# ─────────────────────────────────────────────────────
def build_faiss_index() -> Dict[str, Any]:
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("[Step 1] Building FAISS index...")
    runtime = _get_runtime_acceleration()

    df = _load_disease_dataframe()
    records = _build_retrieval_records(df)
    texts = [str(record["text_representation"]) for record in records]

    rag_backend = "tfidf"
    rag_embedding_model = None
    rag_embedding_device = "cpu"
    rag_vectorizer_path: str | None = None

    try:
        _prepare_transformers_backend()
        from sentence_transformers import SentenceTransformer

        rag_embedding_device = "cuda" if runtime["cuda_available"] else "cpu"
        print(
            f"  Encoding {len(texts)} disease records with {DEFAULT_RAG_EMBEDDING_MODEL} "
            f"on {rag_embedding_device}..."
        )
        embedding_model = SentenceTransformer(DEFAULT_RAG_EMBEDDING_MODEL, device=rag_embedding_device)
        embeddings = embedding_model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32), dtype=np.float32)
        rag_backend = "sentence-transformers"
        rag_embedding_model = DEFAULT_RAG_EMBEDDING_MODEL
    except Exception as error:
        print(f"  [WARN] Sentence-transformer RAG failed. Falling back to TF-IDF. ({error})")
        print(f"  Vectorizing {len(texts)} disease records with TF-IDF...")
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=4096)
        sparse_embeddings = vectorizer.fit_transform(texts)
        dense_embeddings = np.asarray(cast(Any, sparse_embeddings).toarray(), dtype=np.float32)
        embeddings = np.ascontiguousarray(_l2_normalize_dense(dense_embeddings), dtype=np.float32)
        rag_vectorizer_path = str(RAG_VECTORIZER_PATH)
        with open(RAG_VECTORIZER_PATH, "wb") as file:
            pickle.dump(vectorizer, file)

    # Cosine-style search via inner product on normalized vectors.
    dimension = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dimension)

    if runtime["cuda_available"] and hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
        try:
            print(f"  Using FAISS GPU indexing on {runtime['cuda_device_name']}...")
            gpu_resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
            cast(Any, gpu_index).add(embeddings)
            index = faiss.index_gpu_to_cpu(gpu_index)
            runtime["faiss_device"] = "cuda"
        except Exception as error:
            print(f"  [WARN] FAISS GPU indexing unavailable. Falling back to CPU. ({error})")
            index = cpu_index
            cast(Any, index).add(embeddings)
    else:
        index = cpu_index
        cast(Any, index).add(embeddings)

    faiss.write_index(index, str(MODELS_DIR / "disease_rag.index"))
    with open(MODELS_DIR / "disease_records.json", "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    with open(RAG_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "backend": rag_backend,
                "embedding_model": rag_embedding_model,
                "embedding_device": rag_embedding_device,
                "vectorizer_path": rag_vectorizer_path,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"  [OK] FAISS index built: {index.ntotal} records -> models/disease_rag.index")
    return {
        "num_retrieval_records": len(records),
        "rag_vector_dim": int(dimension),
        "rag_backend": rag_backend,
        "rag_embedding_model": rag_embedding_model,
        "rag_embedding_device": rag_embedding_device,
        "rag_vectorizer_path": rag_vectorizer_path,
        "faiss_device": runtime["faiss_device"],
    }


# ─────────────────────────────────────────────────────
# STEP 2: Train XGBoost Classifier
# ─────────────────────────────────────────────────────
def train_xgboost_classifier() -> Dict[str, Any]:
    import xgboost as xgb
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    print("\n[Step 2] Training XGBoost disease classifier...")
    runtime = _get_runtime_acceleration()

    df = _load_disease_dataframe()

    # Build symptom vocabulary
    all_symptoms = set()
    for symptoms_str in df["symptoms"].dropna():
        for symptom in _extract_symptoms(str(symptoms_str)):
            all_symptoms.add(symptom)
    symptom_list = sorted(list(all_symptoms))
    symptom_to_index = {symptom: idx for idx, symptom in enumerate(symptom_list)}

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["disease"].fillna("Unknown"))

    # Binarize features
    X = np.zeros((len(df), len(symptom_list)), dtype=np.float32)
    for i, symptoms_str in enumerate(df["symptoms"].fillna("")):
        for symptom in _extract_symptoms(str(symptoms_str)):
            j = symptom_to_index.get(symptom)
            if j is not None:
                X[i, j] = 1.0

    split_mode = "stratified"
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        split_mode = "random"
        print("  [!] Stratified split unavailable for the current label distribution. Falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print(f"  Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    model_kwargs: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "device": runtime["xgboost_device"],
    }
    if runtime["xgboost_device"] == "cuda":
        print(f"  Using XGBoost GPU training on {runtime['cuda_device_name']}...")

    model = xgb.XGBClassifier(**model_kwargs)
    trained_device = runtime["xgboost_device"]
    try:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    except xgb.core.XGBoostError as error:
        if runtime["xgboost_device"] == "cuda":
            print(f"  [WARN] XGBoost GPU training failed. Falling back to CPU. ({error})")
            model_kwargs["device"] = "cpu"
            model = xgb.XGBClassifier(**model_kwargs)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            trained_device = "cpu"
        else:
            raise

    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n  Macro F1 Score: {macro_f1:.4f}")
    if macro_f1 < 0.70:
        print("  [WARN] F1 below 0.70 target. Consider more training data.")

    model.save_model(str(MODELS_DIR / "disease_classifier.json"))
    with open(MODELS_DIR / "label_encoder.json", "w", encoding="utf-8") as file:
        json.dump(le.classes_.tolist(), file, ensure_ascii=False, indent=2)
    with open(MODELS_DIR / "symptom_list.json", "w", encoding="utf-8") as file:
        json.dump(symptom_list, file, ensure_ascii=False, indent=2)

    print("  [OK] Model saved -> models/disease_classifier.json")
    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(le.classes_))),
        target_names=[str(label) for label in le.classes_],
        zero_division=0,
    )
    print(f"\n{report_text}")

    return {
        "macro_f1": float(macro_f1),
        "num_diseases": int(len(le.classes_)),
        "num_symptoms": int(len(symptom_list)),
        "num_rows": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "split_mode": split_mode,
        "xgboost_device": trained_device,
        "cuda_available": bool(runtime["cuda_available"]),
        "cuda_device_name": runtime["cuda_device_name"],
    }


# ─────────────────────────────────────────────────────
# Mock data builder (used when real dataset is absent)
# ─────────────────────────────────────────────────────
def _create_mock_disease_data() -> pd.DataFrame:
    rows = [
        {"disease": "Dengue", "symptoms": "জ্বর,মাথাব্যথা,চোখ লাল,গা ব্যথা", "urgency": "URGENT", "specialist": "Medicine", "facility": "Upazila Health Complex"},
        {"disease": "Typhoid", "symptoms": "জ্বর,পেটব্যথা,দুর্বলতা,মাথাব্যথা", "urgency": "URGENT", "specialist": "Medicine", "facility": "Upazila Health Complex"},
        {"disease": "Pneumonia", "symptoms": "কাশি,শ্বাসকষ্ট,জ্বর,বুকে ব্যথা", "urgency": "URGENT", "specialist": "Respiratory", "facility": "District Hospital"},
        {"disease": "Gastroenteritis", "symptoms": "বমি,ডায়রিয়া,পেটব্যথা", "urgency": "URGENT", "specialist": "Medicine", "facility": "Upazila Health Complex"},
        {"disease": "Malaria", "symptoms": "জ্বর,কাঁপুনি,মাথাব্যথা,ঘাম", "urgency": "URGENT", "specialist": "Medicine", "facility": "Upazila Health Complex"},
        {"disease": "Upper Respiratory Infection", "symptoms": "গলাব্যথা,সর্দি,হালকা জ্বর,কাশি", "urgency": "SELF-CARE", "specialist": "General", "facility": "Community Clinic"},
        {"disease": "Hypertension", "symptoms": "মাথাব্যথা,মাথা ঘোরা,বুকে ব্যথা", "urgency": "URGENT", "specialist": "Cardiology", "facility": "District Hospital"},
        {"disease": "Diabetes", "symptoms": "বেশি পানি পান,বার বার প্রস্রাব,দুর্বলতা", "urgency": "URGENT", "specialist": "Endocrinology", "facility": "Upazila Health Complex"},
        {"disease": "Cholera", "symptoms": "তীব্র ডায়রিয়া,বমি,পানিশূন্যতা", "urgency": "EMERGENCY", "specialist": "Medicine", "facility": "District Hospital"},
        {"disease": "Asthma", "symptoms": "শ্বাসকষ্ট,শোঁ শোঁ শব্দ,কাশি", "urgency": "URGENT", "specialist": "Respiratory", "facility": "Upazila Health Complex"},
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("HealthMax Dataset Processor")
    print("=" * 60)
    rag_summary = build_faiss_index()
    classifier_summary = train_xgboost_classifier()
    full_summary = {**classifier_summary, **rag_summary}
    _save_training_summary(full_summary)
    print("\n[OK] All done! You can now start the backend.")
    print("   Run: uvicorn backend.main:app --host 0.0.0.0 --port 8000")
    print(f"   Summary saved: {TRAINING_SUMMARY_PATH}")
