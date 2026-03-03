"""
HealthMax Dataset Processor
Run this script ONCE before starting the backend.
It builds the FAISS index and trains the XGBoost classifier.

Usage:
    python data/process_datasets.py
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────
# STEP 1: Build FAISS Index from disease-symptom dataset
# ─────────────────────────────────────────────────────
def build_faiss_index():
    import faiss
    from sentence_transformers import SentenceTransformer

    print("[Step 1] Building FAISS index...")

    # Load dataset (Bangla Symptoms-Disease Dataset)
    csv_path = DATA_DIR / "symptoms_disease_bangla.csv"
    if not csv_path.exists():
        print(f"  [!] Dataset not found at {csv_path}. Using mock data.")
        df = _create_mock_disease_data()
    else:
        df = pd.read_csv(csv_path)

    # Prepare text for embedding
    records = []
    texts = []
    for _, row in df.iterrows():
        symptom_text = str(row.get("symptoms", "")) + " " + str(row.get("disease", ""))
        texts.append(symptom_text)
        records.append({
            "disease": str(row.get("disease", "")),
            "symptoms": str(row.get("symptoms", "")),
            "urgency": str(row.get("urgency", "URGENT")),
            "specialist": str(row.get("specialist", "General Physician")),
            "facility": str(row.get("facility", "Upazila Health Complex"))
        })

    print(f"  Encoding {len(texts)} disease records...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(MODELS_DIR / "disease_rag.index"))
    with open(MODELS_DIR / "disease_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"  ✅ FAISS index built: {index.ntotal} records → models/disease_rag.index")


# ─────────────────────────────────────────────────────
# STEP 2: Train XGBoost Classifier
# ─────────────────────────────────────────────────────
def train_xgboost_classifier():
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score

    print("\n[Step 2] Training XGBoost disease classifier...")

    csv_path = DATA_DIR / "symptoms_disease_bangla.csv"
    if not csv_path.exists():
        print(f"  [!] Dataset not found. Using mock data.")
        df = _create_mock_disease_data()
    else:
        df = pd.read_csv(csv_path)

    # Build symptom vocabulary
    all_symptoms = set()
    for symptoms_str in df["symptoms"].dropna():
        for s in str(symptoms_str).split(","):
            all_symptoms.add(s.strip())
    symptom_list = sorted(list(all_symptoms))

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["disease"].fillna("Unknown"))

    # Binarize features
    X = np.zeros((len(df), len(symptom_list)), dtype=np.float32)
    for i, symptoms_str in enumerate(df["symptoms"].fillna("")):
        for symptom in str(symptoms_str).split(","):
            symptom = symptom.strip()
            if symptom in symptom_list:
                j = symptom_list.index(symptom)
                X[i, j] = 1.0

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n  📊 Macro F1 Score: {macro_f1:.4f}")
    if macro_f1 < 0.70:
        print("  ⚠️  WARNING: F1 below 0.70 target. Consider more training data.")

    # Save model and metadata
    model.save_model(str(MODELS_DIR / "disease_classifier.json"))
    with open(MODELS_DIR / "label_encoder.json", "w", encoding="utf-8") as f:
        json.dump(le.classes_.tolist(), f, ensure_ascii=False)
    with open(MODELS_DIR / "symptom_list.json", "w", encoding="utf-8") as f:
        json.dump(symptom_list, f, ensure_ascii=False)

    print("  ✅ Model saved → models/disease_classifier.json")
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_[:20]))


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
    build_faiss_index()
    train_xgboost_classifier()
    print("\n✅ All done! You can now start the backend.")
    print("   Run: uvicorn backend.main:app --host 0.0.0.0 --port 8000")
