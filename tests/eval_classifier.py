"""
HealthMax — Automated Classifier Evaluation
Usage: python tests/eval_classifier.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import xgboost as xgb


def run_evaluation():
    print("=" * 60)
    print("HealthMax Classifier Evaluation")
    print("=" * 60)

    # Load model artifacts
    model = xgb.XGBClassifier()
    model.load_model("models/disease_classifier.json")

    with open("models/label_encoder.json") as f:
        label_classes = json.load(f)
    with open("models/symptom_list.json") as f:
        symptom_list = json.load(f)

    # Load test data
    try:
        df = pd.read_csv("data/raw/symptoms_disease_bangla.csv")
    except FileNotFoundError:
        print("[!] Dataset not found. Generating mock test cases...")
        df = _mock_test_cases(symptom_list, label_classes)

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    le = LabelEncoder()
    le.classes_ = np.array(label_classes)
    y = le.transform(df["disease"].fillna("Unknown"))

    X = np.zeros((len(df), len(symptom_list)), dtype=np.float32)
    for i, symptoms_str in enumerate(df["symptoms"].fillna("")):
        for s in str(symptoms_str).split(","):
            s = s.strip()
            if s in symptom_list:
                X[i, symptom_list.index(s)] = 1.0

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n📊 Macro F1 Score: {macro_f1:.4f}")
    status = "✅ PASS" if macro_f1 >= 0.80 else ("⚠️ BELOW TARGET" if macro_f1 >= 0.70 else "❌ FAIL")
    print(f"   Target: > 0.80 | Status: {status}")

    print("\n--- Per-Class Report ---")
    print(classification_report(y_test, y_pred, target_names=label_classes))

    # Check fail condition: any high-prevalence disease below F1=0.60
    report = classification_report(y_test, y_pred, target_names=label_classes, output_dict=True)
    critical_diseases = ["Dengue", "Typhoid", "Pneumonia", "Malaria", "Gastroenteritis"]
    for disease in critical_diseases:
        if disease in report and report[disease]["f1-score"] < 0.60:
            print(f"❌ CRITICAL FAIL: {disease} F1 = {report[disease]['f1-score']:.3f} (below 0.60)")

    return macro_f1


def _mock_test_cases(symptom_list, label_classes):
    rows = []
    for label in label_classes[:10]:
        for _ in range(5):
            rows.append({"disease": label, "symptoms": symptom_list[0]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_evaluation()
