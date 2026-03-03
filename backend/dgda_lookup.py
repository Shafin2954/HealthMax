import os
import pandas as pd
from typing import List, Dict

_dgda_df = None
DGDA_DATA_PATH = "data/raw/dgda_medicines.csv"


def _load_dgda_data():
    global _dgda_df
    if _dgda_df is None:
        if not os.path.exists(DGDA_DATA_PATH):
            print("[DGDA] Dataset not found. Using mock data for development.")
            _dgda_df = _get_mock_data()
        else:
            print("[DGDA] Loading DGDA medicine dataset...")
            _dgda_df = pd.read_csv(DGDA_DATA_PATH, encoding="utf-8")
            # Normalize column names
            _dgda_df.columns = [c.lower().strip().replace(" ", "_") for c in _dgda_df.columns]
            print(f"[DGDA] Loaded {len(_dgda_df)} medicine records.")
    return _dgda_df


def _get_mock_data() -> pd.DataFrame:
    """Fallback mock data for development without the real DGDA dataset."""
    return pd.DataFrame([
        {"generic_name": "Paracetamol", "brand_name": "Napa", "indication": "Fever,Headache",
         "price_bdt": 1.5, "unit": "tablet", "manufacturer": "Beximco"},
        {"generic_name": "Amoxicillin", "brand_name": "Moxacil", "indication": "Pneumonia,Infection",
         "price_bdt": 3.0, "unit": "capsule", "manufacturer": "Square"},
        {"generic_name": "Metronidazole", "brand_name": "Amodis", "indication": "Gastroenteritis,Diarrhea",
         "price_bdt": 2.0, "unit": "tablet", "manufacturer": "ACI"},
        {"generic_name": "ORS Saline", "brand_name": "Gastrolyte", "indication": "Diarrhea,Dehydration",
         "price_bdt": 5.0, "unit": "sachet", "manufacturer": "Renata"},
        {"generic_name": "Cetirizine", "brand_name": "Alatrol", "indication": "Allergy,Cold",
         "price_bdt": 1.0, "unit": "tablet", "manufacturer": "Drug International"},
        {"generic_name": "Salbutamol", "brand_name": "Sultolin", "indication": "Asthma,Breathing difficulty",
         "price_bdt": 2.5, "unit": "tablet", "manufacturer": "GlaxoSmithKline"},
        {"generic_name": "Doxycycline", "brand_name": "Doxytet", "indication": "Typhoid,Malaria",
         "price_bdt": 4.0, "unit": "capsule", "manufacturer": "Opsonin"},
    ])


# Mapping from Bangla disease names to English for lookup
DISEASE_NAME_MAP = {
    "জ্বর": "Fever",
    "ডেঙ্গু": "Dengue",
    "ম্যালেরিয়া": "Malaria",
    "টাইফয়েড": "Typhoid",
    "নিউমোনিয়া": "Pneumonia",
    "ডায়রিয়া": "Diarrhea",
    "গ্যাস্ট্রোএন্টেরাইটিস": "Gastroenteritis",
    "হাঁপানি": "Asthma",
    "অ্যালার্জি": "Allergy",
    "ইউটিআই": "UTI",
    "সর্দি": "Cold",
}


def lookup_drugs(disease_name: str, top_n: int = 3) -> List[Dict]:
    """
    Query DGDA dataset to find cheapest generic medicines for a disease.
    Returns list of drug dicts: generic_name, brand_name, price_bdt, unit, affordable
    """
    df = _load_dgda_data()

    # Translate Bangla to English if needed
    search_term = DISEASE_NAME_MAP.get(disease_name, disease_name)

    # Search by indication (case-insensitive partial match)
    mask = df["indication"].str.contains(search_term, case=False, na=False)
    matched = df[mask].copy()

    if matched.empty:
        # Fallback: Paracetamol + ORS are always safe to recommend for general symptoms
        fallback = df[df["generic_name"].str.contains("Paracetamol|ORS", case=False, na=False)]
        matched = fallback.copy()

    # Sort by price ascending (cheapest first)
    matched = matched.sort_values("price_bdt", ascending=True).head(top_n)

    results = []
    for _, row in matched.iterrows():
        price = float(row.get("price_bdt", 0))
        results.append({
            "generic_name": row.get("generic_name", ""),
            "brand_example": row.get("brand_name", ""),
            "price_bdt": price,
            "unit": row.get("unit", "tablet"),
            "affordable": price <= 5.0,
            "affordable_label": "সাশ্রয়ী 💚" if price <= 5.0 else "মধ্যম মূল্য"
        })

    return results
