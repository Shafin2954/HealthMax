import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_dgda_df: pd.DataFrame | None = None

_CANONICAL_COLUMNS = [
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

DATA_PATH_CANDIDATES = [
    os.getenv("DGDA_CSV_PATH", "").strip(),
    "assets/medicine.csv",
    "healthmax-ai-assistant/src/data/medicine.csv",
    "data/raw/dgda_medicines.csv",
]

PRICE_PATTERN = re.compile(r"৳\s*([0-9]+(?:\.[0-9]+)?)")


def _resolve_data_path() -> Path | None:
    for candidate in DATA_PATH_CANDIDATES:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    return None


def _extract_price_bdt(*values: object) -> float:
    for value in values:
        text = str(value or "")
        match = PRICE_PATTERN.search(text)
        if match:
            return float(match.group(1))
    return float("nan")


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).lower().strip().replace(" ", "_") for column in normalized.columns]

    if {"brand_name", "generic_name"}.issubset(normalized.columns):
        pass
    elif normalized.shape[1] >= len(_CANONICAL_COLUMNS):
        normalized = normalized.iloc[:, : len(_CANONICAL_COLUMNS)].copy()
        normalized.columns = _CANONICAL_COLUMNS
    else:
        raise ValueError("Medicine dataset does not match an expected layout.")

    text_columns = [
        "brand_name",
        "generic_name",
        "dosage_form",
        "manufacturer",
        "price_info",
        "pack_info",
    ]
    for column in text_columns:
        if column not in normalized.columns:
            normalized[column] = ""
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
    return normalized


def _load_dgda_data() -> pd.DataFrame:
    global _dgda_df
    if _dgda_df is None:
        data_path = _resolve_data_path()
        if data_path is None:
            print("[DGDA] Dataset not found. Using mock data for development.")
            _dgda_df = _normalize_dataframe(_get_mock_data())
        else:
            print(f"[DGDA] Loading medicine dataset from {data_path}...")
            try:
                raw_df = pd.read_csv(data_path)
                _dgda_df = _normalize_dataframe(raw_df)
            except ValueError:
                raw_df = pd.read_csv(data_path, header=None, names=_CANONICAL_COLUMNS)
                _dgda_df = _normalize_dataframe(raw_df)
            print(f"[DGDA] Loaded {len(_dgda_df)} medicine records.")
    return _dgda_df


def _get_mock_data() -> pd.DataFrame:
    """Fallback mock data for development without the real medicine dataset."""
    return pd.DataFrame(
        [
            {
                "brand_name": "Napa",
                "dosage_form": "Tablet",
                "generic_name": "Paracetamol",
                "manufacturer": "Beximco",
                "price_info": "Unit Price: ৳ 1.50",
                "pack_info": "",
            },
            {
                "brand_name": "Acme's ORS",
                "dosage_form": "Powder",
                "generic_name": "Oral Rehydration Salt [Powder]",
                "manufacturer": "ACME Laboratories Ltd.",
                "price_info": "Unit Price: ৳ 5.00",
                "pack_info": "",
            },
            {
                "brand_name": "Bimuty",
                "dosage_form": "Tablet",
                "generic_name": "Zinc Sulfate Monohydrate",
                "manufacturer": "Drug International",
                "price_info": "Unit Price: ৳ 2.00",
                "pack_info": "",
            },
            {
                "brand_name": "Adtrizin",
                "dosage_form": "Tablet",
                "generic_name": "Cetirizine Hydrochloride",
                "manufacturer": "Team Pharmaceuticals Ltd.",
                "price_info": "Unit Price: ৳ 2.50",
                "pack_info": "",
            },
            {
                "brand_name": "Aire",
                "dosage_form": "Tablet",
                "generic_name": "Levosalbutamol (Oral)",
                "manufacturer": "Delta Pharma Ltd.",
                "price_info": "Unit Price: ৳ 1.70",
                "pack_info": "",
            },
        ]
    )


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


# Use supportive-care generic hints against the real medicine CSV.
# This avoids pretending we have disease-indication labels when we do not.
DISEASE_GENERIC_HINTS = {
    "fever": ["paracetamol"],
    "dengue": ["paracetamol", "oral rehydration salt"],
    "malaria": ["paracetamol", "oral rehydration salt"],
    "typhoid": ["paracetamol", "oral rehydration salt"],
    "diarrhea": ["oral rehydration salt", "zinc sulfate"],
    "gastroenteritis": ["oral rehydration salt", "zinc sulfate"],
    "cholera": ["oral rehydration salt", "zinc sulfate"],
    "allergy": ["cetirizine", "desloratadine", "ketotifen"],
    "cold": ["paracetamol", "cetirizine"],
    "asthma": ["levosalbutamol", "salbutamol"],
    "pneumonia": ["paracetamol"],
}

FALLBACK_HINTS = ["paracetamol", "oral rehydration salt", "cetirizine"]
UNSAFE_DOSAGE_PATTERNS = ("injection", "infusion", "vial", "bag")


def _search_dataframe(df: pd.DataFrame, terms: List[str]) -> pd.DataFrame:
    matched_frames: List[pd.DataFrame] = []
    for term in terms:
        escaped_term = re.escape(term.casefold())
        mask = df["search_text"].str.contains(escaped_term, regex=True, na=False)
        current = df[mask].copy()
        if not current.empty:
            matched_frames.append(current)

    if not matched_frames:
        return df.iloc[0:0].copy()

    matched = pd.concat(matched_frames, ignore_index=True).drop_duplicates(
        subset=["brand_name", "generic_name", "dosage_form", "manufacturer", "price_info"]
    )
    safe_mask = ~matched["dosage_form"].str.casefold().str.contains(
        "|".join(UNSAFE_DOSAGE_PATTERNS),
        regex=True,
        na=False,
    )
    safe_matches = matched[safe_mask].copy()
    return safe_matches if not safe_matches.empty else matched


def lookup_drugs(disease_name: str, top_n: int = 3) -> List[Dict]:
    """
    Find low-cost supportive medicines from the real medicine CSV.
    Returns a short list for UI display, not a prescription.
    """
    df = _load_dgda_data()

    normalized_name = DISEASE_NAME_MAP.get(disease_name, disease_name).casefold()
    hints = DISEASE_GENERIC_HINTS.get(normalized_name, [])
    terms = hints or [normalized_name]

    matched = _search_dataframe(df, terms)
    if matched.empty:
        matched = _search_dataframe(df, FALLBACK_HINTS)

    matched = matched.copy()
    matched["price_bdt"] = matched["price_bdt"].replace({np.nan: np.inf})
    matched = matched.sort_values(["price_bdt", "brand_name"], ascending=[True, True]).head(top_n)

    results = []
    for _, row in matched.iterrows():
        price = row.get("price_bdt", np.inf)
        price_value = float(price) if np.isfinite(price) else 0.0
        dosage_form = row.get("dosage_form", "") or "unit"
        results.append(
            {
                "generic_name": row.get("generic_name", ""),
                "brand_example": row.get("brand_name", ""),
                "price_bdt": price_value,
                "unit": dosage_form,
                "manufacturer": row.get("manufacturer", ""),
                "affordable": price_value <= 10.0,
                "affordable_label": "সাশ্রয়ী 💚" if price_value <= 10.0 else "মধ্যম মূল্য",
            }
        )

    return results
