"""
HealthMax — Layer 6: DGDA Drug Lookup
Returns the cheapest generic medicine options for a given disease/indication
from the Bangladesh DGDA 50,000-medicine registry dataset.

Dataset: Mendeley 3x5gsr2jm3.1 (Bangladesh DGDA Medicine Registry)
         Columns expected: brand_name, generic_name, indication, price_bdt, unit, manufacturer

Collaborator instructions:
    - Process the raw DGDA dataset in data/process_datasets.py and save a cleaned CSV.
    - Implement lookup_drugs() — the primary function called by main.py.
    - Return the top 3 cheapest generics for the given disease indication.
    - Mark items below ৳5 per unit as "সাশ্রয়ী" (affordable) — FLEX feature.
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger("healthmax.dgda")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DGDA_CSV_PATH = os.environ.get(
    "DGDA_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "dgda_medicines_clean.csv"),
)

# Price threshold for "affordable" flag (BDT per unit)
AFFORDABLE_THRESHOLD_BDT = 5.0
MAX_RESULTS = 3

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_dgda_df: Optional[pd.DataFrame] = None


def load_dgda_data() -> Optional[pd.DataFrame]:
    """
    Load and index the cleaned DGDA medicine dataset into memory.

    Expected CSV columns:
        brand_name, generic_name, indication, price_bdt, unit, manufacturer

    TODO (collaborator):
        1. Read the CSV with pd.read_csv(DGDA_CSV_PATH, encoding='utf-8').
        2. Normalize text columns: lowercase, strip whitespace.
        3. Create a combined 'indication_text' column merging generic_name + indication
           for fuzzy search.
        4. Cache in _dgda_df global.
    """
    global _dgda_df

    if _dgda_df is not None:
        return _dgda_df

    if not os.path.exists(DGDA_CSV_PATH):
        logger.warning("DGDA CSV not found at %s. Run process_datasets.py first.", DGDA_CSV_PATH)
        return None

    try:
        _dgda_df = pd.read_csv(DGDA_CSV_PATH, encoding="utf-8")
        # Normalize
        for col in ["brand_name", "generic_name", "indication"]:
            if col in _dgda_df.columns:
                _dgda_df[col] = _dgda_df[col].astype(str).str.strip().str.lower()
        logger.info("DGDA dataset loaded: %d records", len(_dgda_df))
    except Exception as e:
        logger.error("Failed to load DGDA data: %s", e)
        _dgda_df = None

    return _dgda_df


# ---------------------------------------------------------------------------
# Core lookup
# ---------------------------------------------------------------------------

def lookup_drugs(disease_name: str, max_results: int = MAX_RESULTS) -> list:
    """
    Find the cheapest generic medicines for a given disease or indication.

    Args:
        disease_name: Disease or indication string (English or Bangla).
                      This is matched against the 'indication' and 'generic_name' columns.
        max_results:  Maximum number of drug entries to return.

    Returns:
        List of dicts, sorted by price (cheapest first):
            [
                {
                    'generic_name':  str,   # Generic medicine name
                    'brand_example': str,   # One example brand name
                    'price_bdt':     float, # Price in BDT per unit
                    'unit':          str,   # 'tablet', 'ml', etc.
                    'affordable':    bool,  # True if price < AFFORDABLE_THRESHOLD_BDT
                },
                ...
            ]
        Returns a safe fallback message list if no match found or data not loaded.

    TODO (collaborator):
        1. Load data via load_dgda_data().
        2. Filter rows where 'indication' contains disease_name (case-insensitive).
        3. If no exact match, try fuzzy matching with rapidfuzz (threshold: 70).
        4. Sort filtered rows by price_bdt ascending.
        5. Group by generic_name and take cheapest price per generic.
        6. Return top max_results records with the affordable flag set.
    """
    df = load_dgda_data()

    if df is None or disease_name.strip() == "":
        logger.warning("No DGDA data or empty disease name; returning fallback.")
        return _fallback_drugs()

    try:
        disease_lower = disease_name.strip().lower()

        # Simple substring match — collaborator should upgrade to fuzzy
        mask = df["indication"].str.contains(disease_lower, na=False, case=False)
        matched = df[mask].copy()

        if matched.empty:
            logger.info("No DGDA match for '%s'; returning fallback.", disease_name)
            return _fallback_drugs()

        # Sort by price
        if "price_bdt" in matched.columns:
            matched = matched.sort_values("price_bdt")

        # Deduplicate by generic_name — keep cheapest
        matched = matched.drop_duplicates(subset=["generic_name"], keep="first")
        matched = matched.head(max_results)

        results = []
        for _, row in matched.iterrows():
            price = float(row.get("price_bdt", 0.0))
            results.append({
                "generic_name":  row.get("generic_name", ""),
                "brand_example": row.get("brand_name", ""),
                "price_bdt":     price,
                "unit":          row.get("unit", "tablet"),
                "affordable":    price < AFFORDABLE_THRESHOLD_BDT,
            })
        return results

    except Exception as e:
        logger.exception("Drug lookup failed: %s", e)
        return _fallback_drugs()


def _fallback_drugs() -> list:
    """
    Return a safe fallback drug suggestion when lookup fails.
    Uses Paracetamol as the universal symptomatic fallback.
    """
    return [
        {
            "generic_name":  "Paracetamol (প্যারাসিটামল)",
            "brand_example": "Napa / Ace",
            "price_bdt":     2.50,
            "unit":          "tablet",
            "affordable":    True,
        }
    ]


# ---------------------------------------------------------------------------
# Dev test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dgda_data()
    diseases = ["dengue", "typhoid", "gastroenteritis"]
    for d in diseases:
        drugs = lookup_drugs(d)
        print(f"\n{d}:")
        for drug in drugs:
            flag = "সাশ্রয়ী ✅" if drug["affordable"] else ""
            print(f"  {drug['generic_name']} ({drug['brand_example']}) — ৳{drug['price_bdt']}/{drug['unit']} {flag}")
