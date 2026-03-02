"""
HealthMax — Automated Classifier & Clinical Safety Evaluation

Runs three evaluation phases:

Phase 1: Unit test — all 172 symptoms individually → check disease mapping → compute Macro F1
Phase 2: Clinical vignette test — load tests/clinical_vignettes.csv → run rule engine → tally Pass/Sub/Unsafe
Phase 3: Summary report

Usage:
    python tests/eval_classifier.py               # Runs all phases
    python tests/eval_classifier.py --phase 1     # Phase 1 only
    python tests/eval_classifier.py --phase 2     # Phase 2 only

Collaborator instructions:
    - Populate tests/clinical_vignettes.csv with 50 scenarios before running Phase 2.
    - Target: Macro F1 > 0.80, 0 UNSAFE vignette outputs.
    - Run this before every pitch rehearsal.
"""

import os
import sys
import json
import argparse
import logging

import pandas as pd
import numpy as np

# Allow importing backend modules from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("healthmax.eval")

VIGNETTES_CSV = os.path.join(os.path.dirname(__file__), "clinical_vignettes.csv")
PASS_THRESHOLD_UNSAFE = 0   # Max allowed unsafe outputs (must be 0)
F1_TARGET = 0.80

# ---------------------------------------------------------------------------
# Phase 1: Classifier unit test
# ---------------------------------------------------------------------------

def phase1_classifier_eval():
    """
    Load the trained XGBoost classifier and evaluate it on the held-out test split.

    Steps:
        1. Re-load the full symptoms-disease dataset.
        2. Apply the same 80/20 train-test split (use same random_state=42).
        3. Run predict_diseases() on test features.
        4. Compute classification_report (per-class F1) and macro F1.
        5. Assert macro F1 > F1_TARGET.
        6. Flag any single disease class with F1 < 0.60 as a warning.

    TODO (collaborator):
        - Import predict_diseases, symptoms_to_feature_vector from classifier.py
        - Load the test split using sklearn.model_selection.train_test_split with same seed
        - Use sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)
    """
    logger.info("=== Phase 1: Classifier Evaluation ===")

    try:
        from classifier import predict_diseases, load_model  # type: ignore
    except ImportError as e:
        logger.error("Cannot import classifier: %s", e)
        return None

    # TODO: Load dataset and split — same procedure as in process_datasets.py
    # TODO: Run batch predictions
    # TODO: Compute and print classification_report

    logger.warning("Phase 1 NOT YET IMPLEMENTED — complete process_datasets.py and classifier.py first.")
    return None


# ---------------------------------------------------------------------------
# Phase 2: Clinical vignette safety test
# ---------------------------------------------------------------------------

def phase2_vignette_test():
    """
    Load 50 clinical vignettes from CSV and run the clinical rule engine on each.

    CSV columns expected:
        id, input_bn (Bangla symptom text), expected_urgency, notes

    Output:
        - PASS / SUBOPTIMAL / UNSAFE classification per vignette
        - Total counts
        - Lists any UNSAFE outputs to the console (must be 0)

    TODO (collaborator):
        - Import apply_clinical_rules and validate_vignette from rules.py
        - Also run predict_diseases on each vignette for end-to-end context
    """
    logger.info("=== Phase 2: Clinical Vignette Safety Test ===")

    if not os.path.exists(VIGNETTES_CSV):
        logger.error("Vignettes CSV not found: %s. Create it first.", VIGNETTES_CSV)
        return None

    try:
        from rules import validate_vignette  # type: ignore
    except ImportError as e:
        logger.error("Cannot import rules: %s", e)
        return None

    df = pd.read_csv(VIGNETTES_CSV, encoding="utf-8")
    required_cols = {"id", "input_bn", "expected_urgency"}
    if not required_cols.issubset(df.columns):
        logger.error("CSV missing columns. Required: %s", required_cols)
        return None

    results = []
    unsafe_cases = []
    suboptimal_cases = []

    for _, row in df.iterrows():
        vignette_id = row["id"]
        text = str(row["input_bn"])
        expected = str(row["expected_urgency"]).strip().upper()

        result = validate_vignette(text, expected)
        outcome = "PASS" if result["pass"] else ("UNSAFE" if result["unsafe"] else "SUBOPTIMAL")

        results.append({
            "id": vignette_id,
            "input": text[:60],
            "expected": expected,
            "actual": result["actual"],
            "outcome": outcome,
            "triggered_by": result["triggered_by"],
        })

        if outcome == "UNSAFE":
            unsafe_cases.append(results[-1])
        elif outcome == "SUBOPTIMAL":
            suboptimal_cases.append(results[-1])

    # Summary
    pass_count = sum(1 for r in results if r["outcome"] == "PASS")
    total = len(results)

    logger.info("Results: %d PASS, %d SUBOPTIMAL, %d UNSAFE / %d total",
                pass_count, len(suboptimal_cases), len(unsafe_cases), total)

    if unsafe_cases:
        logger.error("❌ UNSAFE OUTPUTS DETECTED — fix before demo:")
        for uc in unsafe_cases:
            logger.error("  [%s] %s → expected %s, got %s", uc["id"], uc["input"], uc["expected"], uc["actual"])
    else:
        logger.info("✅ 0 UNSAFE outputs — safety target met.")

    if len(suboptimal_cases) > 5:
        logger.warning("⚠️ %d SUBOPTIMAL outputs (target: <5). Review these.", len(suboptimal_cases))

    return {
        "total": total,
        "pass": pass_count,
        "suboptimal": len(suboptimal_cases),
        "unsafe": len(unsafe_cases),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HealthMax evaluation suite")
    parser.add_argument("--phase", type=int, choices=[1, 2], help="Run only a specific phase.")
    args = parser.parse_args()

    if args.phase == 1:
        phase1_classifier_eval()
    elif args.phase == 2:
        phase2_vignette_test()
    else:
        phase1_classifier_eval()
        phase2_vignette_test()


if __name__ == "__main__":
    main()
