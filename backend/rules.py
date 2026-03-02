"""
HealthMax — Layer 5: Clinical Rule Engine
Hard-coded clinical safety override layer.

⚠️  CRITICAL — DO NOT LET AN LLM OR ML MODEL GENERATE THIS LOGIC.
    All rules here are hand-written and must be medically reviewed
    before deployment. Rules override ML output unconditionally.

Philosophy:
    - ML is advisory.
    - Rules are binding.
    - Any emergency pattern → ALWAYS returns EMERGENCY + 999 instruction.
    - No probabilistic reasoning for life-threatening symptoms.

Collaborator instructions:
    1. Add symptom keywords in Bangla EXACTLY as they would appear in NER output
       or transcribed text. Include common spelling variants.
    2. Run all 50 clinical vignettes through this module before Week 4.
    3. The target is 0 unsafe outputs. This module is the final safety net.
    4. Do NOT remove or weaken any emergency rule without medical justification.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("healthmax.rules")

# ---------------------------------------------------------------------------
# Emergency symptom keyword lists (Bangla)
# Add all phonetic/dialectal variants you can think of.
# ---------------------------------------------------------------------------

EMERGENCY_KEYWORDS = [
    # Chest pain / cardiac
    "বুকে ব্যথা", "বুকব্যথা", "বুক ব্যাথা", "বুকে চাপ",
    # Breathing difficulty
    "শ্বাস কষ্ট", "শ্বাস নিতে পারছি না", "শ্বাস নিতে পারছেন না",
    "শ্বাস বন্ধ", "নিঃশ্বাস নিতে পারছি না",
    # Unconsciousness
    "অজ্ঞান", "জ্ঞান হারিয়েছে", "জ্ঞান নেই", "অচেতন",
    # Seizure
    "খিঁচুনি", "মৃগী", "ফিট হয়েছে", "ফিট",
    # Heavy bleeding
    "প্রচুর রক্ত", "রক্তক্ষরণ বন্ধ হচ্ছে না", "অনেক রক্ত পড়ছে",
    # Stroke signs
    "মুখ বাঁকা", "হাত-পা অবশ", "কথা জড়িয়ে যাচ্ছে", "পক্ষাঘাত",
    # High fever in infant
    "শিশুর তীব্র জ্বর", "বাচ্চার খিঁচুনি",
    # Severe poisoning / overdose
    "বিষ খেয়েছে", "ওষুধ বেশি খেয়েছে", "বিষক্রিয়া",
    # Snake/animal bite
    "সাপে কেটেছে", "সাপের কামড়", "কুকুরে কামড়েছে",
]

URGENT_KEYWORDS = [
    # High fever
    "তীব্র জ্বর", "খুব জ্বর", "১০৪ জ্বর", "১০৫ জ্বর",
    # Severe abdominal pain
    "তীব্র পেটব্যথা", "তীব্র পেট ব্যথা", "অনেক পেট ব্যথা",
    # Blood in vomit / stool
    "বমিতে রক্ত", "রক্ত বমি", "পায়খানায় রক্ত", "রক্ত পায়খানা",
    # Severe diarrhea / dehydration
    "বারবার পাতলা পায়খানা", "প্রচুর পাতলা পায়খানা", "পানিশূন্যতা",
    # Diabetic emergency signs
    "হাত পা কাঁপছে", "ঘাম হচ্ছে মাথা ঘুরছে",
    # Severe allergic reaction
    "গলা ফুলে গেছে", "শ্বাস নিতে কষ্ট হচ্ছে",
    # Head injury
    "মাথায় আঘাত", "মাথা ফেটে গেছে",
]

# ---------------------------------------------------------------------------
# Urgency → Facility mapping
# ---------------------------------------------------------------------------

FACILITY_MAP = {
    "EMERGENCY": "জেলা হাসপাতাল / ইমার্জেন্সি — এখনই যান। ৯৯৯ কল করুন।",
    "URGENT":    "উপজেলা স্বাস্থ্য কমপ্লেক্স — আজকের মধ্যে যান।",
    "SELF-CARE": "কমিউনিটি ক্লিনিক বা স্থানীয় ওষুধের দোকান।",
}

# ---------------------------------------------------------------------------
# Core rule engine
# ---------------------------------------------------------------------------

def _keyword_match(text: str, keywords: list) -> Optional[str]:
    """
    Check if any keyword from the list appears in the text.

    Args:
        text:     Bangla text to search.
        keywords: List of Bangla keyword phrases.

    Returns:
        The first matching keyword, or None.
    """
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            return kw
    return None


def apply_clinical_rules(
    raw_text: str,
    entities: dict,
    ml_top_diseases: list,
) -> dict:
    """
    Apply the clinical rule engine to determine the final urgency level.

    This function OVERRIDES the ML output if emergency or urgent keywords
    are detected. The ML output is passed through unchanged only if no
    rule is triggered.

    Args:
        raw_text:         Original Bangla symptom text.
        entities:         NER output dict {'symptoms': [...], 'diseases': [...], ...}
        ml_top_diseases:  Top-3 disease predictions from the XGBoost classifier.

    Returns:
        dict:
            {
                'urgency':      str,    # 'EMERGENCY' | 'URGENT' | 'SELF-CARE'
                'facility':     str,    # Bangla facility instruction
                'top_diseases': list,   # May be overridden on EMERGENCY
                'triggered_by': str,    # Which keyword triggered the rule (or 'ML')
                'overridden':   bool,   # True if ML output was overridden
            }
    """
    # Combine raw text + extracted symptom strings for matching
    search_corpus = raw_text + " " + " ".join(entities.get("symptoms", []))

    # -----------------------------------------------------------------------
    # EMERGENCY check — highest priority
    # -----------------------------------------------------------------------
    emergency_trigger = _keyword_match(search_corpus, EMERGENCY_KEYWORDS)
    if emergency_trigger:
        logger.warning("EMERGENCY rule triggered by: '%s'", emergency_trigger)
        return {
            "urgency": "EMERGENCY",
            "facility": FACILITY_MAP["EMERGENCY"],
            "top_diseases": ml_top_diseases,   # Keep ML output for context but urgency overpowers
            "triggered_by": emergency_trigger,
            "overridden": True,
        }

    # -----------------------------------------------------------------------
    # URGENT check — second priority
    # -----------------------------------------------------------------------
    urgent_trigger = _keyword_match(search_corpus, URGENT_KEYWORDS)
    if urgent_trigger:
        logger.info("URGENT rule triggered by: '%s'", urgent_trigger)
        return {
            "urgency": "URGENT",
            "facility": FACILITY_MAP["URGENT"],
            "top_diseases": ml_top_diseases,
            "triggered_by": urgent_trigger,
            "overridden": True,
        }

    # -----------------------------------------------------------------------
    # No rule triggered — fall through to ML output
    # ML urgency determination could be added here in a future iteration.
    # -----------------------------------------------------------------------
    logger.info("No clinical rule triggered; using ML output as SELF-CARE at default.")
    return {
        "urgency": "SELF-CARE",
        "facility": FACILITY_MAP["SELF-CARE"],
        "top_diseases": ml_top_diseases,
        "triggered_by": "ML",
        "overridden": False,
    }


# ---------------------------------------------------------------------------
# Validation helper (run during testing)
# ---------------------------------------------------------------------------

def validate_vignette(raw_text: str, expected_urgency: str) -> dict:
    """
    Validate a single clinical vignette against the rule engine.

    Used by tests/eval_classifier.py for the 50-vignette safety test.

    Args:
        raw_text:         Bangla symptom description.
        expected_urgency: 'EMERGENCY' | 'URGENT' | 'SELF-CARE'

    Returns:
        dict: {'pass': bool, 'expected': str, 'actual': str, 'triggered_by': str}
    """
    result = apply_clinical_rules(raw_text, {"symptoms": []}, [])
    actual_urgency = result["urgency"]
    is_safe = actual_urgency == expected_urgency

    # An UNSAFE result = expected EMERGENCY but got SELF-CARE or URGENT
    is_unsafe = expected_urgency == "EMERGENCY" and actual_urgency != "EMERGENCY"

    return {
        "pass": is_safe,
        "unsafe": is_unsafe,
        "expected": expected_urgency,
        "actual": actual_urgency,
        "triggered_by": result["triggered_by"],
    }


# ---------------------------------------------------------------------------
# Dev test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("বুকে ব্যথা, শ্বাস কষ্ট", "EMERGENCY"),
        ("তিন দিন জ্বর, মাথাব্যথা, চোখ লাল", "URGENT"),
        ("গলা ব্যথা, সর্দি, হালকা জ্বর", "SELF-CARE"),
    ]
    for text, expected in test_cases:
        result = apply_clinical_rules(text, {"symptoms": []}, [])
        status = "✅" if result["urgency"] == expected else "❌"
        print(f"{status} [{expected} → {result['urgency']}] {text[:50]}")
