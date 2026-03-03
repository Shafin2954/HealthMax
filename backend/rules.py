from typing import List, Dict

# ─────────────────────────────────────────────
# EMERGENCY TRIGGERS — ANY match → EMERGENCY
# These override ALL ML output. Rules are BINDING.
# ─────────────────────────────────────────────
EMERGENCY_KEYWORDS = [
    # Cardiac / Respiratory
    "বুকে ব্যথা", "বুক ব্যথা", "শ্বাস নিতে পারছি না", "শ্বাসকষ্ট", "শ্বাস কষ্ট",
    # Neurological
    "অজ্ঞান", "খিঁচুনি", "স্ট্রোক", "মুখ বাঁকা", "হাত অসাড়",
    # Bleeding
    "প্রচুর রক্তপাত", "রক্ত বমি", "মুখ দিয়ে রক্ত",
    # Pediatric
    "শিশুর উচ্চ জ্বর", "নবজাতক জ্বর",
    # Trauma
    "সাপে কেটেছে", "সাপে কামড়",
    # Severe Sepsis indicators
    "সারা শরীর নীল", "জ্ঞান নেই"
]

# ─────────────────────────────────────────────
# URGENT TRIGGERS — ANY match → URGENT (if no emergency)
# ─────────────────────────────────────────────
URGENT_KEYWORDS = [
    "উচ্চ জ্বর", "১০৪ জ্বর", "১০৫ জ্বর",
    "তীব্র পেটব্যথা", "প্রচণ্ড পেটব্যথা",
    "রক্তে বমি", "পানিশূন্যতা", "ডিহাইড্রেশন",
    "তীব্র ডায়রিয়া", "কলেরার মতো"
]

# ─────────────────────────────────────────────
# FACILITY MAPPING
# ─────────────────────────────────────────────
FACILITY_MAP = {
    "EMERGENCY": "জেলা হাসপাতাল বা মেডিকেল কলেজ হাসপাতse dicts with: disease, symptoms, urgency, specia",
    "SELF-CARE": "কমিউনিটি ক্লিনিক বা বাড়িতে চিকিৎসা"
}

URGENCY_BANGLA = {
    "EMERGENCY": "অতি জরুরি 🚨 — এখনই যান",
    "URGENT": "জরুরি ⚠️ — আজই যান",
    "SELF-CARE": "স্বাস্থ্যসেবা ✅ — বাড়িতে চিকিৎse dicts with: disease, symptoms, urgency, specialy_triage_rules(
    text: str,
    symptoms: List[str],
    classifier_results: List[Dict],
    rag_results: List[Dict]
) -> Dict:
    """
    Apply clinical triage rules. Rules are BINDING over ML output.
    Emergency check runs first and overrides everything.
    """
    combined_text = text + " " + " ".join(symptoms)

    # ── EMERGENCY CHECK (highest priority) ──
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in combined_text:
            return {
                "urgency_level": "EMERGENCY",
                "urgency_label_bn": URGENCY_BANGLA["EMERGENCY"],
                "facility": FACILITY_MAP["EMERGENCY"],
                "emergency_override": True,
                "triggered_rule": keyword,
                "top_disease": classifier_results[0]["disease"] if classifier_results else "অজানা",
                "top_diseases": classifier_results,
                "action_instruction": (
                    f"⚠️ '{keyword}' উপসর্গ শনাক্ত হয়েছে। "
                    "এখনই ৯৯৯ কল করুন অথবা নিকটস্থ জেলা হাসপাতালে নিয়ে যান।"
                )
            }

    # ── URGENT CHECK ──
    for keyword in URGENT_KEYWORDS:
        if keyword in combined_text:
            top_disease = classifier_results[0]["disease"] if classifier_results else (
                rag_results[0]["disease"] if rag_results else "অজানা"
            )
            return {
                "urgency_level": "URGENT",
                "urgency_label_bn": URGENCY_BANGLA["URGENT"],
                "facility": FACILITY_MAP["URGENT"],
                "emergency_override": False,
                "triggered_rule": keyword,
                "top_disease": top_disease,
                "top_diseases": classifier_results,
                "action_instruction": (
                    "আজই উপজেলা স্বাস্থ্য কমপ্লেক্সে যান। দেরি করবেন না।"
                )
            }

    # ── ML-BASED URGENCY (advisory) ──
    # Use classifier result to determine urgency
    urgency = "SELF-CARE"
    if classifier_results:
        top_disease = classifier_results[0]["disease"]
        high_urgency_diseases = [
            "Dengue", "Typhoid", "Pneumonia", "Malaria", "Cholera",
            "ডেঙ্গু", "টাইফয়েড", "নিউমোনিয়া", "ম্যালেরিয়া", "কলেরা"
        ]
        if any(d in top_disease for d in high_urgency_diseases):
            urgency = "URGENT"
    elif rag_results:
        urgency = rag_results[0].get("urgency", "SELF-CARE").upper()

    top_disease = classifier_results[0]["disease"] if classifier_results else (
        rag_results[0]["disease"] if rag_results else "নির্ধারণ সম্ভব হয়নি"
    )

    return {
        "urgency_level": urgency,
        "urgency_label_bn": URGENCY_BANGLA.get(urgency, URGENCY_BANGLA["SELF-CARE"]),
        "facility": FACILITY_MAP.get(urgency, FACILITY_MAP["SELF-CARE"]),
        "emergency_override": False,
        "triggered_rule": None,
        "top_disease": top_disease,
        "top_diseases": classifier_results,
        "action_instruction": "স্থানীয় স্বাস্থ্যকেন্দ্রে যান এবং ডাক্তারের পরামর্শ নিন।"
    }
