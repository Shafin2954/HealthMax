from typing import Any, Dict, List

# Emergency keywords override all model output.
EMERGENCY_KEYWORDS = [
    "বুকে ব্যথা",
    "বুক ব্যথা",
    "শ্বাস নিতে পারছি না",
    "শ্বাসকষ্ট",
    "শ্বাস কষ্ট",
    "অজ্ঞান",
    "খিঁচুনি",
    "স্ট্রোক",
    "মুখ বাঁকা",
    "হাত অসাড়",
    "প্রচুর রক্তপাত",
    "রক্ত বমি",
    "মুখ দিয়ে রক্ত",
    "শিশুর উচ্চ জ্বর",
    "নবজাতক জ্বর",
    "সাপে কেটেছে",
    "সাপে কামড়",
    "সারা শরীর নীল",
    "জ্ঞান নেই",
]

URGENT_KEYWORDS = [
    "উচ্চ জ্বর",
    "১০৪ জ্বর",
    "১০৫ জ্বর",
    "তীব্র পেটব্যথা",
    "প্রচণ্ড পেটব্যথা",
    "রক্তে বমি",
    "পানিশূন্যতা",
    "ডিহাইড্রেশন",
    "তীব্র ডায়রিয়া",
    "কলেরার মতো",
]

FACILITY_MAP = {
    "EMERGENCY": "জেলা হাসপাতাল বা মেডিকেল কলেজ হাসপাতাল",
    "URGENT": "উপজেলা স্বাস্থ্য কমপ্লেক্স বা নিকটস্থ ডাক্তার",
    "SELF-CARE": "কমিউনিটি ক্লিনিক বা বাড়িতে চিকিৎসা",
}

URGENCY_BANGLA = {
    "EMERGENCY": "অতি জরুরি 🚨 — এখনই যান",
    "URGENT": "জরুরি ⚠️ — আজই যান",
    "SELF-CARE": "স্বাস্থ্যসেবা ✅ — বাড়িতে চিকিৎসা",
}


def _record_disease_name(record: Dict[str, Any]) -> str:
    return str(
        record.get("disease")
        or record.get("disease_name")
        or record.get("disease_name_bn")
        or "অজানা"
    )


def _record_urgency(record: Dict[str, Any]) -> str:
    return str(record.get("urgency") or record.get("urgency_level") or "SELF-CARE").upper()


def _rag_predictions(rag_results: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    selected = rag_results[:top_n]
    total_score = sum(float(record.get("retrieval_score", 0.0)) for record in selected)
    if total_score <= 0:
        total_score = 1.0
    return [
        {
            "disease": _record_disease_name(record),
            "probability": float(record.get("retrieval_score", 0.0)) / total_score,
        }
        for record in selected
    ]


def apply_triage_rules(
    text: str,
    symptoms: List[str],
    classifier_results: List[Dict[str, Any]],
    rag_results: List[Dict[str, Any]],
    merged_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Apply clinical triage rules. Rules are BINDING over ML output.
    Emergency check runs first and overrides everything.
    """
    combined_text = text + " " + " ".join(symptoms)
    preferred_predictions = merged_results or classifier_results or _rag_predictions(rag_results)

    # ── EMERGENCY CHECK (highest priority) ──
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in combined_text:
            return {
                "urgency_level": "EMERGENCY",
                "urgency_label_bn": URGENCY_BANGLA["EMERGENCY"],
                "facility": FACILITY_MAP["EMERGENCY"],
                "emergency_override": True,
                "triggered_rule": keyword,
                "top_disease": preferred_predictions[0]["disease"] if preferred_predictions else "অজানা",
                "top_diseases": preferred_predictions,
                "action_instruction": (
                    f"⚠️ '{keyword}' উপসর্গ শনাক্ত হয়েছে। "
                    "এখনই ৯৯৯ কল করুন অথবা নিকটস্থ জেলা হাসপাতালে নিয়ে যান।"
                )
            }

    # ── URGENT CHECK ──
    for keyword in URGENT_KEYWORDS:
        if keyword in combined_text:
            fallback_predictions = preferred_predictions or _rag_predictions(rag_results)
            top_disease = preferred_predictions[0]["disease"] if preferred_predictions else (
                _record_disease_name(rag_results[0]) if rag_results else "অজানা"
            )
            return {
                "urgency_level": "URGENT",
                "urgency_label_bn": URGENCY_BANGLA["URGENT"],
                "facility": FACILITY_MAP["URGENT"],
                "emergency_override": False,
                "triggered_rule": keyword,
                "top_disease": top_disease,
                "top_diseases": fallback_predictions,
                "action_instruction": (
                    "আজই উপজেলা স্বাস্থ্য কমপ্লেক্সে যান। দেরি করবেন না।"
                )
            }

    # ── ML-BASED URGENCY (advisory) ──
    top_prediction = preferred_predictions[0] if preferred_predictions else None
    top_probability = float(top_prediction.get("probability", 0.0)) if top_prediction else 0.0

    urgency = "SELF-CARE"
    if top_prediction and top_probability >= 0.20:
        top_disease = str(top_prediction["disease"])
        high_urgency_diseases = [
            "Dengue", "Typhoid", "Pneumonia", "Malaria", "Cholera",
            "ডেঙ্গু", "টাইফয়েড", "নিউমোনিয়া", "ম্যালেরিয়া", "ম্যালেরিয়া", "কলেরা"
        ]
        if any(d in top_disease for d in high_urgency_diseases):
            urgency = "URGENT"
    elif rag_results:
        urgency = _record_urgency(rag_results[0])

    top_disease = str(top_prediction["disease"]) if top_prediction else (
        _record_disease_name(rag_results[0]) if rag_results else (
            classifier_results[0]["disease"] if classifier_results else "নির্ধারণ সম্ভব হয়নি"
        )
    )
    display_predictions = preferred_predictions or (_rag_predictions(rag_results) or classifier_results)

    return {
        "urgency_level": urgency,
        "urgency_label_bn": URGENCY_BANGLA.get(urgency, URGENCY_BANGLA["SELF-CARE"]),
        "facility": FACILITY_MAP.get(urgency, FACILITY_MAP["SELF-CARE"]),
        "emergency_override": False,
        "triggered_rule": None,
        "top_disease": top_disease,
        "top_diseases": display_predictions,
        "action_instruction": "স্থানীয় স্বাস্থ্যকেন্দ্রে যান এবং ডাক্তারের পরামর্শ নিন।"
    }
