from typing import Dict, List

_ner_pipeline = None


def _load_ner_model():
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        # Use fine-tuned BanglaBERT or the pre-available BanglaNER
        # Replace model path with your fine-tuned S3-loaded checkpoint after training
        model_name = "sagorsarker/bangla-bert-base"
        print(f"[NER] Loading model: {model_name}")
        _ner_pipeline = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple"
        )
        print("[NER] Model loaded.")
    return _ner_pipeline


# Symptom keyword dictionary for rule-based fallback NER
SYMPTOM_KEYWORDS = [
    "জ্বর", "মাথাব্যথা", "বুকে ব্যথা", "শ্বাসকষ্ট", "বমি", "ডায়রিয়া",
    "পেটব্যথা", "কাশি", "গলাব্যথা", "সর্দি", "দুর্বলতা", "মাথা ঘোরা",
    "চোখ লাল", "গা ব্যথা", "খিঁচুনি", "অজ্ঞান", "রক্তপাত", "চর্মরোগ",
    "হাত পা কাঁপছে", "ঘাম", "পানিশূন্যতা", "প্রস্রাব জ্বালা", "হলুদ চোখ"
]

DISEASE_KEYWORDS = [
    "ডেঙ্গু", "ম্যালেরিয়া", "টাইফয়েড", "নিউমোনিয়া", "ডায়াবেটিস",
    "উচ্চ রক্তচাপ", "যক্ষ্মা", "কলেরা", "জন্ডিস", "হাঁপানি"
]

MEDICINE_KEYWORDS = [
    "প্যারাসিটামল", "মেট্রোনিডাজল", "অ্যামোক্সিসিলিন", "ওরস্যালাইন",
    "ইনসুলিন", "এমলোডিপিন", "সালবিউটামল"
]


def extract_symptoms(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from Bangla text.
    Returns dict with keys: symptoms, diseases, medicines
    Uses BanglaBERT NER + rule-based keyword fallback.
    """
    entities = {"symptoms": [], "diseases": [], "medicines": []}

    # Rule-based extraction (always runs as fallback/supplement)
    for symptom in SYMPTOM_KEYWORDS:
        if symptom in text and symptom not in entities["symptoms"]:
            entities["symptoms"].append(symptom)

    for disease in DISEASE_KEYWORDS:
        if disease in text and disease not in entities["diseases"]:
            entities["diseases"].append(disease)

    for medicine in MEDICINE_KEYWORDS:
        if medicine in text and medicine not in entities["medicines"]:
            entities["medicines"].append(medicine)

    # Model-based NER (if model available and text is non-empty)
    try:
        ner = _load_ner_model()
        model_entities = ner(text)
        for entity in model_entities:
            label = entity.get("entity_group", "").upper()
            word = entity.get("word", "").strip()
            if not word:
                continue
            if "SYMPTOM" in label and word not in entities["symptoms"]:
                entities["symptoms"].append(word)
            elif "DISEASE" in label and word not in entities["diseases"]:
                entities["diseases"].append(word)
            elif "MEDICINE" in label and word not in entities["medicines"]:
                entities["medicines"].append(word)
    except Exception as e:
        print(f"[NER] Model inference failed, using keyword fallback only: {e}")

    return entities
