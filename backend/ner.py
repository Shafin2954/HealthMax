import csv
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

_ner_pipeline = None
_rule_based_symptoms: List[tuple[str, str]] | None = None
_known_disease_surfaces: set[str] | None = None
_known_medicine_surfaces: set[str] | None = None

LOCAL_NER_MODEL_PATHS = [
    Path("models/ner-banglabert-medical"),
    Path("models/ner_model"),
]

SYMPTOM_LIST_PATHS = [
    Path("models/symptom_list.json"),
    Path("data/symptom_vocab.json"),
]

DISEASE_LIST_PATHS = [
    Path("models/disease_records.json"),
    Path("models/label_encoder.json"),
]

MEDICINE_LIST_PATHS = [
    Path("healthmax-ai-assistant/src/data/medicine_ner_v2.csv"),
    Path("healthmax-ai-assistant/src/data/medicine_ner.csv"),
]

COMMON_SYMPTOM_ALIASES = {
    "মাথাব্যথা": "মাথা ব্যথা",
    "মাথা ব্যাথা": "মাথা ব্যথা",
    "গা ব্যথা": "শরীর ব্যথা",
    "গায়ে ব্যথা": "শরীর ব্যথা",
    "গায়ে ব্যথা": "শরীর ব্যথা",
    "শরীর ব্যথা": "শরীর ব্যথা",
    "পেটব্যথা": "পেট ব্যথা",
    "পেটে ব্যথা": "পেট ব্যথা",
    "বুক ব্যথা": "বুকে ব্যথা",
    "বুকে ব্যথা": "বুকে ব্যথা",
    "শ্বাস নিতে কষ্ট": "শ্বাসকষ্ট",
    "শ্বাস কষ্ট": "শ্বাসকষ্ট",
    "চোখের পেছনে ব্যথা": "চোখের পেছনে ব্যথা",
    "বমি বমি": "বমি বমি ভাব",
    "বমি বমি লাগছে": "বমি বমি ভাব",
    "র‍্যাশ": "র্যাশ",
    "র‌্যাশ": "র্যাশ",
    "রাশ": "র্যাশ",
}


def _normalize_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u200c", "").replace("\u200d", "")
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[।,;.!?()]+", " ", normalized)
    return " ".join(normalized.split()).casefold()


def _normalize_text(text: str) -> str:
    normalized = _normalize_surface(text)
    return COMMON_SYMPTOM_ALIASES.get(normalized, normalized)


def _load_ner_model():
    global _ner_pipeline
    if _ner_pipeline is None:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        device = -1
        try:
            import torch

            if torch.cuda.is_available():
                device = 0
        except Exception:
            device = -1
        from transformers import pipeline

        configured_model = os.getenv("NER_MODEL_PATH")
        if configured_model:
            model_name = configured_model
        else:
            model_name = "sagorsarker/bangla-bert-base"
            for candidate in LOCAL_NER_MODEL_PATHS:
                if candidate.exists():
                    model_name = str(candidate)
                    break
        print(f"[NER] Loading model: {model_name}")
        _ner_pipeline = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",
            device=device,
        )
        tokenizer = getattr(_ner_pipeline, "tokenizer", None)
        if tokenizer is not None:
            model_max_length = getattr(tokenizer, "model_max_length", None)
            if not isinstance(model_max_length, int) or model_max_length > 100000:
                tokenizer.model_max_length = 128
        print("[NER] Model loaded.")
    return _ner_pipeline


def _surface_form(symptom_name: str) -> str:
    before_paren = symptom_name.split("(", 1)[0]
    return _normalize_text(before_paren)


def _load_rule_based_symptoms() -> List[tuple[str, str]]:
    global _rule_based_symptoms
    if _rule_based_symptoms is not None:
        return _rule_based_symptoms

    canonical_surfaces: set[str] = set()
    for path in SYMPTOM_LIST_PATHS:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
        if isinstance(loaded, list):
            for item in loaded:
                canonical_surfaces.add(_surface_form(str(item)))
        elif isinstance(loaded, dict):
            for key in loaded.keys():
                canonical_surfaces.add(_surface_form(str(key)))

    if not canonical_surfaces:
        canonical_surfaces = {
            "জ্বর",
            "মাথা ব্যথা",
            "বুকে ব্যথা",
            "শ্বাসকষ্ট",
            "বমি",
            "ডায়রিয়া",
            "পেট ব্যথা",
            "কাশি",
            "গলা ব্যথা",
            "সর্দি",
            "দুর্বলতা",
            "মাথা ঘোরা",
            "চোখ লাল হওয়া",
            "শরীর ব্যথা",
            "খিঁচুনি",
            "অজ্ঞান",
            "রক্তপাত",
        }

    aliases: dict[str, str] = {}
    for canonical in canonical_surfaces:
        aliases[canonical] = canonical
        aliases[canonical.replace(" ", "")] = canonical

    for alias, canonical in COMMON_SYMPTOM_ALIASES.items():
        aliases[_normalize_surface(alias)] = _normalize_text(canonical)

    _rule_based_symptoms = sorted(
        aliases.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    return _rule_based_symptoms


def _split_csv_terms(raw_value: str) -> List[str]:
    return [
        cleaned
        for cleaned in (
            _normalize_text(part).strip(" ,;:/|[]{}()\"'।")
            for part in re.split(r"[,\n;/|]+", raw_value)
        )
        if cleaned
    ]


def _load_known_disease_surfaces() -> set[str]:
    global _known_disease_surfaces
    if _known_disease_surfaces is not None:
        return _known_disease_surfaces

    disease_surfaces: set[str] = set()
    for path in DISEASE_LIST_PATHS:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as file:
            loaded = json.load(file)

        if isinstance(loaded, list):
            for item in loaded:
                if isinstance(item, dict):
                    disease_name = str(item.get("disease", "")).strip()
                else:
                    disease_name = str(item).strip()
                full = _normalize_text(disease_name)
                bare = _normalize_text(full.split("(", 1)[0])
                if full:
                    disease_surfaces.add(full.casefold())
                if bare:
                    disease_surfaces.add(bare.casefold())

    for disease in DISEASE_KEYWORDS:
        disease_surfaces.add(_normalize_text(disease).casefold())

    _known_disease_surfaces = disease_surfaces
    return _known_disease_surfaces


def _load_known_medicine_surfaces() -> set[str]:
    global _known_medicine_surfaces
    if _known_medicine_surfaces is not None:
        return _known_medicine_surfaces

    medicine_surfaces: set[str] = set()
    for path in MEDICINE_LIST_PATHS:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                raw_value = str(row.get("Medicine\\ Chemical Name", "")).strip()
                for medicine_name in _split_csv_terms(raw_value):
                    medicine_surfaces.add(medicine_name.casefold())

    for medicine in MEDICINE_KEYWORDS:
        medicine_surfaces.add(_normalize_text(medicine).casefold())

    _known_medicine_surfaces = medicine_surfaces
    return _known_medicine_surfaces


def _is_plausible_model_entity(
    label: str,
    normalized_word: str,
    normalized_text: str,
    score: float | None = None,
) -> bool:
    if not normalized_word:
        return False

    compact_word = normalized_word.replace(" ", "")
    if len(compact_word) < 2:
        return False

    if normalized_word.casefold() not in normalized_text.casefold():
        return False

    if label == "SYMPTOM":
        allowed_surfaces = {
            _normalize_text(alias).casefold()
            for alias, canonical in _load_rule_based_symptoms()
            if alias and canonical
        }
        allowed_surfaces.update(
            _normalize_text(canonical).casefold()
            for _, canonical in _load_rule_based_symptoms()
            if canonical
        )
        return normalized_word.casefold() in allowed_surfaces

    if label == "DISEASE":
        return (
            normalized_word.casefold() in _load_known_disease_surfaces()
            or (score is not None and score >= 0.7)
        )

    if label == "MEDICINE":
        return (
            normalized_word.casefold() in _load_known_medicine_surfaces()
            or (score is not None and score >= 0.7 and len(compact_word) >= 3)
        )

    return False


DISEASE_KEYWORDS = [
    "ডেঙ্গু",
    "ম্যালেরিয়া",
    "টাইফয়েড",
    "নিউমোনিয়া",
    "ডায়াবেটিস",
    "উচ্চ রক্তচাপ",
    "যক্ষ্মা",
    "কলেরা",
    "জন্ডিস",
    "হাঁপানি",
]

MEDICINE_KEYWORDS = [
    "প্যারাসিটামল",
    "মেট্রোনিডাজল",
    "অ্যামোক্সিসিলিন",
    "ওরস্যালাইন",
    "ইনসুলিন",
    "এমলোডিপিন",
    "সালবিউটামল",
]


def extract_symptoms(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from Bangla text.
    Returns dict with keys: symptoms, diseases, medicines.
    Uses rule-based symptom matching first and falls back to model inference if available.
    """
    entities = {"symptoms": [], "diseases": [], "medicines": []}
    normalized_text = _normalize_text(text)
    matchable_text = _normalize_surface(text)
    normalized_text_folded = normalized_text.casefold()

    for alias, canonical in _load_rule_based_symptoms():
        if alias and alias in matchable_text and canonical not in entities["symptoms"]:
            entities["symptoms"].append(canonical)

    for disease in DISEASE_KEYWORDS:
        if disease in text and disease not in entities["diseases"]:
            entities["diseases"].append(disease)

    for medicine in MEDICINE_KEYWORDS:
        if medicine in text and medicine not in entities["medicines"]:
            entities["medicines"].append(medicine)

    for disease_surface in sorted(_load_known_disease_surfaces(), key=len, reverse=True):
        if disease_surface and disease_surface in normalized_text_folded and disease_surface not in entities["diseases"]:
            entities["diseases"].append(disease_surface)

    for medicine_surface in sorted(_load_known_medicine_surfaces(), key=len, reverse=True):
        if medicine_surface and medicine_surface in normalized_text_folded and medicine_surface not in entities["medicines"]:
            entities["medicines"].append(medicine_surface)

    try:
        ner = _load_ner_model()
        model_entities = ner(text)
        for entity in model_entities:
            label = entity.get("entity_group", "").upper()
            raw_word = str(entity.get("word", "")).strip()
            if "##" in raw_word:
                continue
            word = _normalize_text(raw_word)
            score = entity.get("score")
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                numeric_score = None
            if not _is_plausible_model_entity(label, word, normalized_text, numeric_score):
                continue
            if "SYMPTOM" in label and word not in entities["symptoms"]:
                entities["symptoms"].append(word)
            elif "DISEASE" in label and word not in entities["diseases"]:
                entities["diseases"].append(word)
            elif "MEDICINE" in label and word not in entities["medicines"]:
                entities["medicines"].append(word)
    except Exception as error:
        print(f"[NER] Model inference failed, using rule-based extraction only: {error}")

    return entities
