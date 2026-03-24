"""
Build a silver-labeled Bangla medical NER dataset from local project CSVs.

This repo does not currently include a gold BIO-tagged corpus, so we bootstrap
training data by matching symptom, disease, and medicine terms against:

- data/raw/Symptoms.csv
- healthmax-ai-assistant/src/data/medicine_ner.csv
- healthmax-ai-assistant/src/data/medicine_ner_v2.csv
- healthmax-ai-assistant/src/data/specialist_classification.csv

Outputs:
- data/processed/ner_silver_train.jsonl
- data/processed/ner_silver_validation.jsonl
- data/processed/ner_silver_summary.json
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_LIST = [
    "O",
    "B-SYMPTOM",
    "I-SYMPTOM",
    "B-DISEASE",
    "I-DISEASE",
    "B-MEDICINE",
    "I-MEDICINE",
]

RAW_SYMPTOM_DATASET_PATH = Path("data/raw/Symptoms.csv")
SPECIALIST_DATASET_PATH = Path("healthmax-ai-assistant/src/data/specialist_classification.csv")
MEDICINE_DATASET_PATHS = [
    Path("healthmax-ai-assistant/src/data/medicine_ner_v2.csv"),
    Path("healthmax-ai-assistant/src/data/medicine_ner.csv"),
]

OUTPUT_DIR = Path("data/processed")
TRAIN_OUTPUT_PATH = OUTPUT_DIR / "ner_silver_train.jsonl"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "ner_silver_validation.jsonl"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "ner_silver_summary.json"

TOKEN_PATTERN = re.compile(
    r"[\u0980-\u09FFA-Za-z0-9]+(?:[./-][\u0980-\u09FFA-Za-z0-9]+)*|[^\s]",
    flags=re.UNICODE,
)

COMMON_SYMPTOM_ALIASES = {
    "মাথাব্যথা": "মাথা ব্যথা",
    "গা ব্যথা": "শরীর ব্যথা",
    "শরীর ব্যথা": "শরীর ব্যথা",
    "পেটব্যথা": "পেট ব্যথা",
    "পেটে ব্যথা": "পেট ব্যথা",
    "বুক ব্যথা": "বুকে ব্যথা",
    "বুকে ব্যথা": "বুকে ব্যথা",
    "শ্বাস নিতে কষ্ট": "শ্বাসকষ্ট",
    "শ্বাস কষ্ট": "শ্বাসকষ্ট",
    "চোখের পেছনে ব্যথা": "চোখের পেছনে ব্যথা",
    "র‍্যাশ": "র্যাশ",
    "র‌্যাশ": "র্যাশ",
    "রাশ": "র্যাশ",
}

GENERIC_TERMS_TO_SKIP = {
    "",
    "ওষুধ",
    "সমস্যা",
    "অসুস্থতা",
    "উপসর্গ",
    "ব্যাধি",
    "রোগ",
    "ব্যথা",
    "চিকিৎসা",
}

LABEL_PRIORITY = {
    "MEDICINE": 0,
    "DISEASE": 1,
    "SYMPTOM": 2,
}


def _configure_output_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except ValueError:
                pass


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u200c", "").replace("\u200d", "")
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokenize(text: str) -> List[str]:
    return [match.group(0) for match in TOKEN_PATTERN.finditer(text)]


def _tokenize_with_spans(text: str) -> List[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _clean_list_term(value: str) -> str:
    cleaned = _normalize_text(value)
    cleaned = cleaned.strip(" ,;:/|[]{}()\"'।")
    return cleaned


def _split_terms(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = _normalize_text(str(value))
    if not text:
        return []
    parts = re.split(r"[,\n;/|]+", text)
    cleaned = [
        _clean_list_term(part)
        for part in parts
        if _clean_list_term(part)
    ]
    return _unique_preserve_order(cleaned)


def _clean_symptom_name(column_name: str) -> str:
    raw = str(column_name).split("(", 1)[0]
    return _normalize_text(raw)


def _disease_name_variants(disease_name: str) -> List[str]:
    full = _normalize_text(disease_name)
    bare = _normalize_text(full.split("(", 1)[0])
    variants = [variant for variant in (full, bare) if variant]
    return _unique_preserve_order(variants)


def _load_symptom_matrix() -> pd.DataFrame:
    return pd.read_csv(RAW_SYMPTOM_DATASET_PATH)


def _build_symptom_lexicon(df: pd.DataFrame) -> List[str]:
    terms: List[str] = []
    for column in df.columns:
        if column == "prognosis":
            continue
        cleaned = _clean_symptom_name(column)
        if cleaned:
            terms.append(cleaned)
            compact = cleaned.replace(" ", "")
            if compact and compact != cleaned:
                terms.append(compact)

    for alias, canonical in COMMON_SYMPTOM_ALIASES.items():
        terms.append(_normalize_text(alias))
        terms.append(_normalize_text(canonical))

    filtered = [
        term
        for term in _unique_preserve_order(terms)
        if len(term) >= 2 and term not in GENERIC_TERMS_TO_SKIP
    ]
    return filtered


def _build_disease_lexicon(df: pd.DataFrame) -> List[str]:
    terms: List[str] = []
    for disease_name in df["prognosis"].fillna("").astype(str).tolist():
        terms.extend(_disease_name_variants(disease_name))
    filtered = [
        term
        for term in _unique_preserve_order(terms)
        if len(term) >= 2 and term not in GENERIC_TERMS_TO_SKIP
    ]
    return filtered


def _comparison_keys(terms: Sequence[str]) -> tuple[set[str], set[str]]:
    normalized = {_normalize_text(term).casefold() for term in terms if _normalize_text(term)}
    compact = {term.replace(" ", "") for term in normalized if term}
    return normalized, compact


def _classify_term(
    term: str,
    symptom_lookup: set[str],
    symptom_compact_lookup: set[str],
    disease_lookup: set[str],
    disease_compact_lookup: set[str],
    default_label: str | None = None,
) -> str | None:
    normalized = _normalize_text(term).casefold()
    compact = normalized.replace(" ", "")

    if (
        not normalized
        or normalized in GENERIC_TERMS_TO_SKIP
        or len(compact) < 2
    ):
        return None

    if normalized in symptom_lookup or compact in symptom_compact_lookup:
        return "SYMPTOM"

    if normalized in disease_lookup or compact in disease_compact_lookup:
        return "DISEASE"

    return default_label


def _candidate_entities_for_text(
    text: str,
    symptom_terms: Sequence[str],
    disease_terms: Sequence[str],
    row_medicines: Sequence[str] | None = None,
    row_disease_terms: Sequence[str] | None = None,
    row_common_terms: Sequence[str] | None = None,
    symptom_lookup: set[str] | None = None,
    symptom_compact_lookup: set[str] | None = None,
    disease_lookup: set[str] | None = None,
    disease_compact_lookup: set[str] | None = None,
) -> List[tuple[str, str]]:
    candidates: List[tuple[str, str]] = []
    normalized_text = _normalize_text(text).casefold()

    for term in symptom_terms:
        if _normalize_text(term).casefold() in normalized_text:
            candidates.append(("SYMPTOM", term))

    for term in disease_terms:
        if _normalize_text(term).casefold() in normalized_text:
            candidates.append(("DISEASE", term))

    for medicine in row_medicines or []:
        candidates.append(("MEDICINE", medicine))

    if (
        row_disease_terms
        and symptom_lookup is not None
        and symptom_compact_lookup is not None
        and disease_lookup is not None
        and disease_compact_lookup is not None
    ):
        for term in row_disease_terms:
            label = _classify_term(
                term,
                symptom_lookup=symptom_lookup,
                symptom_compact_lookup=symptom_compact_lookup,
                disease_lookup=disease_lookup,
                disease_compact_lookup=disease_compact_lookup,
                default_label="DISEASE",
            )
            if label is not None:
                candidates.append((label, term))

    if (
        row_common_terms
        and symptom_lookup is not None
        and symptom_compact_lookup is not None
        and disease_lookup is not None
        and disease_compact_lookup is not None
    ):
        for term in row_common_terms:
            label = _classify_term(
                term,
                symptom_lookup=symptom_lookup,
                symptom_compact_lookup=symptom_compact_lookup,
                disease_lookup=disease_lookup,
                disease_compact_lookup=disease_compact_lookup,
                default_label=None,
            )
            if label is not None:
                candidates.append((label, term))

    deduped: List[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for label, phrase in candidates:
        key = (label, _normalize_text(phrase).casefold())
        if key[1] and key not in seen:
            seen.add(key)
            deduped.append((label, phrase))
    return deduped


def _label_text(text: str, entities: Sequence[tuple[str, str]]) -> Dict[str, Any] | None:
    normalized_text = _normalize_text(text)
    token_spans = _tokenize_with_spans(normalized_text)
    if not token_spans:
        return None

    tokens = [token for token, _, _ in token_spans]
    comparable_tokens = [token.casefold() for token in tokens]
    span_candidates: List[tuple[int, int, str]] = []

    for label, phrase in entities:
        entity_tokens = [token.casefold() for token in _tokenize(_normalize_text(phrase))]
        if not entity_tokens:
            continue

        entity_length = len(entity_tokens)
        for start_index in range(len(comparable_tokens) - entity_length + 1):
            if comparable_tokens[start_index : start_index + entity_length] == entity_tokens:
                span_candidates.append((start_index, start_index + entity_length, label))

    span_candidates.sort(
        key=lambda item: (
            -(item[1] - item[0]),
            LABEL_PRIORITY.get(item[2], 99),
            item[0],
        )
    )

    selected_spans: List[tuple[int, int, str]] = []
    occupied_indices: set[int] = set()
    for start_index, end_index, label in span_candidates:
        span_indices = set(range(start_index, end_index))
        if span_indices & occupied_indices:
            continue
        occupied_indices.update(span_indices)
        selected_spans.append((start_index, end_index, label))

    if not selected_spans:
        return None

    ner_tags = ["O"] * len(tokens)
    for start_index, end_index, label in selected_spans:
        ner_tags[start_index] = f"B-{label}"
        for token_index in range(start_index + 1, end_index):
            ner_tags[token_index] = f"I-{label}"

    return {
        "text": normalized_text,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "entity_count": sum(1 for tag in ner_tags if tag != "O"),
    }


def _iter_medicine_examples(
    symptom_terms: Sequence[str],
    disease_terms: Sequence[str],
    symptom_lookup: set[str],
    symptom_compact_lookup: set[str],
    disease_lookup: set[str],
    disease_compact_lookup: set[str],
) -> Iterable[Dict[str, Any]]:
    seen_row_keys: set[tuple[str, str]] = set()

    for dataset_path in MEDICINE_DATASET_PATHS:
        if not dataset_path.exists():
            continue
        df = pd.read_csv(dataset_path).fillna("")
        for row in df.to_dict(orient="records"):
            text = _normalize_text(str(row.get("Medical Text", "")))
            if not text:
                continue

            medicines = _split_terms(row.get("Medicine\\ Chemical Name", ""))
            row_disease_terms = _split_terms(row.get("Disease", ""))
            common_terms = _split_terms(row.get("Common Medical Terms", ""))

            row_key = (
                text,
                json.dumps(
                    {
                        "medicines": medicines,
                        "disease_terms": row_disease_terms,
                        "common_terms": common_terms,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            if row_key in seen_row_keys:
                continue
            seen_row_keys.add(row_key)

            entities = _candidate_entities_for_text(
                text=text,
                symptom_terms=symptom_terms,
                disease_terms=disease_terms,
                row_medicines=medicines,
                row_disease_terms=row_disease_terms,
                row_common_terms=common_terms,
                symptom_lookup=symptom_lookup,
                symptom_compact_lookup=symptom_compact_lookup,
                disease_lookup=disease_lookup,
                disease_compact_lookup=disease_compact_lookup,
            )
            example = _label_text(text, entities)
            if example is None:
                continue
            example["source"] = "medicine_ner"
            yield example


def _iter_specialist_examples(
    symptom_terms: Sequence[str],
    disease_terms: Sequence[str],
) -> Iterable[Dict[str, Any]]:
    df = pd.read_csv(SPECIALIST_DATASET_PATH)
    for raw_text in df["Problem"].dropna().astype(str).tolist():
        text = _normalize_text(raw_text)
        entities = _candidate_entities_for_text(
            text=text,
            symptom_terms=symptom_terms,
            disease_terms=disease_terms,
        )
        example = _label_text(text, entities)
        if example is None:
            continue
        example["source"] = "specialist_problem"
        yield example


def _synthetic_text(symptoms: Sequence[str], disease_name: str, template_index: int) -> str:
    symptom_text = ", ".join(symptoms)
    if template_index % 2 == 0:
        return f"আমার {symptom_text} আছে। এটা {disease_name} হতে পারে।"
    return f"রোগীর উপসর্গ {symptom_text}। সম্ভাব্য রোগ {disease_name}।"


def _iter_synthetic_examples(symptom_df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    symptom_columns = [column for column in symptom_df.columns if column != "prognosis"]
    for row_index, row in symptom_df.iterrows():
        disease_name = _normalize_text(str(row.get("prognosis", "")))
        if not disease_name:
            continue

        active_symptoms: List[str] = []
        for column in symptom_columns:
            value = row.get(column)
            try:
                is_active = int(float(value)) == 1
            except (TypeError, ValueError):
                is_active = False
            if is_active:
                active_symptoms.append(_clean_symptom_name(column))

        active_symptoms = _unique_preserve_order(symptom for symptom in active_symptoms if symptom)
        if not active_symptoms:
            continue

        chosen_symptoms = active_symptoms[: min(5, len(active_symptoms))]
        text = _synthetic_text(chosen_symptoms, disease_name, row_index)
        entities = [("SYMPTOM", symptom) for symptom in chosen_symptoms]
        entities.extend(("DISEASE", variant) for variant in _disease_name_variants(disease_name))
        example = _label_text(text, entities)
        if example is None:
            continue
        example["source"] = "synthetic_symptom_matrix"
        yield example


def _deduplicate_examples(examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for example in examples:
        key = json.dumps(
            {
                "text": example["text"],
                "tokens": example["tokens"],
                "ner_tags": example["ner_tags"],
                "source": example["source"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            deduped.append(example)
    return deduped


def _write_jsonl(path: Path, examples: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def build_silver_ner_dataset() -> Dict[str, Any]:
    if not RAW_SYMPTOM_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing symptom dataset: {RAW_SYMPTOM_DATASET_PATH}")
    if not SPECIALIST_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing specialist dataset: {SPECIALIST_DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HealthMax Silver NER Dataset Builder")
    print("=" * 60)

    symptom_df = _load_symptom_matrix()
    symptom_terms = _build_symptom_lexicon(symptom_df)
    disease_terms = _build_disease_lexicon(symptom_df)
    symptom_lookup, symptom_compact_lookup = _comparison_keys(symptom_terms)
    disease_lookup, disease_compact_lookup = _comparison_keys(disease_terms)

    print(f"Loaded {len(symptom_terms)} symptom terms and {len(disease_terms)} disease terms.")

    examples: List[Dict[str, Any]] = []
    examples.extend(
        _iter_medicine_examples(
            symptom_terms=symptom_terms,
            disease_terms=disease_terms,
            symptom_lookup=symptom_lookup,
            symptom_compact_lookup=symptom_compact_lookup,
            disease_lookup=disease_lookup,
            disease_compact_lookup=disease_compact_lookup,
        )
    )
    examples.extend(
        _iter_specialist_examples(
            symptom_terms=symptom_terms,
            disease_terms=disease_terms,
        )
    )
    examples.extend(_iter_synthetic_examples(symptom_df))
    examples = _deduplicate_examples(examples)

    if len(examples) < 50:
        raise RuntimeError("Silver NER dataset is too small to train a useful model.")

    source_counts = Counter(example["source"] for example in examples)
    stratify = list(source_counts.elements())
    if any(count < 2 for count in source_counts.values()):
        stratify_values: List[str] | None = None
    else:
        stratify_values = [example["source"] for example in examples]

    train_examples, validation_examples = train_test_split(
        examples,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=stratify_values,
    )

    _write_jsonl(TRAIN_OUTPUT_PATH, train_examples)
    _write_jsonl(VALIDATION_OUTPUT_PATH, validation_examples)

    tag_counts = Counter()
    for example in examples:
        tag_counts.update(example["ner_tags"])

    summary = {
        "label_list": LABEL_LIST,
        "num_total_examples": len(examples),
        "num_train_examples": len(train_examples),
        "num_validation_examples": len(validation_examples),
        "source_distribution": dict(sorted(source_counts.items())),
        "tag_distribution": dict(sorted(tag_counts.items())),
        "num_symptom_terms": len(symptom_terms),
        "num_disease_terms": len(disease_terms),
        "input_paths": {
            "symptom_matrix": str(RAW_SYMPTOM_DATASET_PATH),
            "specialist_classification": str(SPECIALIST_DATASET_PATH),
            "medicine_ner_paths": [str(path) for path in MEDICINE_DATASET_PATHS if path.exists()],
        },
        "output_paths": {
            "train": str(TRAIN_OUTPUT_PATH),
            "validation": str(VALIDATION_OUTPUT_PATH),
            "summary": str(SUMMARY_OUTPUT_PATH),
        },
        "sample_examples": [
            {
                "source": example["source"],
                "text": example["text"],
                "tokens": example["tokens"][:20],
                "ner_tags": example["ner_tags"][:20],
            }
            for example in examples[:3]
        ],
    }

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(train_examples)} train and {len(validation_examples)} validation examples.")
    print(f"[OK] Summary saved to {SUMMARY_OUTPUT_PATH}")
    return summary


if __name__ == "__main__":
    _configure_output_streams()
    build_silver_ner_dataset()
