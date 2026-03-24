import re
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple

SYMPTOM_ALIASES = {
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
    "বমি বমি": "বমি বমি ভাব",
    "বমি বমি লাগছে": "বমি বমি ভাব",
    "র‍্যাশ": "র্যাশ",
    "র‌্যাশ": "র্যাশ",
    "রাশ": "র্যাশ",
}


def normalize_surface(text: str) -> str:
    before_paren = str(text).split("(", 1)[0]
    normalized = unicodedata.normalize("NFKC", before_paren)
    normalized = normalized.replace("\u200c", "").replace("\u200d", "")
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[।,;.!?()]+", " ", normalized)
    normalized = " ".join(normalized.split()).casefold()
    return SYMPTOM_ALIASES.get(normalized, normalized)


def _best_fuzzy_score(query: str, candidates: Sequence[str]) -> float:
    try:
        from rapidfuzz import fuzz

        return max((float(fuzz.WRatio(query, candidate)) for candidate in candidates), default=0.0)
    except Exception:
        return 0.0


def _unique_normalized(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    normalized_values: List[str] = []
    for value in values:
        normalized = normalize_surface(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_values.append(normalized)
    return normalized_values


def _record_symptom_surfaces(record: Dict[str, Any]) -> List[str]:
    symptoms = record.get("symptoms", [])
    if not isinstance(symptoms, list):
        return []
    return _unique_normalized([str(symptom) for symptom in symptoms])


def _disease_surface(record_or_name: Dict[str, Any] | str) -> str:
    if isinstance(record_or_name, dict):
        return normalize_surface(str(record_or_name.get("disease", "")))
    return normalize_surface(str(record_or_name))


def _symptom_overlap(
    input_symptoms: Sequence[str],
    candidate_symptoms: Sequence[str],
) -> Tuple[float, List[str]]:
    normalized_inputs = _unique_normalized([str(symptom) for symptom in input_symptoms])
    normalized_candidates = _unique_normalized([str(symptom) for symptom in candidate_symptoms])

    if not normalized_inputs or not normalized_candidates:
        return 0.0, []

    matched_candidates: set[int] = set()
    matched_inputs: List[str] = []

    for input_symptom in normalized_inputs:
        best_index = -1
        best_score = 0.0
        for index, candidate_symptom in enumerate(normalized_candidates):
            if index in matched_candidates:
                continue
            if (
                input_symptom == candidate_symptom
                or input_symptom in candidate_symptom
                or candidate_symptom in input_symptom
            ):
                best_index = index
                best_score = 100.0
                break
            fuzzy_score = _best_fuzzy_score(input_symptom, [candidate_symptom])
            if fuzzy_score > best_score:
                best_index = index
                best_score = fuzzy_score

        if best_index >= 0 and best_score >= 86.0:
            matched_candidates.add(best_index)
            matched_inputs.append(input_symptom)

    if not matched_inputs:
        return 0.0, []

    overlap_score = len(matched_inputs) / max(1, len(normalized_inputs))
    return overlap_score, matched_inputs


def _disease_mention_strength(disease_mentions: Sequence[str], disease_name: str) -> float:
    candidate_surface = _disease_surface(disease_name)
    if not candidate_surface:
        return 0.0

    best_score = 0.0
    for mention in disease_mentions:
        normalized_mention = normalize_surface(str(mention))
        if not normalized_mention:
            continue
        if normalized_mention == candidate_surface:
            return 1.0
        if (
            normalized_mention in candidate_surface
            or candidate_surface in normalized_mention
        ):
            best_score = max(best_score, 0.85)
            continue
        fuzzy_score = _best_fuzzy_score(normalized_mention, [candidate_surface])
        if fuzzy_score >= 92.0:
            best_score = max(best_score, 0.75)

    return best_score


def merge_disease_predictions(
    symptoms: Sequence[str],
    disease_mentions: Sequence[str],
    classifier_results: Sequence[Dict[str, Any]],
    rag_results: Sequence[Dict[str, Any]],
    all_disease_records: Sequence[Dict[str, Any]],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    record_lookup = {
        str(record.get("disease", "")).strip(): dict(record)
        for record in all_disease_records
        if str(record.get("disease", "")).strip()
    }
    candidates: Dict[str, Dict[str, Any]] = {}

    def ensure_candidate(disease_name: str) -> Dict[str, Any]:
        candidate = candidates.setdefault(
            disease_name,
            {
                "disease": disease_name,
                "record": record_lookup.get(disease_name),
                "classifier_score": 0.0,
                "rag_score": 0.0,
                "symptom_overlap": 0.0,
                "matched_symptoms": [],
                "mention_score": 0.0,
                "raw_score": 0.0,
            },
        )
        if candidate["record"] is None and disease_name in record_lookup:
            candidate["record"] = record_lookup[disease_name]
        return candidate

    normalized_symptoms = _unique_normalized([str(symptom) for symptom in symptoms])

    max_classifier_probability = max(
        (float(result.get("probability", 0.0)) for result in classifier_results),
        default=0.0,
    )
    if max_classifier_probability > 0:
        for result in classifier_results:
            disease_name = str(result.get("disease", "")).strip()
            if not disease_name:
                continue
            candidate = ensure_candidate(disease_name)
            probability = float(result.get("probability", 0.0))
            candidate["classifier_score"] = max(
                float(candidate["classifier_score"]),
                probability / max_classifier_probability,
            )

    max_rag_score = max(
        (max(0.0, float(result.get("retrieval_score", 0.0))) for result in rag_results),
        default=0.0,
    )
    if max_rag_score > 0:
        for result in rag_results:
            disease_name = str(result.get("disease", "")).strip()
            if not disease_name:
                continue
            candidate = ensure_candidate(disease_name)
            candidate["record"] = dict(result)
            retrieval_score = max(0.0, float(result.get("retrieval_score", 0.0)))
            candidate["rag_score"] = max(
                float(candidate["rag_score"]),
                retrieval_score / max_rag_score,
            )

    for record in all_disease_records:
        disease_name = str(record.get("disease", "")).strip()
        if not disease_name:
            continue

        overlap_score, matched_symptoms = _symptom_overlap(
            normalized_symptoms,
            _record_symptom_surfaces(record),
        )
        mention_score = _disease_mention_strength(disease_mentions, disease_name)
        if overlap_score <= 0 and mention_score <= 0:
            continue

        candidate = ensure_candidate(disease_name)
        candidate["record"] = dict(record)
        if overlap_score > float(candidate["symptom_overlap"]):
            candidate["symptom_overlap"] = overlap_score
            candidate["matched_symptoms"] = matched_symptoms
        candidate["mention_score"] = max(float(candidate["mention_score"]), mention_score)

    if not candidates:
        return []

    for candidate in candidates.values():
        matched_count = len(candidate["matched_symptoms"])
        mention_bonus = 1.35 * float(candidate["mention_score"])
        overlap_bonus = 0.95 * float(candidate["symptom_overlap"]) + 0.22 * matched_count
        classifier_bonus = 0.30 * float(candidate["classifier_score"])
        rag_bonus = 0.25 * float(candidate["rag_score"])
        candidate["raw_score"] = mention_bonus + overlap_bonus + classifier_bonus + rag_bonus

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda candidate: (
            float(candidate["raw_score"]),
            float(candidate["mention_score"]),
            float(candidate["symptom_overlap"]),
            float(candidate["classifier_score"]),
            float(candidate["rag_score"]),
        ),
        reverse=True,
    )[: max(1, top_n)]

    total_score = sum(max(float(candidate["raw_score"]), 0.0) for candidate in ranked_candidates)
    if total_score <= 0:
        total_score = float(len(ranked_candidates))

    merged_results: List[Dict[str, Any]] = []
    for candidate in ranked_candidates:
        record = candidate.get("record") or {}
        probability = max(float(candidate["raw_score"]), 0.0) / total_score
        merged_results.append(
            {
                "disease": str(candidate["disease"]),
                "probability": probability,
                "urgency": str(record.get("urgency", "URGENT")),
                "specialist": str(record.get("specialist", "General Physician")),
                "facility": str(record.get("facility", "Upazila Health Complex")),
                "matched_symptoms": list(candidate["matched_symptoms"]),
                "support": {
                    "classifier_score": round(float(candidate["classifier_score"]), 4),
                    "rag_score": round(float(candidate["rag_score"]), 4),
                    "symptom_overlap": round(float(candidate["symptom_overlap"]), 4),
                    "mention_score": round(float(candidate["mention_score"]), 4),
                },
            }
        )

    return merged_results
