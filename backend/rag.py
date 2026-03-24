import json
import os
import pickle
from typing import Any, Dict, List, Tuple, cast

import numpy as np

_faiss_index: Any = None
_disease_records: List[Dict[str, Any]] | None = None
_rag_vectorizer: Any = None
_rag_embedding_model: Any = None
_rag_config: Dict[str, Any] | None = None

FAISS_INDEX_PATH = "models/disease_rag.index"
DISEASE_RECORDS_PATH = "models/disease_records.json"
RAG_VECTORIZER_PATH = "models/disease_rag_vectorizer.pkl"
RAG_CONFIG_PATH = "models/rag_config.json"


def _prepare_transformers_backend() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def _load_rag_config() -> Dict[str, Any]:
    global _rag_config
    if _rag_config is None:
        if os.path.exists(RAG_CONFIG_PATH):
            with open(RAG_CONFIG_PATH, "r", encoding="utf-8") as file:
                _rag_config = dict(json.load(file))
        else:
            _rag_config = {
                "backend": "tfidf",
                "embedding_model": None,
                "embedding_device": "cpu",
                "vectorizer_path": RAG_VECTORIZER_PATH,
            }
    return _rag_config


def _load_embedding_model() -> Any:
    global _rag_embedding_model
    if _rag_embedding_model is None:
        config = _load_rag_config()
        model_name = str(config.get("embedding_model") or "paraphrase-multilingual-MiniLM-L12-v2")

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        _prepare_transformers_backend()
        from sentence_transformers import SentenceTransformer

        print(f"[RAG] Loading embedding model: {model_name} on {device}")
        _rag_embedding_model = SentenceTransformer(model_name, device=device)
        print("[RAG] Embedding model loaded.")
    return _rag_embedding_model


def _load_vectorizer() -> Any:
    global _rag_vectorizer
    if _rag_vectorizer is None:
        config = _load_rag_config()
        vectorizer_path = str(config.get("vectorizer_path") or RAG_VECTORIZER_PATH)
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"RAG vectorizer not found at {vectorizer_path}. "
                "Run: python data/process_datasets.py"
            )
        print("[RAG] Loading TF-IDF vectorizer...")
        with open(vectorizer_path, "rb") as file:
            _rag_vectorizer = pickle.load(file)
        print("[RAG] TF-IDF vectorizer loaded.")
    return _rag_vectorizer


def _load_faiss_index() -> Tuple[Any, List[Dict[str, Any]]]:
    global _faiss_index, _disease_records
    if _faiss_index is None:
        import faiss

        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run: python data/process_datasets.py"
            )
        print("[RAG] Loading FAISS index...")
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)

        with open(DISEASE_RECORDS_PATH, "r", encoding="utf-8") as file:
            loaded_records = json.load(file)

        # Keep compatibility with older artifact formats that used disease_name.
        _disease_records = []
        for record in loaded_records:
            normalized_record = dict(record)
            normalized_record["disease"] = (
                normalized_record.get("disease")
                or normalized_record.get("disease_name")
                or ""
            )
            _disease_records.append(normalized_record)

        print(f"[RAG] FAISS index loaded. {_faiss_index.ntotal} disease records.")
    assert _faiss_index is not None
    assert _disease_records is not None
    return _faiss_index, _disease_records


def _l2_normalize_dense(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def retrieve_diseases(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k matching disease records from FAISS index.
    Returns list of disease dicts with: disease, symptoms, urgency, specialist, score
    """
    config = _load_rag_config()
    index, records = _load_faiss_index()

    backend = str(config.get("backend") or "tfidf")
    if backend == "sentence-transformers":
        embedding_model = _load_embedding_model()
        query_embedding = embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.ascontiguousarray(np.asarray(query_embedding, dtype=np.float32), dtype=np.float32)
    else:
        vectorizer = _load_vectorizer()
        query_embedding = cast(Any, vectorizer).transform([query_text])
        query_embedding = np.asarray(cast(Any, query_embedding).toarray(), dtype=np.float32)
        query_embedding = np.ascontiguousarray(_l2_normalize_dense(query_embedding), dtype=np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(records):
            continue
        record = records[idx].copy()
        record["retrieval_score"] = float(score)
        results.append(record)

    return results


def get_disease_records() -> List[Dict[str, Any]]:
    _, records = _load_faiss_index()
    return [dict(record) for record in records]
