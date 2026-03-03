import os
import json
import numpy as np
from typing import List, Dict

_faiss_index = None
_disease_records = None
_embedding_model = None

FAISS_INDEX_PATH = "models/disease_rag.index"
DISEASE_RECORDS_PATH = "models/disease_records.json"


def _load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("[RAG] Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("[RAG] Embedding model loaded.")
    return _embedding_model


def _load_faiss_index():
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

        with open(DISEASE_RECORDS_PATH, "r", encoding="utf-8") as f:
            _disease_records = json.load(f)
        print(f"[RAG] FAISS index loaded. {_faiss_index.ntotal} disease records.")
    return _faiss_index, _disease_records


def retrieve_diseases(query_text: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k matching disease records from FAISS index.
    Returns list of disease dicts with: disease, symptoms, urgency, specialist, score
    """
    model = _load_embedding_model()
    index, recse dicts with: disease, symptoms, urgency, speciandex()

    query_embedding = model.encode([query_text], convert_to_numpy=True)
    query_embedding = query_embedding.astype(np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(records):
            continue
        record = records[idx].copy()
        record["retrieval_score"] = float(1 / (1 + dist))  # Convert L2 distance to similarity
        results.append(record)

    return results
