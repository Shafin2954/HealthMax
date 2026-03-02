"""
HealthMax — Layer 3: RAG Retrieval Pipeline
Retrieves the most relevant disease records from a FAISS vector index
based on semantic similarity to the user's symptom description.

Embedding model: paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers)
Vector index:    models/disease_rag.index  (built by data/process_datasets.py)

Collaborator instructions:
    - The FAISS index must be built first by running data/process_datasets.py.
    - Implement retrieve_diseases() — the primary function called by main.py.
    - TOP_K default is 5; tune based on precision tests (see plan.md Day 11-12).
    - Each retrieved record must include: disease_name, symptoms, urgency, specialist.
"""

import logging
import os
import json
from typing import Optional

import numpy as np

logger = logging.getLogger("healthmax.rag")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_PATH = os.environ.get(
    "FAISS_INDEX_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "disease_rag.index"),
)
DISEASE_RECORDS_PATH = os.environ.get(
    "DISEASE_RECORDS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "disease_records.json"),
)
TOP_K_DEFAULT = 5

# ---------------------------------------------------------------------------
# Model and index loading
# ---------------------------------------------------------------------------

_embedder = None
_faiss_index = None
_disease_records: list = []  # Parallel list to FAISS vectors — metadata store


def load_embedder():
    """
    Load the multilingual sentence embedding model.
    Called once at startup.

    TODO (collaborator):
        - Use SentenceTransformer(EMBEDDING_MODEL_ID)
        - Cache result in _embedder global
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_ID)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        _embedder = None

    return _embedder


def load_index():
    """
    Load the FAISS index and accompanying disease metadata records.
    Called once at startup.

    TODO (collaborator):
        - import faiss; _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        - Load _disease_records from DISEASE_RECORDS_PATH (JSON list)
        - Verify len(_disease_records) == _faiss_index.ntotal
    """
    global _faiss_index, _disease_records

    if _faiss_index is not None:
        return _faiss_index

    try:
        import faiss  # type: ignore
        if not os.path.exists(FAISS_INDEX_PATH):
            logger.warning("FAISS index not found at %s. Run process_datasets.py first.", FAISS_INDEX_PATH)
            return None
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        logger.info("FAISS index loaded: %d vectors", _faiss_index.ntotal)
    except Exception as e:
        logger.error("Failed to load FAISS index: %s", e)
        return None

    try:
        with open(DISEASE_RECORDS_PATH, "r", encoding="utf-8") as f:
            _disease_records = json.load(f)
        logger.info("Disease records loaded: %d records", len(_disease_records))
    except Exception as e:
        logger.error("Failed to load disease records: %s", e)
        _disease_records = []

    return _faiss_index


# ---------------------------------------------------------------------------
# Core RAG retrieval
# ---------------------------------------------------------------------------

def embed_text(text: str) -> Optional[np.ndarray]:
    """
    Embed a Bangla text string into a dense vector.

    Args:
        text: Input Bangla string.

    Returns:
        1-D numpy float32 array of shape (embedding_dim,), or None on failure.
    """
    embedder = load_embedder()
    if embedder is None:
        return None
    try:
        vector = embedder.encode([text], normalize_embeddings=True)
        return vector.astype(np.float32)
    except Exception as e:
        logger.exception("Embedding failed: %s", e)
        return None


def retrieve_diseases(symptom_text: str, top_k: int = TOP_K_DEFAULT) -> list:
    """
    Retrieve the top-k most semantically similar disease records
    for the given Bangla symptom description.

    Args:
        symptom_text: Bangla description of symptoms.
        top_k:        Number of records to retrieve (default 5).

    Returns:
        List of up to top_k dicts, each containing:
            {
                'disease_name': str,
                'symptoms':     list[str],
                'urgency':      str,       # 'EMERGENCY' | 'URGENT' | 'SELF-CARE'
                'specialist':   str,       # e.g. 'Cardiologist', 'General Physician'
                'score':        float,     # Cosine similarity score
            }
        Returns [] if index not loaded or embedding fails.

    TODO (collaborator):
        1. Call embed_text(symptom_text) to get query vector.
        2. Call _faiss_index.search(query_vector, top_k) → distances, indices.
        3. For each index, fetch record from _disease_records[idx].
        4. Attach the distance/score to each record.
        5. Return sorted list (highest similarity first).
    """
    index = load_index()
    if index is None:
        logger.warning("FAISS index not available; returning empty retrieval.")
        return []

    query_vector = embed_text(symptom_text)
    if query_vector is None:
        return []

    try:
        distances, indices = index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(_disease_records):
                continue
            record = dict(_disease_records[idx])
            record["score"] = float(dist)
            results.append(record)
        return results
    except Exception as e:
        logger.exception("FAISS search failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Dev test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_embedder()
    load_index()
    sample = "তিন দিন ধরে জ্বর, মাথাব্যথা, চোখ লাল"
    results = retrieve_diseases(sample, top_k=3)
    print(f"Query: {sample}")
    for r in results:
        print(f"  {r.get('disease_name')} (score={r.get('score', 0):.3f})")
