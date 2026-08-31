from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from financial_research_agent.domain import DocumentHit

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PACKAGE_ROOT / "data" / "documents"


def _tokenize(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()) if len(token) > 2
    }


def _token_counts(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def search_documents(query: str, limit: int = 5) -> list[DocumentHit]:
    query_terms = _tokenize(query)
    hits: list[DocumentHit] = []

    for path in sorted(DOCUMENTS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_terms = _tokenize(text)
        overlap = query_terms & doc_terms
        if not overlap:
            continue
        score = len(overlap) / max(len(query_terms), 1)
        snippet = " ".join(text.replace("\n", " ").split()[:42])
        hits.append(DocumentHit(path=str(path), snippet=snippet, score=score))

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:limit]


def search_documents_vector(query: str, limit: int = 5) -> list[DocumentHit]:
    """Rank local documents with a lightweight vector-space retriever.

    This is intentionally dependency-free. It behaves like a small local RAG
    retriever by embedding documents as normalized token-frequency vectors and
    ranking them with cosine similarity.
    """

    query_vector = _token_counts(query)
    hits: list[DocumentHit] = []

    for path in sorted(DOCUMENTS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_vector = _token_counts(text)
        score = _cosine_similarity(query_vector, doc_vector)
        if score <= 0:
            continue
        snippet = " ".join(text.replace("\n", " ").split()[:42])
        hits.append(DocumentHit(path=str(path), snippet=snippet, score=score))

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:limit]


def search_documents_semantic(query: str, limit: int = 5) -> list[DocumentHit]:
    """Search local documents with sentence embeddings and FAISS when available.

    This is the production-style RAG path. It stays optional because FAISS and
    sentence-transformers are free but large dependencies. If they are missing,
    the function falls back to the dependency-free vector-space retriever.
    """

    try:
        import faiss  # type: ignore[import-not-found]
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional native/ML dependencies fail in several ways
        return search_documents_vector(query=query, limit=limit)

    paths = sorted(DOCUMENTS_DIR.glob("*.md"))
    if not paths:
        return []

    texts = [path.read_text(encoding="utf-8") for path in paths]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_embeddings = model.encode(texts, normalize_embeddings=True)
    query_embedding = model.encode([query], normalize_embeddings=True)

    doc_matrix = np.asarray(doc_embeddings, dtype="float32")
    query_matrix = np.asarray(query_embedding, dtype="float32")
    index = faiss.IndexFlatIP(doc_matrix.shape[1])
    index.add(doc_matrix)
    scores, indices = index.search(query_matrix, min(limit, len(paths)))

    hits: list[DocumentHit] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        text = texts[int(idx)]
        snippet = " ".join(text.replace("\n", " ").split()[:42])
        hits.append(DocumentHit(path=str(paths[int(idx)]), snippet=snippet, score=float(score)))
    return hits
