"""
Phase 9 — Cross-Encoder Reranker
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to re-score retrieved chunks.
The cross-encoder reads (query, passage) pairs jointly, giving much finer
relevance judgements than the bi-encoder used during retrieval.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_model = None  # lazy singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        print("[reranker] Loading cross-encoder model…")
        _model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
        print("[reranker] Cross-encoder loaded.")
    return _model


def rerank(query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
    """
    Score each chunk against the query and return them sorted by score descending.

    Args:
        query:   The user question (original or rewritten).
        chunks:  List of chunk dicts, each must have a "text" key.
        top_k:   If given, return only the top-k results.

    Returns:
        Chunks sorted by cross-encoder score, each with a "_ce_score" key added.
    """
    if not chunks:
        return chunks

    model = _get_model()
    texts = [c.get("text", "") for c in chunks]
    pairs = [(query, t) for t in texts]

    scores = model.predict(pairs)  # numpy array of floats

    scored = []
    for chunk, score in zip(chunks, scores):
        c = dict(chunk)
        c["_ce_score"] = float(score)
        scored.append(c)

    scored.sort(key=lambda x: x["_ce_score"], reverse=True)

    if top_k is not None:
        scored = scored[:top_k]

    # Strip internal score key before returning
    for c in scored:
        c.pop("_ce_score", None)

    return scored
