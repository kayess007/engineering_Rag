import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from app.vectordb import get_vectorstore

CHUNKED_DIR = Path("storage/chunked")

_chunk_cache: List[Dict] = []
_cached_file_count: int = -1


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_/.\-]+", text.lower())


def load_all_chunk_files() -> List[Dict]:
    """Load chunked JSON files with file-count-based cache invalidation."""
    global _chunk_cache, _cached_file_count

    files = list(CHUNKED_DIR.glob("*_chunks.json")) if CHUNKED_DIR.exists() else []

    if len(files) == _cached_file_count:
        return _chunk_cache

    all_chunks: List[Dict] = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                if isinstance(chunks, list):
                    all_chunks.extend(chunks)
        except Exception as e:
            print(f"Failed to load chunk file {file_path}: {e}")

    _chunk_cache = all_chunks
    _cached_file_count = len(files)
    return _chunk_cache


def chunk_to_result(chunk: Dict) -> Dict:
    return {
        "text": chunk.get("text", "").strip(),
        "metadata": {
            "chunk_id": chunk.get("chunk_id"),
            "manual_id": chunk.get("manual_id"),
            "section_title": chunk.get("section_title"),
            "content_type": chunk.get("content_type"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "source_file": chunk.get("metadata", {}).get("source_file"),
            "parent_section": chunk.get("metadata", {}).get("parent_section"),
            "table_title": chunk.get("metadata", {}).get("table_title"),
        },
    }


def score_result(query: str, item: Dict) -> float:
    query_norm = normalize_text(query)
    query_tokens = set(tokenize(query))

    text = item.get("text", "")
    metadata = item.get("metadata", {})
    content_type = str(metadata.get("content_type", ""))

    body_norm = normalize_text(text)
    body_tokens = set(tokenize(text))

    section_title = normalize_text(str(metadata.get("section_title", "")))
    parent_section = normalize_text(str(metadata.get("parent_section", "")))

    score = 0.0

    # Semantic similarity from vector search (normalized cosine, [0, 1])
    score += item.get("_vector_similarity", 0.0) * 6.0

    # Exact phrase in body
    if query_norm and query_norm in body_norm:
        score += 15.0

    # Token overlap in body
    overlap = len(query_tokens.intersection(body_tokens))
    score += overlap * 2.5

    # Exact phrase in titles
    if query_norm and query_norm in section_title:
        score += 8.0
    if query_norm and query_norm in parent_section:
        score += 6.0

    # Partial title overlap
    title_tokens = set(tokenize(section_title)) | set(tokenize(parent_section))
    title_overlap = len(query_tokens.intersection(title_tokens))
    score += title_overlap * 3.0

    # Engineering domain term boost
    engineering_terms = {
        "iso", "ppm", "torque", "class", "viscosity", "filter",
        "hydraulic", "lubrication", "oil", "pressure", "maintenance",
        "interval", "specification", "spec", "hfe", "hlp", "hvlp",
    }
    exact_terms = query_tokens.intersection(engineering_terms).intersection(body_tokens)
    score += len(exact_terms) * 1.5

    # Prefer richer chunks
    if len(text.strip()) > 120:
        score += 2.0

    # Table boost for spec-style queries
    spec_terms = {"specification", "spec", "torque", "interval", "class", "ppm", "iso"}
    if query_tokens.intersection(spec_terms) and content_type == "table":
        score += 4.0

    return score


def keyword_search(query: str, k: int = 10) -> List[Dict]:
    """Keyword retrieval over cached chunked JSON files."""
    chunks = load_all_chunk_files()
    scored: List[Tuple[float, Dict]] = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        result = chunk_to_result(chunk)
        s = score_result(query, result)

        if s > 0:
            scored.append((s, result))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]


def vector_search(query: str, collection_name: str, k: int = 10) -> List[Dict]:
    vectorstore = get_vectorstore(collection_name)
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=max(k * 4, 20))

    results = []
    for doc, distance in docs_and_scores:
        text = doc.page_content.strip()
        if not text:
            continue

        # With normalize_embeddings=True, cosine distance is in [0, 2].
        # Convert to similarity in [0, 1] so it contributes meaningfully to reranking.
        similarity = max(0.0, 1.0 - distance / 2.0)

        results.append(
            {
                "text": text,
                "metadata": doc.metadata,
                "_vector_similarity": similarity,
            }
        )

    return results


def deduplicate_results(results: List[Dict], max_results: int) -> List[Dict]:
    seen: set = set()
    deduped: List[Dict] = []

    for item in results:
        metadata = item.get("metadata", {})
        section_title = str(metadata.get("section_title", "")).strip()
        page_start = metadata.get("page_start") or -1
        text_prefix = item.get("text", "").strip()[:120]

        key = (section_title, page_start, text_prefix)
        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

        if len(deduped) >= max_results:
            break

    return deduped


def retrieve_chunks(query: str, collection_name: str, k: int = 10) -> List[Dict]:
    # 1. Semantic retrieval (with similarity scores for reranking)
    vector_results = vector_search(query, collection_name=collection_name, k=k)

    # 2. Keyword retrieval over cached chunk files
    keyword_results = keyword_search(query, k=max(k * 3, 15))

    # 3. Merge and rerank with unified scoring
    merged = vector_results + keyword_results
    ranked = sorted(
        merged,
        key=lambda item: score_result(query, item),
        reverse=True,
    )

    # 4. Deduplicate and strip internal scoring keys
    final = deduplicate_results(ranked, max_results=k)
    for item in final:
        item.pop("_vector_similarity", None)

    return final
