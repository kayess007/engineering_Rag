import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from app.vectordb import get_vectorstore

CHUNKED_DIR = Path("storage/chunked")

_chunk_cache: List[Dict] = []
_cached_file_count: int = -1

# BM25 index — rebuilt whenever chunk files change
_bm25 = None
_bm25_chunks: List[Dict] = []  # child chunks in same order as BM25 corpus


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_/.\-]+", text.lower())


_parent_index: Dict[str, Dict] = {}  # parent_id → parent chunk


def load_all_chunk_files() -> List[Dict]:
    """Load chunked JSON files with file-count-based cache invalidation.
    Also rebuilds BM25 index and parent lookup on change."""
    global _chunk_cache, _cached_file_count, _parent_index
    global _bm25, _bm25_chunks

    files = list(CHUNKED_DIR.glob("*_chunks.json")) if CHUNKED_DIR.exists() else []

    if len(files) == _cached_file_count:
        return _chunk_cache

    all_chunks: List[Dict] = []
    parent_index: Dict[str, Dict] = {}
    child_chunks: List[Dict] = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                if isinstance(chunks, list):
                    all_chunks.extend(chunks)
                    for c in chunks:
                        if c.get("chunk_type") == "parent":
                            parent_index[c["chunk_id"]] = c
                        elif c.get("chunk_type", "child") == "child" and c.get("text", "").strip():
                            child_chunks.append(c)
        except Exception as e:
            print(f"Failed to load chunk file {file_path}: {e}")

    _chunk_cache = all_chunks
    _parent_index = parent_index
    _cached_file_count = len(files)

    # Rebuild BM25 index over child chunks
    if child_chunks:
        try:
            from rank_bm25 import BM25Okapi
            corpus = [tokenize(c.get("text", "")) for c in child_chunks]
            _bm25 = BM25Okapi(corpus)
            _bm25_chunks = child_chunks
            print(f"[retriever] BM25 index built: {len(child_chunks)} child chunks")
        except ImportError:
            print("[retriever] rank-bm25 not installed — falling back to token overlap")
            _bm25 = None
            _bm25_chunks = []
    else:
        _bm25 = None
        _bm25_chunks = []

    return _chunk_cache


def resolve_to_parent(item: Dict) -> Dict:
    """
    If item is a child chunk, return its parent chunk instead.
    The parent has the full ~1200-token context the LLM needs.
    Falls back to the child itself if the parent is not found.
    """
    parent_id = item.get("metadata", {}).get("parent_id")
    if not parent_id:
        return item

    load_all_chunk_files()  # ensures _parent_index is populated
    parent = _parent_index.get(parent_id)
    if not parent:
        return item

    # Build a result dict from the parent chunk, preserving retrieval metadata
    return {
        "text": parent.get("text", ""),
        "metadata": {
            "chunk_id": parent.get("chunk_id"),
            "chunk_type": "parent",
            "manual_id": parent.get("manual_id"),
            "section_title": parent.get("section_title"),
            "content_type": parent.get("content_type"),
            "page_start": parent.get("page_start"),
            "page_end": parent.get("page_end"),
            "source_file": parent.get("metadata", {}).get("source_file"),
            "parent_section": parent.get("metadata", {}).get("parent_section"),
        },
        "_vector_similarity": item.get("_vector_similarity", 0.0),
    }


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


def _chunk_matches_filters(chunk: Dict, filters: dict | None) -> bool:
    """Return True if chunk metadata satisfies all filters."""
    if not filters:
        return True
    for key, value in filters.items():
        if key == "page_start":
            if (chunk.get("page_start") or 0) < int(value):
                return False
        else:
            # Check both top-level and nested metadata
            chunk_val = chunk.get(key) or chunk.get("metadata", {}).get(key)
            if str(chunk_val) != str(value):
                return False
    return True


def bm25_search(query: str, k: int = 10, filters: dict | None = None) -> List[Dict]:
    """
    BM25 sparse retrieval over child chunks.
    Returns parent chunks for full LLM context.
    Falls back to token-overlap scoring if rank-bm25 is not installed.
    """
    load_all_chunk_files()  # ensures _bm25 and _bm25_chunks are populated

    if _bm25 is None or not _bm25_chunks:
        return _token_overlap_search(query, k, filters=filters)

    query_tokens = tokenize(query)
    scores = _bm25.get_scores(query_tokens)

    # Pair each child chunk with its BM25 score, applying metadata filters
    scored = sorted(
        [
            (score, chunk)
            for score, chunk in zip(scores, _bm25_chunks)
            if _chunk_matches_filters(chunk, filters)
        ],
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, chunk in scored[:k * 2]:  # fetch extra before parent resolution deduplicates
        if score <= 0:
            break
        result = chunk_to_result(chunk)
        result["_bm25_score"] = float(score)
        results.append(resolve_to_parent(result))

    # Deduplicate parents that appeared via multiple children
    seen = set()
    deduped = []
    for r in results:
        key = r.get("metadata", {}).get("chunk_id", id(r))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
        if len(deduped) >= k:
            break

    return deduped


def _token_overlap_search(query: str, k: int = 10, filters: dict | None = None) -> List[Dict]:
    """Fallback keyword search using token overlap scoring."""
    chunks = load_all_chunk_files()
    scored: List[Tuple[float, Dict]] = []

    for chunk in chunks:
        if chunk.get("chunk_type", "child") == "parent":
            continue
        if not _chunk_matches_filters(chunk, filters):
            continue
        text = chunk.get("text", "").strip()
        if not text:
            continue
        result = chunk_to_result(chunk)
        s = score_result(query, result)
        if s > 0:
            scored.append((s, result))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [resolve_to_parent(item) for _, item in scored[:k]]


def keyword_search(query: str, k: int = 10, filters: dict | None = None) -> List[Dict]:
    """Public interface — uses BM25 when available, falls back to token overlap."""
    return bm25_search(query, k, filters=filters)


def vector_search(
    query: str,
    collection_name: str,
    k: int = 10,
    filters: dict | None = None,
) -> List[Dict]:
    from app.vectordb import build_chroma_filter
    vectorstore = get_vectorstore(collection_name)
    where = build_chroma_filter(filters) if filters else None

    kwargs = {"k": max(k * 4, 20)}
    if where:
        kwargs["filter"] = where

    docs_and_scores = vectorstore.similarity_search_with_score(query, **kwargs)

    results = []
    for doc, distance in docs_and_scores:
        text = doc.page_content.strip()
        if not text:
            continue
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


def retrieve_chunks_advanced(
    query: str,
    collection_name: str,
    k: int = 8,
    rewrite_model: str = "gpt-4.1-mini",
    filters: dict | None = None,
) -> List[Dict]:
    """
    Phase 9 enhanced retrieval:
      1. Rewrite query into multiple variants
      2. Run hybrid retrieval for each variant (wider net)
      3. Merge + deduplicate candidates
      4. Cross-encoder rerank → return top-k
    """
    from app.query_rewriter import rewrite_query
    from app.reranker import rerank

    queries = rewrite_query(query, model=rewrite_model)
    print(f"[advanced] Query variants: {queries}")

    # Collect candidates from all query variants (larger pool)
    candidate_pool: List[Dict] = []
    fetch_k = max(k * 4, 20)

    for q in queries:
        vec = vector_search(q, collection_name=collection_name, k=fetch_k, filters=filters)
        vec = [resolve_to_parent(r) for r in vec]
        kw = keyword_search(q, k=fetch_k, filters=filters)
        merged = vec + kw
        ranked = sorted(merged, key=lambda item: score_result(q, item), reverse=True)
        candidate_pool.extend(ranked[:fetch_k])

    # Deduplicate across all variants
    deduped = deduplicate_results(candidate_pool, max_results=min(len(candidate_pool), k * 8))

    # Strip internal scoring key before passing to cross-encoder
    for item in deduped:
        item.pop("_vector_similarity", None)

    # Cross-encoder rerank using original query
    reranked = rerank(query, deduped, top_k=k)
    return reranked


def retrieve_chunks(
    query: str,
    collection_name: str,
    k: int = 10,
    filters: dict | None = None,
) -> List[Dict]:
    # 1. Semantic retrieval (child chunks — small, precise)
    vector_results = vector_search(query, collection_name=collection_name, k=k, filters=filters)

    # 2. Resolve child chunks → parent chunks for full context
    vector_results = [resolve_to_parent(r) for r in vector_results]

    # 3. BM25 keyword retrieval with same filters
    keyword_results = keyword_search(query, k=max(k * 3, 15), filters=filters)

    # 4. Merge and rerank with unified scoring
    merged = vector_results + keyword_results
    ranked = sorted(
        merged,
        key=lambda item: score_result(query, item),
        reverse=True,
    )

    # 5. Deduplicate and strip internal scoring keys
    final = deduplicate_results(ranked, max_results=k)
    for item in final:
        item.pop("_vector_similarity", None)
        item.pop("_bm25_score", None)

    return final
