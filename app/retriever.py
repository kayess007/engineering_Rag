import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

_PART_NUM_RE = re.compile(r'\b\d{3}-\d{4}\b')

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
            "parent_id": chunk.get("parent_id"),   # required for resolve_to_parent()
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

    # Table boost for spec-style queries and part-lookup queries
    spec_terms = {"specification", "spec", "torque", "interval", "class", "ppm", "iso",
                  "part", "number", "serial", "item", "quantity", "plug", "filter", "kit"}
    if query_tokens.intersection(spec_terms) and content_type == "table":
        score += 4.0

    # Lookup mode: query contains a direct part number OR asks "part number" OR is a
    # short component-name query (≤7 tokens) that mentions equipment/component terms.
    _LOOKUP_COMPONENT_TERMS = {
        "plug", "filter", "seal", "bearing", "belt", "hose", "gasket",
        "valve", "pump", "sensor", "relay", "fuse", "alternator", "starter",
        "injector", "nozzle", "ring", "piston", "kit", "bolt", "nut",
    }
    is_lookup = (
        bool(_PART_NUM_RE.search(query))                          # "295-3099"
        or ("part" in query_tokens and "number" in query_tokens)  # "part number"
        or (                                                        # "spark plug c13"
            len(query_tokens) <= 7
            and query_tokens.intersection(_LOOKUP_COMPONENT_TERMS)
        )
    )
    if is_lookup and _PART_NUM_RE.search(text):
        score += 10.0

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
    fetch_k = max(k * 3, 15)

    if where:
        # ChromaDB bug: get(where=..., include=["embeddings"]) and
        # query(where=...) both fail with "Error finding id".
        # Workaround: two-step — get IDs via filter, fetch embeddings by IDs.
        from app.embeddings import get_embedding_model as _get_emb
        import numpy as np

        embedding_model = _get_emb()
        query_embedding = np.array(embedding_model.embed_query(query))

        # Step 1: get matching IDs + docs + metadata (no embeddings)
        id_result = vectorstore._collection.get(
            where=where,
            limit=5000,
            include=["documents", "metadatas"],
        )
        matching_ids = id_result.get("ids", [])
        if not matching_ids:
            return []

        # Step 2: fetch embeddings by IDs (no filter — avoids the bug)
        emb_result = vectorstore._collection.get(
            ids=matching_ids,
            include=["embeddings", "documents", "metadatas"],
        )

        results = []
        for text, metadata, emb in zip(
            emb_result["documents"], emb_result["metadatas"], emb_result["embeddings"]
        ):
            if not text or not text.strip():
                continue
            chunk_emb = np.array(emb)
            similarity = float(np.dot(query_embedding, chunk_emb))
            results.append({
                "text": text,
                "metadata": metadata,
                "_vector_similarity": max(0.0, similarity),
            })

        results.sort(key=lambda x: x["_vector_similarity"], reverse=True)
        return results[:fetch_k]

    # No filter — use LangChain wrapper as normal
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)
    results = []
    for doc, distance in docs_and_scores:
        text = doc.page_content.strip()
        if not text:
            continue
        similarity = max(0.0, 1.0 - distance / 2.0)
        results.append({
            "text": text,
            "metadata": doc.metadata,
            "_vector_similarity": similarity,
        })
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
    collection_name: str | None = None,
    k: int = 5,
    rewrite_model: str = "gpt-4.1-mini",
    filters: dict | None = None,
) -> List[Dict]:
    """
    Phase 9 enhanced retrieval:
      1. Classify query → route to correct collection(s)
      2. Rewrite query into multiple variants
      3. Run hybrid retrieval for each variant (wider net)
      4. Merge + deduplicate candidates
      5. Cross-encoder rerank → return top-k
    """
    from app.query_classifier import classify_query, collection_for_type
    from app.query_rewriter import rewrite_query
    from app.reranker import rerank

    query_type = classify_query(query)

    # Resolve collection(s)
    if collection_name is not None:
        collections = [collection_name]
    else:
        collections = collection_for_type(query_type)
        print(f"[advanced] Query type: {query_type} → collections: {collections}")

    query_tokens = set(tokenize(query))
    spec_terms = {
        "torque", "viscosity", "cleanliness", "class", "standard", "standards",
        "iso", "ppm", "interval", "capacity", "temperature", "recommended",
        "spec", "specification",
    }
    if query_type == "parts":
        max_variants = 3
    elif query_tokens.intersection(spec_terms):
        max_variants = 1
    else:
        max_variants = 2

    queries = rewrite_query(query, model=rewrite_model, max_variants=max_variants)
    print(f"[advanced] Query variants: {queries}")

    # Collect candidates from all query variants across all collections
    candidate_pool: List[Dict] = []
    fetch_k = max(k * 3, 15)

    for coll in collections:
        for q in queries:
            vec = vector_search(q, collection_name=coll, k=fetch_k, filters=filters)
            vec = [resolve_to_parent(r) for r in vec]
            kw = keyword_search(q, k=fetch_k, filters=filters)
            merged = vec + kw
            ranked = sorted(merged, key=lambda item: score_result(q, item), reverse=True)
            candidate_pool.extend(ranked[:fetch_k])

    # Deduplicate across all variants and collections
    deduped = deduplicate_results(candidate_pool, max_results=min(len(candidate_pool), k * 3))

    # Strip internal scoring key before passing to cross-encoder
    for item in deduped:
        item.pop("_vector_similarity", None)

    # Enrich table chunk text with section title so cross-encoder has context
    for item in deduped:
        if item.get("metadata", {}).get("content_type") == "table":
            section = item.get("metadata", {}).get("section_title", "")
            text = item.get("text", "")
            if section and section.lower() not in text.lower():
                item["text"] = f"{section}\n{text}"

    # Cross-encoder rerank using original query
    reranked = rerank(query, deduped, top_k=k, min_score=1.5)
    return reranked


def retrieve_chunks(
    query: str,
    collection_name: str | None = None,
    k: int = 5,
    filters: dict | None = None,
) -> List[Dict]:
    from app.query_classifier import classify_query, collection_for_type
    from app.reranker import rerank

    # Classify query to determine retrieval strategy and collection(s)
    query_type = classify_query(query)
    is_lookup = query_type == "parts"

    # Resolve collection(s): explicit arg overrides classifier
    if collection_name is not None:
        collections = [collection_name]
    else:
        collections = collection_for_type(query_type)
        print(f"[retriever] Query type: {query_type} → collections: {collections}")

    # For multi-collection ("both"), run retrieval on each and merge before reranking
    if len(collections) > 1:
        all_candidates: List[Dict] = []
        for coll in collections:
            all_candidates.extend(
                _retrieve_single_collection(query, coll, k, filters, is_lookup=False)
            )
        deduped = deduplicate_results(
            sorted(all_candidates, key=lambda item: score_result(query, item), reverse=True),
            max_results=k * 3,
        )
        for item in deduped:
            item.pop("_vector_similarity", None)
            item.pop("_bm25_score", None)
        return rerank(query, deduped, top_k=k, min_score=1.5)

    return _retrieve_single_collection(query, collections[0], k, filters, is_lookup=is_lookup)


def _retrieve_single_collection(
    query: str,
    collection_name: str,
    k: int,
    filters: dict | None,
    is_lookup: bool,
) -> List[Dict]:
    from app.reranker import rerank
    query_toks = set(tokenize(query))

    # 1. Semantic retrieval (child chunks — small, precise)
    vector_results = vector_search(query, collection_name=collection_name, k=k, filters=filters)

    # 2. Resolve child chunks → parent chunks for full context
    vector_results = [resolve_to_parent(r) for r in vector_results]

    # 3. BM25 keyword retrieval — wider pool for lookup queries so part-number chunks
    #    buried deep in BM25 ranking still make it into the candidate pool.
    bm25_k = max(k * 4, 24) if is_lookup else max(k * 2, 10)
    keyword_results = keyword_search(query, k=bm25_k, filters=filters)

    # 4. Merge and score with unified scoring (score_result gives +10 to part-number chunks)
    merged = vector_results + keyword_results
    ranked = sorted(merged, key=lambda item: score_result(query, item), reverse=True)

    # 5. Deduplicate — keep a wider candidate pool for cross-encoder
    candidates = deduplicate_results(ranked, max_results=k * 3)
    for item in candidates:
        item.pop("_vector_similarity", None)
        item.pop("_bm25_score", None)

    # Enrich table chunk text with section title so cross-encoder has context
    for item in candidates:
        if item.get("metadata", {}).get("content_type") == "table":
            section = item.get("metadata", {}).get("section_title", "")
            text = item.get("text", "")
            if section and section.lower() not in text.lower():
                item["text"] = f"{section}\n{text}"

    # 6. Cross-encoder rerank → precise top-k (improves context precision)
    final = rerank(query, candidates, top_k=k, min_score=1.5)

    # 7. Part-number fallback: if this is a lookup query but no chunk that contains BOTH
    #    a part number AND the distinctive query terms survived reranking, scan the full
    #    merged pool and inject the best matching chunk.
    #
    #    "Distinctive" = query tokens that are NOT generic component/stop words.
    #    Example: "what is the spark plug for c13"
    #      → generic: plug, for, what, is, the, c13
    #      → distinctive: {"spark"}
    #    A "PLUG GP-WATER LINES" chunk passes part_num but NOT "spark" → fallback triggers.
    if is_lookup:
        _LOOKUP_COMPONENTS = {
            "plug", "filter", "seal", "bearing", "belt", "hose", "gasket",
            "valve", "pump", "sensor", "relay", "fuse", "alternator", "starter",
            "injector", "nozzle", "ring", "piston", "kit", "bolt", "nut",
        }
        _STOP = {"what","is","the","a","an","for","of","in","to","how","does","do","which",
                 "are","give","me","find","show","list","type","model","use","used","does"}
        _MODELS = {"c13","c9","c7","c15","c18","c12","c3","cat","caterpillar","3406"}
        generic = _LOOKUP_COMPONENTS | _STOP | _MODELS
        distinctive = query_toks - generic   # e.g. {"spark"} for "spark plug c13"

        def _relevant(text: str) -> bool:
            """
            Chunk must contain a part number co-located with the distinctive query
            terms. Two format-aware checks:

            1. Figure-legend format ("1=295-3099, ..., 1=SPARK PLUG"):
               both "1=<part_num>" and "1=<distinctive_tok>" exist in the same text.

            2. Table-row or prose: part number and distinctive token appear on the
               same line, or within 60 chars of each other.
            """
            if not _PART_NUM_RE.search(text):
                return False
            if not distinctive:
                return True
            t = text.lower()
            if not all(tok in t for tok in distinctive):
                return False  # fast exit

            # Check 1: figure-legend format ("1=295-3099 … 1=spark plug")
            if re.search(r"1=\d{3}-\d{4}", t):
                for tok in distinctive:
                    if re.search(r"1=" + re.escape(tok), t):
                        return True  # same item-group in figure legend

            # Check 2: same-line (parts table rows, structured prose)
            for line in t.splitlines():
                if _PART_NUM_RE.search(line) and all(tok in line for tok in distinctive):
                    return True

            # Check 3: close proximity (within 60 chars)
            for m in _PART_NUM_RE.finditer(t):
                window = t[max(0, m.start() - 10): m.end() + 60]
                if all(tok in window for tok in distinctive):
                    return True

            return False

        has_relevant = any(_relevant(r.get("text", "")) for r in final)
        if not has_relevant:
            all_pool = vector_results + keyword_results
            # Table row pattern: [item] [optional graphic ref] [part_num] [qty] [name...]
            _TABLE_ROW_RE = re.compile(
                r"^\s*\d+\s+(?:\d+\s+)?(\d{3}-\d{4})\s+\d+\s+(.+)$"
            )

            def _fallback_score(c: Dict) -> tuple:
                text = c.get("text", "")
                # Prefer genuine table-row format (part_num directly adjacent to name)
                # over figure-legend format where they can be 100+ chars apart.
                table_row_match = any(
                    (m := _TABLE_ROW_RE.match(line))
                    and all(tok in m.group(2).lower() for tok in distinctive)
                    for line in text.splitlines()
                )
                return (1 if table_row_match else 0, score_result(query, c))

            part_pool = [c for c in all_pool if _relevant(c.get("text", ""))]

            # If no table-row format found in the BM25/vector pool, scan raw chunk JSONs
            # directly — table children with noisy header tokens often rank too low for BM25.
            if not any(_fallback_score(c)[0] for c in part_pool):
                raw_chunks = load_all_chunk_files()
                for chunk in raw_chunks:
                    if chunk.get("chunk_type") != "child":
                        continue
                    text = chunk.get("text", "")
                    if _relevant(text) and _fallback_score(chunk_to_result(chunk))[0]:
                        resolved = resolve_to_parent(chunk_to_result(chunk))
                        part_pool.insert(0, resolved)  # table rows take priority
                        break  # one is enough

            part_pool.sort(key=_fallback_score, reverse=True)
            if part_pool:
                inject = deduplicate_results(part_pool, max_results=2)
                final = inject + final[:k - len(inject)]

    return final
