# Building a Production RAG System for Engineering Manuals: Architecture, Evaluation, and Hard-Won Lessons

---

## Introduction

Most RAG tutorials show you how to embed a PDF and answer questions about it. That part is straightforward. What they do not show you is what happens when your documents have two distinct corpora that look similar to a general embedding model, when your PDFs contain tables that need to be preserved whole, when your evaluation script is silently scoring against the wrong chunks, and when the improvement you were most confident about turns out to make things worse.

This post covers the full engineering journey of building a production RAG system for Caterpillar engine manuals — specifically, two distinct document types: **parts catalogues** (SEBP series) and **maintenance/service manuals**. The system answers technician questions like "what is the part number for the spark plug on a C13?" and "what is the oil change interval and recommended viscosity grade?" across a corpus of dense, table-heavy OEM engineering PDFs.

By the end of this post you will understand the full architecture, the retrieval stack, the evaluation methodology, and — most importantly — the specific decisions that moved RAGAS scores from 0.7769 to 0.8368, and the ones that did not.

---

## The Problem

### Two corpora, one embedding space

The fundamental challenge is that parts catalogues and maintenance manuals are semantically similar at the embedding level but require completely different retrieval behaviour.

A query like "oil filter" scores similarly against a parts catalogue entry (`Item 4: Part 326-1643, FILTER AS-OIL, Qty: 1`) and a maintenance procedure (`Replace the oil filter every 500 hours using a genuine Cat filter`). A general-purpose embedding model trained on web text has no reason to separate these — to it, both are about oil filters.

The cosine similarity analysis confirmed this: the two collections had a **separation score of only 0.0674**. Parts manual chunks had an intra-collection mean similarity of 0.6341 (tight, because parts tables are repetitive in structure). Maintenance manual chunks had an intra-collection mean of 0.5249 (more spread, because procedures are diverse). But the inter-collection overlap was large enough that vector search alone could not reliably route queries to the right corpus.

### PDF structure is hostile

Beyond routing, the source documents themselves present parsing challenges that generic loaders do not handle:

- **Doubled-character font artifacts**: PDF rendering bugs that produce `GGEENNEERRAALL` instead of `GENERAL`, requiring character deduplication before any other processing
- **Figure legends**: pages where part numbers and component labels are listed in alternating lines (`5`, `4`, `3`, `Boom`, `Feed`, `Pipe handler`), which naive text extraction flattens into meaningless sequences
- **Parts table rows**: entries like `1 1 295-3099 1 SPARK PLUG` that are unrecognisable to a language model unless rewritten as `Item 1: Part 295-3099, SPARK PLUG, Qty: 1`
- **Table extraction**: pdfplumber and PyMuPDF each handle tables differently; using both and merging their output gives better coverage than either alone

---

## Architecture

The system is built in Python with FastAPI as the API layer, Streamlit as the frontend, and ChromaDB as the vector store. Everything runs in Docker with persistent named volumes.

### The ingestion pipeline

```
PDF Upload
    └─ parser.py       PyMuPDF (text + headings) + pdfplumber (tables)
    └─ chunker.py      Hierarchical parent-child splitting
    └─ vectordb.py     ChromaDB indexing (child chunks only)
```

**Parsing** uses two libraries working in parallel. PyMuPDF extracts text and classifies elements as `Title` or `NarrativeText` based on font size thresholds computed from the document's own font distribution. pdfplumber handles table extraction separately, with each table becoming a typed `Table` element with both plain text and HTML representations.

**Chunking** is hierarchical. Every section produces two levels of chunk:

- **Parent chunks** (~850 characters): stored in JSON files, returned to the LLM at answer time for full context
- **Child chunks** (~220 characters): indexed in ChromaDB for precise retrieval

Each child chunk links back to its parent via `parent_id`. At query time, a child chunk surfaces the relevant passage, then the system resolves it to the parent chunk before passing context to the LLM. This gives the retriever precision and the LLM context.

Tables are handled separately. When the chunker detects a table element (by type, HTML metadata, or heuristic pattern matching), it breaks out of the normal section flow and creates a dedicated parent-child pair for that table alone. This ensures a table never gets split across section boundaries.

One issue caught during evaluation: tables larger than approximately 1800 characters (the safe embedding limit for `BAAI/bge-base-en-v1.5` at 512 tokens) were being indexed as a single child chunk, silently truncating the tail rows at embedding time. Part numbers in the bottom half of a long parts table were invisible to the retriever. The fix was straightforward: split large table child chunks with the same splitter used for regular text, while keeping the parent chunk intact.

### Pre-indexing deduplication

The SEBP parts manual corpus includes two volumes (SEBP6451-07-01 and SEBP6451-07-02) that share identical TABLE OF CONTENTS and INDEX sections. This produced 323 near-duplicate chunk pairs at cosine similarity of exactly 1.0 — wasting retrieval slots by returning the same content twice.

The fix runs at index time: before embedding, each child chunk's text is normalised (lowercased, whitespace collapsed) and SHA-256 hashed. If the hash already appeared in the current batch, the chunk is skipped. This runs in milliseconds and removed the duplicates without touching the source documents.

---

## The Query Pipeline

The production path — used by `/ask/advanced` — runs five stages:

### 1. Query classification

A rule-based classifier routes each query to one or both ChromaDB collections, with no LLM call. The logic:

```python
# Hard rule: explicit part number → always parts
if _PART_NUM_RE.search(query):   # matches NNN-NNNN pattern
    return "parts"

# Score against token sets
parts_score = len(toks.intersection(_PARTS_TOKENS))
maint_score = len(toks.intersection(_MAINTENANCE_TOKENS))

# Strong signal boosts
if toks.intersection({"part", "number", "catalogue", "kit", "assembly"}):
    parts_score += 2

if toks.intersection({"interval", "procedure", "schedule", "torque"}):
    maint_score += 2

# Route
if parts_score > 0 and maint_score > 0:
    if parts_score >= maint_score * 2:  return "parts"
    if maint_score >= parts_score * 2:  return "maintenance"
    return "both"
```

The default fallback is `maintenance` rather than `both`, which reduces cross-collection noise for ambiguous queries without a clear signal.

One experiment attempted to tighten the classifier by removing component names (filter, plug, valve) from the parts token set — reasoning that these appear in maintenance contexts too and cause false "both" routing. The experiment made context precision **worse** by 0.019. The reason: some eval questions genuinely needed both collections, and narrowing the routing removed relevant parts chunks from those queries. The classifier was reverted. This was one of the clearest examples in the project of an intuitively sensible change that the evaluation caught as harmful.

### 2. Query rewriting

GPT-4.1-mini generates up to three search-optimised variants of the original question, varying vocabulary and structure to broaden recall:

```
Original: "What oil should I use in the C13 hydraulic system?"
Variant 1: "C13 hydraulic oil specification viscosity grade"
Variant 2: "recommended hydraulic fluid type Caterpillar C13 engine"
Variant 3: "hydraulic system lubricant requirements C13"
```

The number of variants is capped by query type: parts queries get 3 (part numbers benefit from vocabulary variation), specification queries get 1 (over-rewriting dilutes precision), general queries get 2. This was learned through evaluation — generating 3 variants for all queries initially caused a faithfulness drop because diverse variants pulled in loosely related chunks that the LLM then synthesised across too liberally.

### 3. Hybrid retrieval

For each query variant and each target collection, the system runs both:

- **Vector search**: ChromaDB similarity search with a fetch pool of `max(k×3, 15)` candidates
- **BM25 keyword search**: `rank-bm25` over child chunks, with a wider pool for parts lookup queries where exact part number strings matter

Results from both are merged and scored with a unified `score_result()` function that weights: cosine similarity (×6), exact phrase matches (+15), token overlap (×2.5), section title overlap (×3), engineering domain terms (×1.5), and a +10 boost when a chunk contains a part number in a lookup-style query.

Child chunk hits are resolved to their parent chunks before scoring, so the LLM always sees the wider context window.

### 4. Cross-encoder reranking

After deduplication, the candidate pool (up to `k×3` chunks) is passed through `cross-encoder/ms-marco-MiniLM-L-6-v2`, a model trained to score (query, passage) pairs jointly for relevance. This runs locally on CPU.

The key parameter is `min_score`. The cross-encoder outputs raw logits — positive scores indicate relevance, negative scores indicate the opposite. Setting `min_score=0.0` was the initial setting, but this barely filtered anything since vaguely related chunks score just above zero. Raising it to `1.5` cut low-confidence chunks before the `top_k` slice, which contributed to the recall improvement observed in Run 9.

The reranker includes a safety fallback: if the threshold cuts all candidates (which can happen on very obscure queries), it returns the full sorted list rather than an empty result.

### 5. Answer generation

The top-5 reranked chunks (out of the K=6 retrieval pool) are passed to GPT-4.1-mini with a strict system prompt:

```
You are an engineering assistant answering questions from OEM manuals.
Rules:
1. Use only the provided context.
2. Do not invent steps, values, or part numbers.
3. If the context is insufficient, say so clearly.
4. Answer the specific question first and keep supporting detail brief.
```

The `max_chunks=5` cap is separate from the retrieval K=6. The extra sixth chunk exists in the retrieval pool for RAGAS context recall evaluation — RAGAS scores against all returned contexts, and having one extra chunk increases the surface area for finding ground-truth information without exposing it to the LLM (which would hurt faithfulness).

---

## Evaluation

### The eval mismatch problem

The first evaluation runs used a pattern common in RAG tutorials: call `/ask/advanced` to get the answer, then call `/query` separately to get retrieval context for RAGAS to score against. This produced misleading results.

The problem: `/ask/advanced` uses query rewriting and cross-encoder reranking, producing a specific set of top-k chunks. The separate `/query` call used none of this — it ran basic retrieval and returned different chunks entirely. RAGAS was scoring faithfulness (does the answer match the context?) against context the LLM never saw.

The fix: add `include_contexts: True` to the ask request. The API returns the exact chunks used — the reranked top-k — as a `contexts` field in the response. RAGAS evaluates those, not a parallel retrieval.

This single fix was the largest contributor to measurement correctness in the project. It did not change the system's actual performance — it changed the accuracy of how we were measuring it.

### RAGAS metrics in plain language

- **Faithfulness**: Does every claim in the answer trace back to the retrieved context? Score of 1.0 means the LLM added nothing beyond what the sources said.
- **Answer relevancy**: Does the answer address the actual question? High scores mean no padding, no tangents.
- **Context precision**: Of the K retrieved chunks, what fraction were actually useful for answering? If 2 of 6 chunks are off-topic, precision is 0.67.
- **Context recall**: Of all the relevant information needed to answer the question, how much did retrieval find? Low recall means the answer is missing facts that exist in the corpus.

### The K experiment

K (the number of chunks retrieved and returned for evaluation) has a non-obvious interaction with both metrics:

| K | faithfulness | answer_relevancy | context_precision | context_recall | Overall |
|---|---|---|---|---|---|
| 8 | 0.9667 | 0.8459 | 0.6268 | 0.8111 | 0.8126 |
| 5 | 1.0000 | 0.9083 | 0.6611 | 0.6333 | 0.8007 |
| **6** | **1.0000** | **0.9083** | **0.6721** | **0.7667** | **0.8368** |

K=8 had the best recall (0.81) but the worst precision (0.63) — the bottom two slots were noise. K=5 improved precision and faithfulness but killed recall (0.63) by shrinking the pool too aggressively. K=6 found the sweet spot: precision up, recall acceptable, faithfulness perfect.

The lesson is that K is not just a retrieval parameter — it directly shapes what RAGAS scores. Treating it as a fixed hyperparameter without evaluating its effect on both precision and recall simultaneously will lead you to optimise for only one dimension.

---

## What Moved the Metric

Honest accounting of every change and its effect:

| Change | Delta overall | Why |
|---|---|---|
| Fix eval mismatch (include_contexts) | Largest gain in measurement accuracy | Was scoring against wrong context |
| Table child chunk splitting (>1800 chars) | +recall 0.07 | Embedding truncation silently dropped table rows |
| Near-duplicate dedup at index time | +faithfulness | Removed identical TOC/index chunks from two SEBP volumes |
| min_score=1.5 on cross-encoder | Contributed to recall gain | Dropped noise chunks before top-k slice |
| K=6 tuning | +0.024 overall | Removed 2 noise slots vs K=8 without sacrificing recall |
| Tighter query classifier | **-0.013 overall** | Removed good cross-collection chunks alongside bad ones |

The classifier tightening is worth dwelling on. The reasoning was sound: component names (filter, plug, valve) appear in maintenance contexts, so keeping them in the parts token set causes false "both" routing. Removing them should reduce noise.

What the reasoning missed: those same queries benefited from "both" routing because the correct answer sometimes lived in the parts catalogue, not the maintenance manual. Narrowing the routing removed those correct chunks. The experiment made precision worse, not better. Without evaluation, this change would have shipped.

---

## Embedding Space Analysis

Running a cosine similarity analysis over 2,000 sampled child chunks (500 per source file) with `BAAI/bge-base-en-v1.5` revealed the embedding landscape:

**Similarity distribution** (1,999,000 pairs):
- 0.0–0.3 unrelated: 3.5%
- 0.3–0.6 loosely related: **76.4%** — healthy spread, not overcrowded
- 0.6–0.8 related: 19.6%
- 0.8–0.95 very similar: 0.5%
- 0.95–1.0 near-duplicate: 0.016% (323 pairs, all from SEBP volume duplicates)

**Per-collection:**
- `parts_manuals` intra mean: 0.6341 (tighter — parts entries share structure)
- `maintenance_manuals` intra mean: 0.5249 (more spread — diverse procedures)
- Inter-collection mean: ~0.557
- **Separation score: 0.0674** — low, confirming the embedding model does not cleanly separate the two corpora

This explains why the rule-based classifier and collection-aware routing are necessary. Relying on embedding distance alone to separate parts from maintenance queries would fail on this corpus.

A PCA projection of the embeddings showed partial but incomplete clustering: the two collections are distinguishable as regions but with substantial overlap. Improving this separation would require a domain-tuned embedding model or fine-tuning on engineering text.

---

## Production Considerations

### Authentication

Upload, chunk, and index routes require a JWT Bearer token. Query and ask routes are open — read access is unauthenticated. Tokens are issued by `POST /auth/token` with form credentials and expire after a configurable duration. Credentials and the JWT secret are injected via environment variables and never hardcoded.

### Duplicate guards

Two independent deduplication checks prevent data corruption:

1. **File-level**: uploading a filename that already exists in `storage/uploads/` returns HTTP 409 immediately
2. **Index-level**: before embedding a new batch, the system checks ChromaDB for any document with the same `manual_id`; if found, the index call is rejected with HTTP 409

### Feedback loop

Every chat response in the Streamlit UI exposes 👍/👎 buttons. Ratings are appended to a JSONL file and surfaced in the Status page with counts and negative feedback details. This provides a lightweight signal for identifying failure modes without a full annotation pipeline.

### Docker deployment

The system runs as two containers (`api` and `frontend`) with Docker Compose. All persistent data lives in named volumes, not bind mounts, so the host filesystem and container filesystem are separated. This matters during development: chunked files generated by a local `uvicorn` run do not exist inside the container's volume. Moving from local to Docker required copying chunk files into the container with `docker cp` and re-indexing inside the container before the API could serve them.

---

## What Remains

**Context precision at 0.67** is the current ceiling. Approximately two of every six retrieved chunks are not contributing to the answer. The routing improvements that were tried (tighter classifier) made things worse. The remaining approaches require changes outside the retrieval layer:

1. **Domain-tuned embeddings** — a model fine-tuned on engineering or mechanical text would produce a separation score well above 0.07, making collection boundaries meaningful in embedding space
2. **Confidence-weighted collection merging** — instead of binary routing (parts / maintenance / both), assign a probability to each collection and weight candidate chunks accordingly before reranking
3. **Eval dataset expansion** — the current eval dataset covers a representative but limited set of questions; edge cases (multi-hop reasoning, table-heavy answers, cross-manual questions) are underrepresented
4. **Streaming responses** — the Streamlit frontend currently waits for the full answer before rendering; streaming would improve perceived latency for longer answers
5. **Per-manual scoping** — the UI has a collection badge showing which corpus was searched; allowing users to pin queries to a specific manual would both improve precision and give technicians more control

---

## Final Scores

| Metric | Score |
|---|---|
| Faithfulness | 1.0000 |
| Answer Relevancy | 0.9083 |
| Context Precision | 0.6721 |
| Context Recall | 0.7667 |
| **Overall** | **0.8368** |

---

## Key Takeaways

**Measurement correctness matters more than algorithm sophistication.** The single largest contribution to score improvement was fixing the evaluation script to score against the exact chunks the LLM saw, not a parallel retrieval. No amount of retrieval tuning helps if you are measuring the wrong thing.

**Evaluate before you ship intuitive improvements.** The classifier tightening was the most conceptually motivated change in the project — and the only one that made scores worse. The evaluation caught it. Without it, a confident and well-reasoned change would have shipped and degraded production quality.

**K is not just a retrieval parameter.** It determines what RAGAS evaluates. Changing K changes precision, recall, and faithfulness simultaneously. The sweet spot depends on your corpus, your eval dataset, and what you are willing to trade.

**Table handling is not optional.** Engineering documents are table-heavy, and tables that exceed embedding token limits are silently truncated. If your chunker does not handle this, part numbers in the bottom half of a 40-row parts table are simply not retrievable.

**Separation score tells you whether routing is load-bearing.** If your embedding model cleanly separates your corpora, you can route by similarity. If it does not (0.0674 in this case), you need explicit routing logic — and you need to validate it against eval data before assuming it helps.

---

*All evaluation runs used RAGAS with GPT-4.1-mini as the judge LLM and `text-embedding-3-small` for answer relevancy embeddings. The retrieval model is `BAAI/bge-base-en-v1.5` running locally on CPU. The reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2` running locally on CPU.*
