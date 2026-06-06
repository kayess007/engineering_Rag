"""
Chunk Embedding Similarity Analysis
====================================
Loads child chunks from storage/chunked/, embeds them with BAAI/bge-base-en-v1.5,
and reports:
  1. Similarity distribution (histogram buckets)
  2. Near-duplicate pairs (cosine > 0.95)
  3. Per-collection spread (intra vs inter similarity)
  4. PCA scatter plot coloured by collection
  5. Top-10 nearest-neighbour pairs

Run from project root:
    python evaluation/chunk_similarity.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKED_DIR = Path("storage/chunked")
SAMPLE_PER_FILE = 500      # chunks sampled per source file
NEAR_DUP_THRESHOLD = 0.95  # cosine similarity considered near-duplicate
RANDOM_SEED = 42
PLOT_OUTPUT = Path("evaluation/results/chunk_similarity.png")

# Map source filename signals to collection label
def _collection_label(source_file: str) -> str:
    lower = (source_file or "").lower()
    if any(s in lower for s in ("sebp", "parts", "catalogue")):
        return "parts_manuals"
    return "maintenance_manuals"


def load_child_chunks(sample: int = SAMPLE_PER_FILE) -> list[dict]:
    random.seed(RANDOM_SEED)
    all_chunks = []
    files = sorted(CHUNKED_DIR.glob("*_chunks.json"))

    for f in files:
        chunks = json.loads(f.read_text(encoding="utf-8"))
        children = [
            c for c in chunks
            if c.get("chunk_type") == "child" and len(c.get("text", "").strip()) >= 50
        ]
        if not children:
            continue
        sampled = random.sample(children, min(sample, len(children)))
        for c in sampled:
            source = c.get("metadata", {}).get("source_file", f.stem)
            c["_collection"] = _collection_label(source)
            c["_source"] = source
        all_chunks.extend(sampled)
        print(f"  Loaded {len(sampled):>4} chunks from {f.name}")

    return all_chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print(f"\nEmbedding {len(chunks)} chunks with BAAI/bge-base-en-v1.5 …")
    model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    texts = [c["text"][:512] for c in chunks]   # safety truncation
    embeddings = model.embed_documents(texts)
    return np.array(embeddings, dtype=np.float32)


def cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    # Embeddings already L2-normalised → dot product = cosine similarity
    return embs @ embs.T


def similarity_stats(sim: np.ndarray, chunks: list[dict]):
    n = len(chunks)
    # Upper triangle only (exclude self-similarity on diagonal)
    idx = np.triu_indices(n, k=1)
    pairs = sim[idx]

    print("\n" + "=" * 60)
    print("SIMILARITY DISTRIBUTION")
    print("=" * 60)
    buckets = [
        (0.0,  0.3,  "unrelated      "),
        (0.3,  0.6,  "loosely related"),
        (0.6,  0.8,  "related        "),
        (0.8,  0.95, "very similar   "),
        (0.95, 1.0,  "near-duplicate "),
    ]
    for lo, hi, label in buckets:
        count = int(((pairs >= lo) & (pairs < hi)).sum())
        pct = count / len(pairs) * 100
        bar = "█" * int(pct / 2)
        print(f"  {lo:.2f}–{hi:.2f}  {label}  {bar:<25}  {count:>7,}  ({pct:.1f}%)")

    print(f"\n  Mean similarity : {pairs.mean():.4f}")
    print(f"  Std deviation   : {pairs.std():.4f}")
    print(f"  Median          : {np.median(pairs):.4f}")
    print(f"  Min             : {pairs.min():.4f}")
    print(f"  Max             : {pairs.max():.4f}")


def near_duplicate_report(sim: np.ndarray, chunks: list[dict], threshold: float = NEAR_DUP_THRESHOLD):
    n = len(chunks)
    idx_i, idx_j = np.triu_indices(n, k=1)
    mask = sim[idx_i, idx_j] >= threshold
    dup_i = idx_i[mask]
    dup_j = idx_j[mask]
    scores = sim[dup_i, dup_j]

    print("\n" + "=" * 60)
    print(f"NEAR-DUPLICATES  (cosine ≥ {threshold})")
    print("=" * 60)
    print(f"  Found {len(scores):,} near-duplicate pairs out of {len(idx_i):,} total")

    if len(scores) == 0:
        print("  No near-duplicates found.")
        return

    # Show top 10
    top = np.argsort(-scores)[:10]
    print(f"\n  Top {min(10, len(scores))} most similar pairs:")
    for rank, k in enumerate(top, 1):
        i, j = dup_i[k], dup_j[k]
        ci, cj = chunks[i], chunks[j]
        print(f"\n  [{rank}] similarity={scores[k]:.4f}")
        print(f"      A: [{ci['_source']}] {ci.get('section_title','')[:50]}")
        print(f"         {ci['text'][:100].strip()}…")
        print(f"      B: [{cj['_source']}] {cj.get('section_title','')[:50]}")
        print(f"         {cj['text'][:100].strip()}…")


def collection_spread(sim: np.ndarray, chunks: list[dict]):
    labels = np.array([c["_collection"] for c in chunks])
    collections = sorted(set(labels))

    print("\n" + "=" * 60)
    print("PER-COLLECTION SPREAD")
    print("=" * 60)

    for coll in collections:
        mask = labels == coll
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        # Intra-collection similarity
        sub = sim[np.ix_(idx, idx)]
        intra_idx = np.triu_indices(len(idx), k=1)
        intra = sub[intra_idx]
        print(f"\n  {coll}  ({len(idx)} chunks)")
        print(f"    Intra mean : {intra.mean():.4f}  (how similar chunks are to each other)")
        print(f"    Intra std  : {intra.std():.4f}")

    if len(collections) == 2:
        m0 = labels == collections[0]
        m1 = labels == collections[1]
        i0 = np.where(m0)[0]
        i1 = np.where(m1)[0]
        inter = sim[np.ix_(i0, i1)].flatten()
        print(f"\n  Inter-collection mean : {inter.mean():.4f}  (parts vs maintenance separation)")
        print(f"  Inter-collection std  : {inter.std():.4f}")
        separation = (
            (sim[np.ix_(i0, i0)][np.triu_indices(len(i0), k=1)].mean() +
             sim[np.ix_(i1, i1)][np.triu_indices(len(i1), k=1)].mean()) / 2
            - inter.mean()
        )
        print(f"  Separation score      : {separation:.4f}  (higher = collections are more distinct)")


def pca_plot(embs: np.ndarray, chunks: list[dict], output: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("\n  [plot skipped — matplotlib or sklearn not installed]")
        return

    print(f"\nGenerating PCA plot → {output}")
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(embs)

    labels = [c["_collection"] for c in chunks]
    collections = sorted(set(labels))
    colours = {"parts_manuals": "#2196F3", "maintenance_manuals": "#4CAF50"}
    default_colours = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, coll in enumerate(collections):
        mask = [l == coll for l in labels]
        x = coords[mask, 0]
        y = coords[mask, 1]
        colour = colours.get(coll, default_colours[i % len(default_colours)])
        ax.scatter(x, y, c=colour, label=coll, alpha=0.4, s=8)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("Chunk Embeddings — PCA Projection\n(BAAI/bge-base-en-v1.5, child chunks)")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output}")


def main():
    print("Loading child chunks…")
    chunks = load_child_chunks()
    print(f"\nTotal chunks loaded: {len(chunks)}")

    if len(chunks) < 10:
        print("Not enough chunks to analyse. Check storage/chunked/.")
        sys.exit(1)

    embs = embed_chunks(chunks)
    sim = cosine_similarity_matrix(embs)

    similarity_stats(sim, chunks)
    near_duplicate_report(sim, chunks)
    collection_spread(sim, chunks)
    pca_plot(embs, chunks, PLOT_OUTPUT)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
