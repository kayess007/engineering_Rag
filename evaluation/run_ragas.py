"""
Phase 6 — RAGAS Evaluation
Run from project root: python evaluation/run_ragas.py
Requires the FastAPI server to be running on http://127.0.0.1:8000
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # noqa: E402
from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbeddings
from openai import OpenAI as OpenAIClient


class _SyncOpenAIEmbeddings(BaseRagasEmbeddings):
    """Thin wrapper that provides the sync embed_query/embed_documents ragas needs."""

    def __init__(self, model: str, client: OpenAIClient):
        self._model = model
        self._client = client

    def embed_query(self, text: str) -> list:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding

    def embed_documents(self, texts: list) -> list:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]

    async def aembed_query(self, text: str) -> list:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list) -> list:
        return self.embed_documents(texts)

API_BASE = "http://127.0.0.1:8000"
EVAL_DATASET_PATH = Path("evaluation/eval_dataset.json")
RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K = 6  # retrieval pool — reranker selects top-5 for the LLM (max_chunks=5 in rag_chain)
MODEL = "gpt-4.1-mini"
USE_ADVANCED = True   # set False to fall back to /ask for comparison runs
RESPONSE_MODE = "concise"


def check_server() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ask(question: str) -> dict:
    endpoint = "/ask/advanced" if USE_ADVANCED else "/ask"
    r = requests.post(
        f"{API_BASE}{endpoint}",
        json={
            "question": question,
            "k": K,
            "model": MODEL,
            "response_mode": RESPONSE_MODE,
            "include_contexts": True,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def query(question: str) -> list:
    r = requests.post(
        f"{API_BASE}/query",
        json={"question": question, "k": K},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["results"]


def extract_context_texts(ask_result: dict) -> list[str]:
    """
    Accept both API shapes for contexts:
    - newer: [{"text": "...", "metadata": {...}}, ...]
    - older: ["...", "..."]
    """
    contexts = []
    for item in ask_result.get("contexts", []):
        if isinstance(item, dict):
            text = item.get("text", "")
            if text:
                contexts.append(text)
        elif isinstance(item, str) and item.strip():
            contexts.append(item)
    return contexts


def summarize_context_payload(ask_result: dict) -> str:
    items = ask_result.get("contexts", [])
    if not isinstance(items, list):
        return f"non-list:{type(items).__name__}"
    if not items:
        return "empty-list"

    first = items[0]
    if isinstance(first, dict):
        keys = ",".join(sorted(first.keys()))
        return f"dict-list len={len(items)} first_keys=[{keys}]"
    if isinstance(first, str):
        return f"string-list len={len(items)}"
    return f"mixed-or-unknown len={len(items)} first_type={type(first).__name__}"


def main():
    mode = "advanced (/ask/advanced)" if USE_ADVANCED else "standard (/ask)"
    print("=" * 60)
    print(f"Engineering RAG — RAGAS Evaluation  [{mode}, k={K}]")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    if not check_server():
        print("ERROR: FastAPI server not running at", API_BASE)
        print("Start it with: uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    print("Server: OK")

    dataset_path = EVAL_DATASET_PATH
    if not dataset_path.exists():
        print(f"ERROR: eval_dataset.json not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print(f"Loaded {len(eval_items)} evaluation questions\n")

    samples = []
    raw_results = []

    for i, item in enumerate(eval_items, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(eval_items)}] {question[:70]}...")

        try:
            # Get answer + the exact chunks the LLM used from /ask/advanced
            ask_result = ask(question)
            answer = ask_result.get("answer", "")

            if i == 1:
                payload_summary = summarize_context_payload(ask_result)
                print(f"         Context payload shape: {payload_summary}")

            # Use contexts returned by the endpoint — same chunks used to generate the answer.
            # /ask/advanced returns "contexts" (full text of top-5 reranked chunks).
            # Fall back to a separate /query call only for the standard /ask endpoint.
            if USE_ADVANCED:
                contexts = extract_context_texts(ask_result)
                if not contexts:
                    # Defensive fallback in case the API is running an older backend
                    # version that ignores include_contexts or returns an unexpected shape.
                    retrieved = query(question)
                    contexts = [r["text"] for r in retrieved if r.get("text")]
            else:
                retrieved = query(question)
                contexts = [r["text"] for r in retrieved if r.get("text")]

            print(f"         Retrieved {len(contexts)} chunks, answer length: {len(answer)} chars")

            samples.append(
                SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                    reference=ground_truth,
                )
            )

            raw_results.append(
                {
                    "question": question,
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "source_file": item.get("source_file"),
                    "contexts": contexts,
                }
            )

        except Exception as e:
            print(f"         FAILED: {e}")
            continue

    if not samples:
        print("No samples collected. Exiting.")
        sys.exit(1)

    print(f"\nCollected {len(samples)} samples. Running RAGAS evaluation...\n")

    openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4.1-mini", client=openai_client)
    embeddings = _SyncOpenAIEmbeddings(model="text-embedding-3-small", client=openai_client)

    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm = llm
    context_recall.llm = llm

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    eval_dataset = EvaluationDataset(samples=samples)

    results = evaluate(dataset=eval_dataset, metrics=metrics)

    # Print results table
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    scores = {}
    df = results.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for col in metric_cols:
        if col in df.columns:
            val = float(df[col].dropna().mean())
            scores[col] = round(val, 4)
            print(f"  {col:<25} {val:.4f}")

    print("=" * 60)
    overall = sum(scores.values()) / len(scores) if scores else 0
    print(f"  {'Overall Average':<25} {overall:.4f}")
    print("=" * 60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": MODEL,
        "k": K,
        "mode": "advanced" if USE_ADVANCED else "standard",
        "sample_count": len(samples),
        "scores": scores,
        "overall_average": round(overall, 4),
        "raw_results": raw_results,
    }

    output_path = RESULTS_DIR / f"ragas_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    return scores


if __name__ == "__main__":
    main()
