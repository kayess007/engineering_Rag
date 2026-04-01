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
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
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

K = 5
MODEL = "gpt-4.1-mini"


def check_server() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ask(question: str) -> dict:
    r = requests.post(
        f"{API_BASE}/ask",
        json={"question": question, "k": K, "model": MODEL},
        timeout=60,
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


def main():
    print("=" * 60)
    print("Engineering RAG — Phase 6 RAGAS Evaluation")
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
            # Get answer from RAG
            ask_result = ask(question)
            answer = ask_result.get("answer", "")

            # Get retrieved contexts
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

    # Set up LLM and embeddings using ragas 0.4.x API
    openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4.1-mini", client=openai_client)
    embeddings = _SyncOpenAIEmbeddings(model="text-embedding-3-small", client=openai_client)

    # Use lowercase singleton metrics and assign llm/embeddings before passing
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
