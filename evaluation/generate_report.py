"""
Evaluation Report Generator
Run from project root: python evaluation/generate_report.py
Requires the FastAPI server running on http://127.0.0.1:8000

Generates evaluation/results/report_YYYYMMDD_HHMMSS.html
Open the HTML file in any browser to review all 15 questions
with full source text for verification.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

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


def query_chunks(question: str) -> list:
    r = requests.post(
        f"{API_BASE}/query",
        json={"question": question, "k": K},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("results", [])


def score_label(text: str, ground_truth: str) -> str:
    """Simple keyword overlap check to flag potential misses."""
    gt_words = set(ground_truth.lower().split())
    answer_words = set(text.lower().split())
    overlap = len(gt_words & answer_words) / max(len(gt_words), 1)
    if overlap >= 0.4:
        return "likely_match"
    elif overlap >= 0.2:
        return "partial_match"
    return "possible_miss"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Engineering RAG — Evaluation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f5f5f5; color: #222; line-height: 1.6; }}
  .header {{ background: #1a1a2e; color: white; padding: 32px 40px; }}
  .header h1 {{ font-size: 24px; font-weight: 700; }}
  .header .meta {{ margin-top: 8px; font-size: 14px; opacity: 0.75; }}
  .summary {{ background: white; border-bottom: 1px solid #e0e0e0;
              padding: 20px 40px; display: flex; gap: 40px; }}
  .stat {{ text-align: center; }}
  .stat .value {{ font-size: 28px; font-weight: 700; color: #1a1a2e; }}
  .stat .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
  .container {{ max-width: 1100px; margin: 32px auto; padding: 0 20px; }}
  .question-card {{ background: white; border-radius: 8px; margin-bottom: 28px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow: hidden; }}
  .q-header {{ padding: 16px 24px; display: flex; align-items: center; gap: 12px;
               border-bottom: 1px solid #f0f0f0; }}
  .q-num {{ background: #1a1a2e; color: white; border-radius: 50%;
             width: 32px; height: 32px; display: flex; align-items: center;
             justify-content: center; font-weight: 700; font-size: 14px;
             flex-shrink: 0; }}
  .q-text {{ font-size: 16px; font-weight: 600; }}
  .badge {{ padding: 4px 10px; border-radius: 12px; font-size: 11px;
             font-weight: 600; text-transform: uppercase; margin-left: auto;
             flex-shrink: 0; }}
  .likely_match {{ background: #d4edda; color: #155724; }}
  .partial_match {{ background: #fff3cd; color: #856404; }}
  .possible_miss {{ background: #f8d7da; color: #721c24; }}
  .q-body {{ padding: 20px 24px; display: grid; gap: 16px; }}
  .section-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase;
                    color: #888; margin-bottom: 6px; letter-spacing: 0.5px; }}
  .answer-box {{ background: #f0f7ff; border-left: 4px solid #2196f3;
                  padding: 14px 16px; border-radius: 0 4px 4px 0; }}
  .ground-truth-box {{ background: #f0fff4; border-left: 4px solid #4caf50;
                        padding: 14px 16px; border-radius: 0 4px 4px 0; }}
  .chunks-section {{ margin-top: 4px; }}
  .chunk {{ background: #fafafa; border: 1px solid #e8e8e8; border-radius: 6px;
             margin-bottom: 10px; overflow: hidden; }}
  .chunk-meta {{ background: #f0f0f0; padding: 8px 14px; font-size: 12px;
                  color: #555; display: flex; gap: 16px; flex-wrap: wrap; }}
  .chunk-meta span {{ display: flex; align-items: center; gap: 4px; }}
  .chunk-meta .tag {{ background: #ddd; padding: 2px 8px; border-radius: 10px;
                       font-size: 11px; font-weight: 600; }}
  .chunk-text {{ padding: 14px; font-size: 13px; line-height: 1.7;
                  white-space: pre-wrap; word-break: break-word; color: #333; }}
  .no-chunks {{ color: #999; font-style: italic; font-size: 13px; }}
  .footer {{ text-align: center; padding: 40px; color: #aaa; font-size: 12px; }}
  @media print {{
    .question-card {{ page-break-inside: avoid; }}
    body {{ background: white; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Engineering RAG — Evaluation Report</h1>
  <div class="meta">
    Generated: {timestamp} &nbsp;|&nbsp;
    Model: {model} &nbsp;|&nbsp;
    k = {k} &nbsp;|&nbsp;
    Questions: {total}
  </div>
</div>

<div class="summary">
  <div class="stat">
    <div class="value">{total}</div>
    <div class="label">Total Questions</div>
  </div>
  <div class="stat">
    <div class="value" style="color:#155724">{likely}</div>
    <div class="label">Likely Match</div>
  </div>
  <div class="stat">
    <div class="value" style="color:#856404">{partial}</div>
    <div class="label">Partial Match</div>
  </div>
  <div class="stat">
    <div class="value" style="color:#721c24">{miss}</div>
    <div class="label">Possible Miss</div>
  </div>
</div>

<div class="container">
{cards}
</div>

<div class="footer">
  Engineering RAG Evaluation Report &mdash; {timestamp}
</div>

</body>
</html>"""


CARD_TEMPLATE = """
<div class="question-card">
  <div class="q-header">
    <div class="q-num">{num}</div>
    <div class="q-text">{question}</div>
    <div class="badge {score_class}">{score_label}</div>
  </div>
  <div class="q-body">

    <div>
      <div class="section-label">RAG Answer</div>
      <div class="answer-box">{answer}</div>
    </div>

    <div>
      <div class="section-label">Ground Truth</div>
      <div class="ground-truth-box">{ground_truth}</div>
    </div>

    <div class="chunks-section">
      <div class="section-label">Retrieved Source Chunks ({chunk_count})</div>
      {chunks_html}
    </div>

  </div>
</div>
"""


CHUNK_TEMPLATE = """
<div class="chunk">
  <div class="chunk-meta">
    <span><strong>#{idx}</strong></span>
    <span>📄 {source_file}</span>
    <span>📌 {section_title}</span>
    <span>📃 Page {page}</span>
    <span><span class="tag">{content_type}</span></span>
  </div>
  <div class="chunk-text">{text}</div>
</div>
"""


def build_chunk_html(chunks: list) -> str:
    if not chunks:
        return '<div class="no-chunks">No chunks retrieved.</div>'

    html = ""
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "").strip()
        source_file = meta.get("source_file") or "unknown"
        section_title = meta.get("section_title") or "—"
        page_start = meta.get("page_start")
        page_end = meta.get("page_end")
        content_type = meta.get("content_type") or "section"

        if page_start and page_end and page_start != page_end:
            page = f"{page_start}–{page_end}"
        elif page_start:
            page = str(page_start)
        else:
            page = "—"

        html += CHUNK_TEMPLATE.format(
            idx=i,
            source_file=source_file,
            section_title=section_title,
            page=page,
            content_type=content_type,
            text=text.replace("<", "&lt;").replace(">", "&gt;"),
        )

    return html


def main():
    print("=" * 60)
    print("Engineering RAG — Evaluation Report Generator")
    print("=" * 60)

    if not check_server():
        print("ERROR: FastAPI server not running at", API_BASE)
        print("Start it with: python -m uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print(f"Loaded {len(eval_items)} questions\n")

    cards = []
    counts = {"likely_match": 0, "partial_match": 0, "possible_miss": 0}

    for i, item in enumerate(eval_items, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i:02d}/{len(eval_items)}] {question[:70]}...")

        try:
            ask_result = ask(question)
            answer = ask_result.get("answer", "No answer returned.")
        except Exception as e:
            answer = f"ERROR: {e}"
            print(f"         ask() failed: {e}")

        try:
            chunks = query_chunks(question)
        except Exception as e:
            chunks = []
            print(f"         query() failed: {e}")

        label = score_label(answer, ground_truth)
        counts[label] += 1
        print(f"         {label} | {len(chunks)} chunks retrieved")

        label_display = label.replace("_", " ").title()
        chunks_html = build_chunk_html(chunks)

        card = CARD_TEMPLATE.format(
            num=i,
            question=question.replace("<", "&lt;").replace(">", "&gt;"),
            answer=answer.replace("<", "&lt;").replace(">", "&gt;"),
            ground_truth=ground_truth.replace("<", "&lt;").replace(">", "&gt;"),
            score_class=label,
            score_label=label_display,
            chunk_count=len(chunks),
            chunks_html=chunks_html,
        )
        cards.append(card)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = HTML_TEMPLATE.format(
        timestamp=timestamp,
        model=MODEL,
        k=K,
        total=len(eval_items),
        likely=counts["likely_match"],
        partial=counts["partial_match"],
        miss=counts["possible_miss"],
        cards="\n".join(cards),
    )

    output_path = RESULTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("\n" + "=" * 60)
    print(f"Report saved to: {output_path}")
    print(f"Open in browser: file:///{output_path.resolve()}")
    print("=" * 60)
    print(f"Likely match:   {counts['likely_match']}")
    print(f"Partial match:  {counts['partial_match']}")
    print(f"Possible miss:  {counts['possible_miss']}")


if __name__ == "__main__":
    main()
