"""
Phase 9 — Query Rewriting
Generates multiple search-optimised variants of the user's question using an LLM.
This broadens recall before the cross-encoder narrows it back down.
"""

import os
import re
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


_SYSTEM_PROMPT = """\
You are a search query optimiser for an engineering manual RAG system.
Given the user's question, produce exactly 3 search queries that will maximise \
retrieval of relevant technical passages.

Rules:
- Each query must be on its own line, prefixed with a number and period (1. 2. 3.)
- Vary vocabulary: use synonyms, expand abbreviations, rephrase as keyword strings
- Keep engineering domain terms precise (e.g. viscosity, torque, cleanliness class)
- Do NOT explain or add any text outside the numbered list
"""


def rewrite_query(question: str, model: str = "gpt-4.1-mini") -> list[str]:
    """
    Returns [original_question] + up to 3 LLM-generated query variants.
    Falls back to [original_question] on any error so retrieval is never blocked.
    """
    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content or ""
        variants = []
        for line in raw.splitlines():
            line = line.strip()
            # Strip leading "1. " / "2. " etc.
            cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
            if cleaned and cleaned.lower() != question.lower():
                variants.append(cleaned)
        # Always include original first; deduplicate while preserving order
        seen = {question.lower()}
        queries = [question]
        for v in variants:
            if v.lower() not in seen:
                seen.add(v.lower())
                queries.append(v)
        return queries[:4]  # original + up to 3 variants
    except Exception as e:
        print(f"[query_rewriter] LLM rewrite failed ({e}), using original query.")
        return [question]
