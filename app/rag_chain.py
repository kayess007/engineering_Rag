import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None
MAX_CONTEXT_CHARS = 40_000


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def build_context(results: List[Dict[str, Any]]) -> str:
    context_parts = []

    for i, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})
        text = item.get("text", "").strip()

        source = (
            f"Source {i} | "
            f"file: {metadata.get('source_file', 'unknown')} | "
            f"section: {metadata.get('section_title', 'unknown')} | "
            f"pages: {metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}"
        )

        context_parts.append(f"{source}\n{text}")

    context = "\n\n".join(context_parts)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated to fit model limits]"

    return context


def generate_rag_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    model: str = "gpt-4.1-mini",
    max_chunks: int = 5,
    response_mode: str = "standard",
) -> Dict[str, Any]:
    client = get_client()
    selected_chunks = retrieved_chunks[:max_chunks]
    context = build_context(selected_chunks)

    if response_mode == "concise":
        system_prompt = """
You are an engineering assistant answering questions from OEM manuals.

Rules:
1. Use only the provided context.
2. Answer only what was asked, in 1-3 sentences.
3. Lead with the exact value or fact when it is directly stated.
4. Do not add adjacent recommendations, examples, or caveats unless the question asks for them.
5. If the context is insufficient, say so clearly.
6. Do not include headings, bullets, or a source list.
"""
    else:
        system_prompt = """
You are an engineering assistant answering questions from OEM manuals.

Rules:
1. Use only the provided context.
2. Do not invent steps, values, or part numbers.
3. If the context is insufficient, say so clearly.
4. Prefer concise, technical answers.
5. Answer the specific question first and keep supporting detail brief.
6. Do not include headings or a source list in the answer body.
"""

    user_prompt = f"""
Question:
{question}

Retrieved context:
{context}

Return a direct answer using only the retrieved context.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "used_chunks": selected_chunks,
    }
