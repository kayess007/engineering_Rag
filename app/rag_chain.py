import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None
MAX_CONTEXT_CHARS = 80_000


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
) -> Dict[str, Any]:
    client = get_client()
    context = build_context(retrieved_chunks)

    system_prompt = """
You are an engineering assistant answering questions from OEM manuals.

Rules:
1. Use only the provided context.
2. Do not invent steps, values, or part numbers.
3. If the context is insufficient, say so clearly.
4. Prefer concise, technical answers.
5. When possible, mention the source file, section, and page range used.
6. If the retrieved chunks do not directly answer the question, say that the answer is uncertain.
"""

    user_prompt = f"""
Question:
{question}

Retrieved context:
{context}

Return your answer in this format:

Answer:
<clear technical answer>

Sources:
- <source file | section | page range>
- <source file | section | page range>
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
        "used_chunks": retrieved_chunks,
    }
