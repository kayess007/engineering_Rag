import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


MIN_SECTION_CHARS = 80


def looks_like_table_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 2:
        return False

    score = 0

    for line in lines[:8]:
        if "|" in line:
            score += 1
        if "\t" in line:
            score += 1
        if "ppm" in line.lower():
            score += 1
        if "iso " in line.lower():
            score += 1
        if any(c.isdigit() for c in line) and any(c.isalpha() for c in line):
            score += 0.5

    return score >= 2


def load_parsed_json(parsed_file_path: str) -> Dict[str, Any]:
    with open(parsed_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunked_json(manual_id: str, chunks: List[Dict[str, Any]]) -> Path:
    output_dir = Path("storage/chunked")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{manual_id}_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return output_path


def is_meaningful_text(text: str) -> bool:
    return bool(text and text.strip())


def is_noisy_text(text: str) -> bool:
    """Filter obvious page furniture / noise."""
    lowered = text.lower().strip()

    noisy_patterns = [
        "copyright ©",
        "maintenance manual",
        "all rights reserved",
        "en-us",
    ]

    return any(pattern in lowered for pattern in noisy_patterns)


def get_parent_splitter() -> RecursiveCharacterTextSplitter:
    """Large chunks (1200 tokens) — sent to the LLM for full context."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )


def get_child_splitter() -> RecursiveCharacterTextSplitter:
    """Small chunks (400 tokens) — indexed in ChromaDB for precise retrieval."""
    return RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []

    for line in lines:
        if not line:
            continue
        if is_noisy_text(line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def make_semantic_chunks(
    manual_id: str,
    filename: str,
    section_title: str,
    combined_text: str,
    element_types: List[str],
    page_numbers: List[int],
) -> List[Dict[str, Any]]:
    """
    Hierarchical parent-child chunking:
    - Parent chunks (1200 tokens) carry full context for the LLM.
    - Child chunks (400 tokens) are indexed in ChromaDB for precise retrieval.
    - Each child links back to its parent via parent_id.
    """
    parent_splitter = get_parent_splitter()
    child_splitter = get_child_splitter()

    parent_texts = parent_splitter.split_text(combined_text)
    page_start = min(page_numbers) if page_numbers else None
    page_end = max(page_numbers) if page_numbers else None

    chunks = []

    for parent_text in parent_texts:
        cleaned_parent = clean_text(parent_text)
        if len(cleaned_parent) < MIN_SECTION_CHARS:
            continue

        parent_id = str(uuid.uuid4())

        # Parent chunk — stored in JSON, returned to LLM
        chunks.append(
            {
                "chunk_id": parent_id,
                "chunk_type": "parent",
                "manual_id": manual_id,
                "section_title": section_title or "Untitled Section",
                "content_type": "section",
                "page_start": page_start,
                "page_end": page_end,
                "text": cleaned_parent,
                "element_types": sorted(list(set(element_types))),
                "metadata": {
                    "source_file": filename,
                    "parent_section": section_title or "Untitled Section",
                },
            }
        )

        # Child chunks — indexed in ChromaDB for vector search
        child_texts = child_splitter.split_text(cleaned_parent)
        for idx, child_text in enumerate(child_texts, start=1):
            cleaned_child = clean_text(child_text)
            if len(cleaned_child) < MIN_SECTION_CHARS:
                continue

            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "manual_id": manual_id,
                    "section_title": section_title or "Untitled Section",
                    "content_type": "section",
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": cleaned_child,
                    "element_types": sorted(list(set(element_types))),
                    "metadata": {
                        "source_file": filename,
                        "parent_section": section_title or "Untitled Section",
                        "child_index": idx,
                    },
                }
            )

    return chunks


def finalize_section_chunk(
    chunks: List[Dict[str, Any]],
    manual_id: str,
    filename: str,
    section_title: str,
    section_elements: List[Dict[str, Any]],
) -> None:
    if not section_elements:
        return

    texts = []
    element_types = []
    page_numbers = []

    for el in section_elements:
        text = str(el.get("text", "")).strip()
        if text:
            texts.append(text)

        element_types.append(str(el.get("type", "Unknown")))

        page_number = el.get("page_number")
        if isinstance(page_number, int):
            page_numbers.append(page_number)

    combined_text = "\n".join(texts).strip()
    combined_text = clean_text(combined_text)

    if not combined_text:
        return

    if combined_text.strip() == (section_title or "").strip():
        return

    semantic_chunks = make_semantic_chunks(
        manual_id=manual_id,
        filename=filename,
        section_title=section_title,
        combined_text=combined_text,
        element_types=element_types,
        page_numbers=page_numbers,
    )

    chunks.extend(semantic_chunks)


def chunk_parsed_document(parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    manual_id = parsed_doc["manual_id"]
    filename = parsed_doc["filename"]
    elements = parsed_doc["elements"]

    chunks: List[Dict[str, Any]] = []

    current_section_title = "Document Start"
    current_section_elements: List[Dict[str, Any]] = []

    for el in elements:
        el_type = str(el.get("type", "")).strip()
        text = str(el.get("text", "")).strip()

        if not is_meaningful_text(text):
            continue

        is_table_like = (
            "table" in el_type.lower()
            or bool(el.get("metadata", {}).get("text_as_html"))
            or looks_like_table_text(text)
        )

        if is_table_like:
            finalize_section_chunk(
                chunks,
                manual_id,
                filename,
                current_section_title,
                current_section_elements,
            )
            current_section_elements = []

            page_number = el.get("page_number")
            cleaned_table_text = clean_text(text)

            if len(cleaned_table_text) < 20:
                continue

            table_id = str(uuid.uuid4())
            chunks.append(
                {
                    "chunk_id": table_id,
                    "chunk_type": "parent",
                    "manual_id": manual_id,
                    "section_title": current_section_title,
                    "content_type": "table",
                    "page_start": page_number,
                    "page_end": page_number,
                    "text": cleaned_table_text,
                    "element_types": [el_type],
                    "metadata": {
                        "source_file": filename,
                        "parent_section": current_section_title,
                        "text_as_html": el.get("metadata", {}).get("text_as_html"),
                    },
                }
            )
            # Table child — indexes the same text so tables are retrievable via vector search
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_type": "child",
                    "parent_id": table_id,
                    "manual_id": manual_id,
                    "section_title": current_section_title,
                    "content_type": "table",
                    "page_start": page_number,
                    "page_end": page_number,
                    "text": cleaned_table_text,
                    "element_types": [el_type],
                    "metadata": {
                        "source_file": filename,
                        "parent_section": current_section_title,
                    },
                }
            )
            continue

        if el_type.lower() == "title":
            finalize_section_chunk(
                chunks,
                manual_id,
                filename,
                current_section_title,
                current_section_elements,
            )
            current_section_title = text
            current_section_elements = [el]
            continue

        current_section_elements.append(el)

    finalize_section_chunk(
        chunks,
        manual_id,
        filename,
        current_section_title,
        current_section_elements,
    )

    return chunks
