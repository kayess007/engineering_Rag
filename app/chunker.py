import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Caterpillar parts table row: [item_ref] [optional graphic_ref] [part_number] [qty] [name...]
_PARTS_ROW_RE = re.compile(
    r"^\s*(\d+)\s+"           # item ref number
    r"(?:\d+\s+)?"            # optional graphic ref
    r"(\d{3}-\d{4})\s+"       # Caterpillar part number (e.g. 295-3099)
    r"(\d+)\s+"               # quantity
    r"(.+)$"                  # part name (rest of line)
)
# Column header lines to discard during normalization
_PARTS_HEADER_RE = re.compile(
    r"^(REF|GRAPHIC|PART\s+NAME|PART\s+NUMBER|NOTE|SEE|QTY|NO\.\s|PRODUCT\s*LEVEL|PAGE)\b",
    re.IGNORECASE,
)


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
    """Large chunks (1200 tokens) — sent to the LLM for full context.
    Overlap increased to 200 to keep component lists and structured
    content intact across chunk boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )


def get_child_splitter() -> RecursiveCharacterTextSplitter:
    """Small chunks (300 tokens) — indexed in ChromaDB for precise retrieval.
    Reduced from 400 to 300 so each embedding captures a tighter, more
    focused semantic unit. Overlap increased to 75 to avoid splitting
    key sentences at chunk boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=220,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )


def fix_doubled_chars(text: str) -> str:
    """Fix PDF font artifact where every character is duplicated.
    e.g. 'GGEENNEERRAALL' → 'GENERAL', 'CCaatt' → 'Cat'

    Strategy: if more than 60% of consecutive character pairs in a word
    are identical, collapse each doubled character to one.
    """
    import re

    def _fix_word(word: str) -> str:
        if len(word) < 4:
            return word
        # Count doubled pairs
        pairs = [word[i] == word[i + 1] for i in range(0, len(word) - 1, 2)]
        if len(pairs) == 0:
            return word
        if sum(pairs) / len(pairs) >= 0.6:
            # Collapse every two chars into one
            return "".join(word[i] for i in range(0, len(word), 2))
        return word

    # Apply word by word, preserving whitespace and punctuation
    return re.sub(r"[A-Za-z]{4,}", lambda m: _fix_word(m.group()), text)


def reconstruct_figure_legend(lines: List[str]) -> str | None:
    """Detect alternating number / label pattern from PDF figure legends and
    reconstruct them as readable prose.

    Pattern:
        5
        4
        3
        ...
        1
        Boom
        2
        Feed
        3
        Pipe handler
        ...

    Returns a single formatted string, or None if pattern not detected.
    """
    import re

    # Collect (number, label) pairs from interleaved lines
    pairs: List[tuple[int, str]] = []
    i = 0
    while i < len(lines) - 1:
        if re.fullmatch(r"\d+", lines[i]) and not re.fullmatch(r"\d+", lines[i + 1]):
            pairs.append((int(lines[i]), lines[i + 1]))
            i += 2
        else:
            i += 1

    # Only treat as a figure legend if we found at least 4 pairs
    if len(pairs) < 4:
        return None

    items = ", ".join(f"{num}={label}" for num, label in sorted(pairs))
    return f"Component locations: {items}"


def normalize_parts_table(text: str) -> str | None:
    """
    Detect Caterpillar parts-manual table rows and rewrite them as structured prose
    so the cross-encoder can match them against natural-language queries.

    Input (example):
        AIR INLET AND EXHAUST SYSTEM
        355-0875 COMBUSTION GP-EXHAUST
        REF GRAPHIC PART NAME SEE NOTE NO REF PART NUMBER QTY ...
        1 1 295-3099 1 SPARK PLUG
        2 1 342-1490 1 PLATE-INFORMATION (EXHAUST COMBUSTION)

    Output:
        AIR INLET AND EXHAUST SYSTEM
        355-0875 COMBUSTION GP-EXHAUST
        Item 1: Part 295-3099, SPARK PLUG, Qty: 1
        Item 2: Part 342-1490, PLATE-INFORMATION (EXHAUST COMBUSTION), Qty: 1
    """
    lines = text.splitlines()
    header_lines: List[str] = []
    structured: List[str] = []
    in_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        m = _PARTS_ROW_RE.match(stripped)
        if m:
            in_data = True
            item_no, part_num, qty, name = m.groups()
            # Strip trailing page-reference numbers (e.g. "SPARK PLUG 392")
            name = re.sub(r"\s+\d+\s*$", "", name.strip())
            structured.append(f"Item {item_no}: Part {part_num}, {name}, Qty: {qty}")
        elif not in_data:
            # Keep context lines before the data rows; drop column header rows
            if not _PARTS_HEADER_RE.match(stripped):
                header_lines.append(stripped)

    if not structured:
        return None

    return "\n".join(header_lines + structured)


def clean_text(text: str) -> str:
    # Fix doubled-character PDF artifact before any other processing
    text = fix_doubled_chars(text)

    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []

    for line in lines:
        if not line:
            continue
        if is_noisy_text(line):
            continue
        cleaned_lines.append(line)

    # Detect and reconstruct figure legends
    reconstructed = reconstruct_figure_legend(cleaned_lines)
    if reconstructed:
        # Preserve any header line (non-numeric, before the legend pairs)
        header = next(
            (l for l in cleaned_lines if not re.fullmatch(r"\d+", l) and l == cleaned_lines[0]),
            None,
        )
        if header and header.lower() not in reconstructed.lower():
            return f"{header}\n{reconstructed}"
        return reconstructed

    joined = "\n".join(cleaned_lines).strip()

    # Normalize Caterpillar parts table rows into structured prose
    normalized = normalize_parts_table(joined)
    if normalized:
        return normalized

    return joined


def make_semantic_chunks(
    manual_id: str,
    filename: str,
    section_title: str,
    combined_text: str,
    element_types: List[str],
    page_numbers: List[int],
    equipment_model: str | None = None,
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
                "equipment_model": equipment_model,
                "section_title": section_title or "Untitled Section",
                "content_type": "section",
                "page_start": page_start,
                "page_end": page_end,
                "text": cleaned_parent,
                "element_types": sorted(list(set(element_types))),
                "metadata": {
                    "source_file": filename,
                    "equipment_model": equipment_model,
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
                    "equipment_model": equipment_model,
                    "section_title": section_title or "Untitled Section",
                    "content_type": "section",
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": cleaned_child,
                    "element_types": sorted(list(set(element_types))),
                    "metadata": {
                        "source_file": filename,
                        "equipment_model": equipment_model,
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
    equipment_model: str | None = None,
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
        equipment_model=equipment_model,
    )

    chunks.extend(semantic_chunks)


def chunk_parsed_document(parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    manual_id = parsed_doc["manual_id"]
    filename = parsed_doc["filename"]
    elements = parsed_doc["elements"]
    equipment_model = parsed_doc.get("equipment_model")

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
                chunks, manual_id, filename,
                current_section_title, current_section_elements,
                equipment_model=equipment_model,
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
                    "equipment_model": equipment_model,
                    "section_title": current_section_title,
                    "content_type": "table",
                    "page_start": page_number,
                    "page_end": page_number,
                    "text": cleaned_table_text,
                    "element_types": [el_type],
                    "metadata": {
                        "source_file": filename,
                        "equipment_model": equipment_model,
                        "parent_section": current_section_title,
                        "text_as_html": el.get("metadata", {}).get("text_as_html"),
                    },
                }
            )
            # Tables > 1800 chars risk silent truncation at the 512-token embedding limit.
            # Split them with the child splitter so each child fits within token budget.
            TABLE_SPLIT_THRESHOLD = 1800
            if len(cleaned_table_text) > TABLE_SPLIT_THRESHOLD:
                table_child_texts = get_child_splitter().split_text(cleaned_table_text)
            else:
                table_child_texts = [cleaned_table_text]
            for idx, child_text in enumerate(table_child_texts, start=1):
                if len(child_text.strip()) < MIN_SECTION_CHARS:
                    continue
                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_type": "child",
                        "parent_id": table_id,
                        "manual_id": manual_id,
                        "equipment_model": equipment_model,
                        "section_title": current_section_title,
                        "content_type": "table",
                        "page_start": page_number,
                        "page_end": page_number,
                        "text": child_text,
                        "element_types": [el_type],
                        "metadata": {
                            "source_file": filename,
                            "equipment_model": equipment_model,
                            "parent_section": current_section_title,
                            "child_index": idx,
                        },
                    }
                )
            continue

        if el_type.lower() == "title":
            # Skip pure numeric or very short titles (e.g. "1", "2", "i05420578")
            # that come from parts manual reference numbers — not real section headings
            import re as _re
            is_noise_title = (
                _re.fullmatch(r"[\d\s\.\-]+", text)          # pure numbers/dots
                or _re.fullmatch(r"i\d{6,}", text)            # item codes like i05420578
                or len(text.strip()) <= 2                      # single char or digit
            )
            if is_noise_title:
                current_section_elements.append(el)
                continue

            finalize_section_chunk(
                chunks, manual_id, filename,
                current_section_title, current_section_elements,
                equipment_model=equipment_model,
            )
            current_section_title = text
            current_section_elements = [el]
            continue

        current_section_elements.append(el)

    finalize_section_chunk(
        chunks, manual_id, filename,
        current_section_title, current_section_elements,
        equipment_model=equipment_model,
    )

    return chunks
