from typing import Any, Dict, List, Tuple  # noqa: F401
import uuid

import fitz  # PyMuPDF
import pdfplumber


def safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    return str(value)


def _build_heading_thresholds(doc: fitz.Document) -> Tuple[float, float, float, float]:
    """
    Analyse font sizes across the whole document and return
    (h1_min, h2_min, h3_min, body_size) thresholds.
    """
    size_counts: Dict[float, int] = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    s = round(span["size"], 1)
                    size_counts[s] = size_counts.get(s, 0) + 1

    if not size_counts:
        return (20.0, 13.0, 11.5, 11.0)

    # Body size = most frequent size
    body_size = max(size_counts, key=lambda s: size_counts[s])

    # Heading candidates: sizes larger than body with at least 5 occurrences
    candidates = sorted(
        [s for s, c in size_counts.items() if s > body_size * 1.1 and c >= 5],
        reverse=True,
    )

    h1_min = candidates[0] if len(candidates) >= 1 else body_size * 2.0
    h2_min = candidates[1] if len(candidates) >= 2 else body_size * 1.3
    h3_min = body_size * 1.05  # bold at body size or slightly larger

    return (h1_min, h2_min, h3_min, body_size)


def _classify_span(
    span: Dict,
    h1_min: float,
    h2_min: float,
    h3_min: float,
    body_size: float,
) -> str:
    size = round(span["size"], 1)
    font = span.get("font", "")
    is_bold = "bold" in font.lower()

    if size >= h1_min:
        return "Title"
    if size >= h2_min:
        return "Title"
    if is_bold and size >= h3_min and size > body_size:
        return "Title"
    return "NarrativeText"


def _extract_tables_pdfplumber(file_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Return a dict of page_number -> list of table elements extracted by pdfplumber.
    Page numbers are 1-based.
    """
    table_elements: Dict[int, List[Dict[str, Any]]] = {}

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue

            page_tables = []
            for table in tables:
                if not table or len(table) < 2:
                    continue

                rows = []
                html_rows = []
                for row in table:
                    cells = [str(c).strip() if c is not None else "" for c in row]
                    rows.append("\t".join(cells))
                    html_rows.append(
                        "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                    )

                text = "\n".join(rows)
                text_as_html = "<table>" + "".join(html_rows) + "</table>"

                if len(text.strip()) < 10:
                    continue

                page_tables.append(
                    {
                        "element_id": str(uuid.uuid4()),
                        "type": "Table",
                        "text": text,
                        "page_number": page_num,
                        "category_depth": None,
                        "parent_id": None,
                        "metadata": {
                            "page_number": page_num,
                            "text_as_html": text_as_html,
                            "source": "pdfplumber",
                        },
                        "_y": 9999,  # place tables after text on the same page
                    }
                )

            if page_tables:
                table_elements[page_num] = page_tables

    return table_elements


def parse_with_pymupdf(file_path: str) -> List[Dict[str, Any]]:
    print(f"Parsing with PyMuPDF: {file_path}")

    # Pass 1: extract tables with pdfplumber (closed before fitz opens the file)
    table_elements = _extract_tables_pdfplumber(file_path)
    print(f"  pdfplumber tables extracted across {len(table_elements)} pages")

    # Pass 2: extract text and headings with fitz
    doc = fitz.open(file_path)
    h1_min, h2_min, h3_min, body_size = _build_heading_thresholds(doc)
    print(f"  Heading thresholds — H1>={h1_min}pt, H2>={h2_min}pt, H3>{body_size}pt bold, body={body_size}pt")

    normalized: List[Dict[str, Any]] = []

    for page_num, page in enumerate(doc, start=1):
        page_elements: List[Dict[str, Any]] = []

        # Text elements from PyMuPDF
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue

            block_y = block["bbox"][1]

            for line in block["lines"]:
                line_text_parts = []
                line_type = "NarrativeText"
                line_depth = None

                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    span_type = _classify_span(span, h1_min, h2_min, h3_min, body_size)
                    line_text_parts.append(text)

                    size = round(span["size"], 1)
                    if span_type == "Title":
                        line_type = "Title"
                        if size >= h1_min:
                            line_depth = 1
                        elif size >= h2_min:
                            line_depth = 2
                        else:
                            line_depth = 3

                line_text = " ".join(line_text_parts).strip()
                if not line_text:
                    continue

                page_elements.append(
                    {
                        "element_id": str(uuid.uuid4()),
                        "type": line_type,
                        "text": line_text,
                        "page_number": page_num,
                        "category_depth": line_depth,
                        "parent_id": None,
                        "metadata": {
                            "page_number": page_num,
                            "source": "pymupdf",
                        },
                        "_y": block_y,
                    }
                )

        # Table elements from pdfplumber for this page
        for tbl in table_elements.get(page_num, []):
            page_elements.append(tbl)

        # Sort by y position
        page_elements.sort(key=lambda e: e.get("_y", 0))

        # Strip internal _y key
        for el in page_elements:
            el.pop("_y", None)
            normalized.append(el)

    doc.close()
    print(f"  PyMuPDF parsing complete. Elements: {len(normalized)}")
    return normalized



def parse_with_pypdf(file_path: str) -> List[Dict[str, Any]]:
    print(f"Parsing with pypdf fallback: {file_path}")

    from pypdf import PdfReader
    reader = PdfReader(file_path)
    normalized: List[Dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            normalized.append(
                {
                    "element_id": str(uuid.uuid4()),
                    "type": "NarrativeText",
                    "text": text,
                    "page_number": page_number,
                    "category_depth": None,
                    "parent_id": None,
                    "metadata": {
                        "page_number": page_number,
                        "source": "pypdf",
                    },
                }
            )

    print(f"pypdf parsing complete. Final elements: {len(normalized)}")
    return normalized


def parse_pdf_to_elements(file_path: str) -> List[Dict[str, Any]]:
    try:
        return parse_with_pymupdf(file_path)
    except Exception as e:
        print(f"PyMuPDF failed. Falling back to pypdf. Reason: {e}")

    return parse_with_pypdf(file_path)
