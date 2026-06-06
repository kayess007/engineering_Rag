import json
import time
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

FEEDBACK_FILE = Path("storage/feedback.jsonl")

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import login, require_auth
from app.query_classifier import classify_query, collection_for_type
from app.chunker import chunk_parsed_document, load_parsed_json, save_chunked_json
from app.logging_config import setup_logging
from app.middleware import RequestLoggingMiddleware
from app.parser import parse_pdf_to_elements
from app.rag_chain import generate_rag_answer
from app.retriever import retrieve_chunks, retrieve_chunks_advanced
from app.schemas import ParsedDocument
from app.utils import save_parsed_json, save_upload_file, UPLOAD_DIR, PARSED_DIR
from app.vectordb import index_chunks, get_vectorstore

setup_logging()

CHUNKED_DIR = Path("storage/chunked")


app = FastAPI(title="Engineering RAG - Phase 10")
app.add_middleware(RequestLoggingMiddleware)

# ── Auth route ────────────────────────────────────────────────────────────────
app.post("/auth/token")(login)


class ChunkRequest(BaseModel):
    parsed_file_path: str


class IndexRequest(BaseModel):
    chunked_file_path: str
    collection_name: str | None = None   # if None, auto-derived from manual_type
    manual_type: str | None = None       # "parts" | "maintenance" — auto-detected from filename if omitted


class QueryRequest(BaseModel):
    question: str
    collection_name: str | None = None  # auto-routed by query classifier if omitted
    k: int = 6
    filters: dict | None = None


class AskRequest(BaseModel):
    question: str
    collection_name: str | None = None  # auto-routed by query classifier if omitted
    k: int = 6
    model: str = "gpt-4.1-mini"
    response_mode: str = "standard"
    include_contexts: bool = False
    filters: dict | None = None


@app.get("/")
def root():
    return {"message": "Engineering RAG API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/manuals/list")
def list_manuals():
    manuals = []
    for json_path in sorted(PARSED_DIR.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            manuals.append(
                {
                    "manual_id": data.get("manual_id"),
                    "filename": data.get("filename"),
                    "equipment_model": data.get("equipment_model"),
                    "element_count": data.get("element_count"),
                }
            )
        except Exception:
            pass
    return {"manuals": manuals, "count": len(manuals)}


@app.delete("/manuals/{manual_id}")
def delete_manual(manual_id: str, collection_name: str = "engineering_manuals", _user: str = Depends(require_auth)):
    try:
        deleted = {}

        # Remove from ChromaDB
        try:
            vectorstore = get_vectorstore(collection_name=collection_name)
            vectorstore._collection.delete(where={"manual_id": manual_id})
            deleted["vectordb"] = True
        except Exception as e:
            deleted["vectordb"] = f"skipped: {e}"

        # Remove parsed JSON
        parsed_path = PARSED_DIR / f"{manual_id}.json"
        if parsed_path.exists():
            parsed_path.unlink()
            deleted["parsed_json"] = True

        # Remove chunked JSON
        chunked_path = CHUNKED_DIR / f"{manual_id}_chunks.json"
        if chunked_path.exists():
            chunked_path.unlink()
            deleted["chunked_json"] = True

        # Remove uploaded PDF (find by manual_id in parsed metadata)
        for upload in UPLOAD_DIR.glob("*.pdf"):
            deleted["uploaded_pdf"] = "not found"
        # best effort — find the filename from the parsed file (already deleted above)
        # so we rely on the caller knowing the filename or skip pdf deletion
        deleted["note"] = "PDF in uploads/ not deleted — remove manually if needed"

        return {"manual_id": manual_id, "deleted": deleted}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


_MODEL_HINTS = {
    "sebu9105": "C13",
    "c13": "C13",
    "maintenance": "DI450-DI560",
    "di450": "DI450",
    "di560": "DI560",
    "sebp6451": "C13-Parts",
}

# Keywords in filenames/models that signal a parts manual
_PARTS_FILENAME_SIGNALS = {"parts", "sebp", "part-manual", "parts-manual", "catalogue", "catalog"}
_PARTS_MODEL_SIGNALS = {"parts", "part"}

# Map manual_type → ChromaDB collection name
COLLECTION_MAP = {
    "parts": "parts_manuals",
    "maintenance": "maintenance_manuals",
}


def _detect_equipment_model(filename: str, hint: str | None = None) -> str | None:
    if hint:
        return hint.strip()
    lower = filename.lower()
    for key, model in _MODEL_HINTS.items():
        if key in lower:
            return model
    return None


def _detect_manual_type(filename: str, equipment_model: str | None = None) -> str:
    """Return 'parts' or 'maintenance' based on filename and equipment_model."""
    lower = filename.lower()
    if any(sig in lower for sig in _PARTS_FILENAME_SIGNALS):
        return "parts"
    if equipment_model:
        model_lower = equipment_model.lower()
        if any(sig in model_lower for sig in _PARTS_MODEL_SIGNALS):
            return "parts"
    return "maintenance"


@app.post("/manuals/upload")
async def upload_manual(
    file: UploadFile = File(...),
    equipment_model: str | None = None,
    _user: str = Depends(require_auth),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported in this phase.",
        )

    # Duplicate upload guard
    destination = UPLOAD_DIR / file.filename
    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail=f"File '{file.filename}' has already been uploaded.",
        )

    manual_id = str(uuid.uuid4())

    try:
        start_time = time.time()
        print(f"Starting upload for: {file.filename}")

        saved_path = save_upload_file(file)
        print(f"Saved file to: {saved_path}")

        elements = parse_pdf_to_elements(str(saved_path))
        print(f"Parsed {len(elements)} elements")

        detected_model = _detect_equipment_model(file.filename, equipment_model)
        detected_type = _detect_manual_type(file.filename, detected_model)

        payload = ParsedDocument(
            manual_id=manual_id,
            filename=file.filename,
            content_type=file.content_type or "application/pdf",
            element_count=len(elements),
            elements=elements,
            equipment_model=detected_model,
        ).model_dump()

        json_path = save_parsed_json(manual_id, payload)
        print(f"Saved parsed JSON to: {json_path}")
        print(f"Total request time: {time.time() - start_time:.2f}s")

        return JSONResponse(
            content={
                "manual_id": manual_id,
                "filename": file.filename,
                "equipment_model": detected_model,
                "manual_type": detected_type,
                "saved_pdf": str(saved_path),
                "saved_json": str(json_path),
                "element_count": len(elements),
                "sample_types": sorted({e["type"] for e in elements})[:20],
            }
        )

    except Exception as e:
        print("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@app.post("/manuals/chunk")
def chunk_manual(req: ChunkRequest, _user: str = Depends(require_auth)):
    try:
        print(f"Loading parsed file: {req.parsed_file_path}")
        parsed_doc = load_parsed_json(req.parsed_file_path)
        manual_id = parsed_doc["manual_id"]

        chunks = chunk_parsed_document(parsed_doc)
        print(f"Created {len(chunks)} chunks")

        chunked_path = save_chunked_json(manual_id, chunks)
        print(f"Saved chunked JSON to: {chunked_path}")

        return {
            "manual_id": manual_id,
            "parsed_file": req.parsed_file_path,
            "chunked_file": str(chunked_path),
            "chunk_count": len(chunks),
            "sample_chunk_types": sorted(list({c["content_type"] for c in chunks})),
        }

    except Exception as e:
        print("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


@app.post("/manuals/index")
def index_manual(req: IndexRequest, _user: str = Depends(require_auth)):
    try:
        print(f"Indexing chunked file: {req.chunked_file_path}")

        # Resolve collection: explicit > manual_type mapping > filename detection > legacy default
        if req.collection_name:
            collection = req.collection_name
            manual_type = req.manual_type or ("parts" if collection == "parts_manuals" else "maintenance")
        else:
            filename = Path(req.chunked_file_path).stem  # e.g. "sebp6451_chunks"
            manual_type = req.manual_type or _detect_manual_type(filename)
            collection = COLLECTION_MAP.get(manual_type, "maintenance_manuals")

        doc_count = index_chunks(req.chunked_file_path, collection_name=collection)
        print(f"Indexed {doc_count} documents into collection: {collection} (type={manual_type})")

        return {
            "chunked_file": req.chunked_file_path,
            "collection_name": collection,
            "manual_type": manual_type,
            "indexed_count": doc_count,
        }

    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        print("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/query")
def query_manuals(req: QueryRequest):
    try:
        print(f"Query received: {req.question}")

        query_type = classify_query(req.question)
        collections = [req.collection_name] if req.collection_name else collection_for_type(query_type)

        results = retrieve_chunks(
            query=req.question,
            collection_name=req.collection_name,
            k=req.k,
            filters=req.filters,
        )

        return {
            "question": req.question,
            "query_type": query_type,
            "collections_searched": collections,
            "result_count": len(results),
            "results": results,
        }

    except Exception as e:
        print("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


class AskAdvancedRequest(BaseModel):
    question: str
    collection_name: str | None = None  # auto-routed by query classifier if omitted
    k: int = 6
    model: str = "gpt-4.1-mini"
    rewrite_model: str = "gpt-4.1-mini"
    response_mode: str = "standard"
    include_contexts: bool = False
    filters: dict | None = None


@app.post("/ask/advanced")
def ask_manuals_advanced(req: AskAdvancedRequest):
    """Phase 9: query rewriting + cross-encoder reranking before answer generation."""
    try:
        print(f"Advanced ask received: {req.question}")

        query_type = classify_query(req.question)
        collections = [req.collection_name] if req.collection_name else collection_for_type(query_type)

        retrieved = retrieve_chunks_advanced(
            query=req.question,
            collection_name=req.collection_name,
            k=req.k,
            rewrite_model=req.rewrite_model,
            filters=req.filters,
        )

        rag_result = generate_rag_answer(
            question=req.question,
            retrieved_chunks=retrieved,
            model=req.model,
            max_chunks=min(5, len(retrieved)),
            response_mode=req.response_mode,
        )

        response = {
            "question": req.question,
            "query_type": query_type,
            "collections_searched": collections,
            "result_count": len(retrieved),
            "answer": rag_result["answer"],
            "sources": [
                {
                    "source_file": r["metadata"].get("source_file"),
                    "section_title": r["metadata"].get("section_title"),
                    "page_start": r["metadata"].get("page_start"),
                    "page_end": r["metadata"].get("page_end"),
                }
                for r in rag_result["used_chunks"]
            ],
        }
        if req.include_contexts:
            # Return all retrieved chunks for eval (recall/precision need full k=8 pool).
            # "used_chunks" (top-5) are what the LLM saw; "contexts" covers the full
            # retrieval so RAGAS context_recall is measured against the whole retrieved set.
            response["contexts"] = retrieved
        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Advanced ask failed: {str(e)}")


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str          # "positive" | "negative"
    comment: str = ""
    sources: list = []


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    if req.rating not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="rating must be 'positive' or 'negative'")
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": req.question,
        "answer": req.answer,
        "rating": req.rating,
        "comment": req.comment,
        "sources": req.sources,
    }
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "saved", "id": entry["id"]}


@app.get("/feedback")
def get_feedback():
    if not FEEDBACK_FILE.exists():
        return {"feedback": [], "count": 0}
    entries = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    positive = sum(1 for e in entries if e["rating"] == "positive")
    negative = sum(1 for e in entries if e["rating"] == "negative")
    return {
        "count": len(entries),
        "positive": positive,
        "negative": negative,
        "feedback": entries,
    }


@app.post("/ask")
def ask_manuals(req: AskRequest):
    try:
        print(f"Ask request received: {req.question}")

        query_type = classify_query(req.question)
        collections = [req.collection_name] if req.collection_name else collection_for_type(query_type)

        retrieved = retrieve_chunks(
            query=req.question,
            collection_name=req.collection_name,
            k=req.k,
            filters=req.filters,
        )

        rag_result = generate_rag_answer(
            question=req.question,
            retrieved_chunks=retrieved,
            model=req.model,
            max_chunks=min(5, len(retrieved)),
            response_mode=req.response_mode,
        )

        response = {
            "question": req.question,
            "query_type": query_type,
            "collections_searched": collections,
            "result_count": len(retrieved),
            "answer": rag_result["answer"],
            "sources": [
                {
                    "source_file": r["metadata"].get("source_file"),
                    "section_title": r["metadata"].get("section_title"),
                    "page_start": r["metadata"].get("page_start"),
                    "page_end": r["metadata"].get("page_end"),
                }
                for r in rag_result["used_chunks"]
            ],
        }
        if req.include_contexts:
            response["contexts"] = rag_result["used_chunks"]
        return response

    except Exception as e:
        print("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")
