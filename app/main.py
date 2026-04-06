import json
import time
import uuid
import traceback
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import login, require_auth
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
    collection_name: str = "engineering_manuals"


class QueryRequest(BaseModel):
    question: str
    collection_name: str = "engineering_manuals"
    k: int = 5
    filters: dict | None = None


class AskRequest(BaseModel):
    question: str
    collection_name: str = "engineering_manuals"
    k: int = 5
    model: str = "gpt-4.1-mini"
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


@app.post("/manuals/upload")
async def upload_manual(file: UploadFile = File(...), _user: str = Depends(require_auth)):
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

        payload = ParsedDocument(
            manual_id=manual_id,
            filename=file.filename,
            content_type=file.content_type or "application/pdf",
            element_count=len(elements),
            elements=elements,
        ).model_dump()

        json_path = save_parsed_json(manual_id, payload)
        print(f"Saved parsed JSON to: {json_path}")
        print(f"Total request time: {time.time() - start_time:.2f}s")

        return JSONResponse(
            content={
                "manual_id": manual_id,
                "filename": file.filename,
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
        doc_count = index_chunks(req.chunked_file_path, collection_name=req.collection_name)
        print(f"Indexed {doc_count} documents into collection: {req.collection_name}")

        return {
            "chunked_file": req.chunked_file_path,
            "collection_name": req.collection_name,
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

        results = retrieve_chunks(
            query=req.question,
            collection_name=req.collection_name,
            k=req.k,
            filters=req.filters,
        )

        return {
            "question": req.question,
            "collection_name": req.collection_name,
            "result_count": len(results),
            "results": results,
        }

    except Exception as e:
        print("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


class AskAdvancedRequest(BaseModel):
    question: str
    collection_name: str = "engineering_manuals"
    k: int = 5
    model: str = "gpt-4.1-mini"
    rewrite_model: str = "gpt-4.1-mini"
    filters: dict | None = None


@app.post("/ask/advanced")
def ask_manuals_advanced(req: AskAdvancedRequest):
    """Phase 9: query rewriting + cross-encoder reranking before answer generation."""
    try:
        print(f"Advanced ask received: {req.question}")

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
        )

        return {
            "question": req.question,
            "collection_name": req.collection_name,
            "result_count": len(retrieved),
            "answer": rag_result["answer"],
            "sources": [
                {
                    "source_file": r["metadata"].get("source_file"),
                    "section_title": r["metadata"].get("section_title"),
                    "page_start": r["metadata"].get("page_start"),
                    "page_end": r["metadata"].get("page_end"),
                }
                for r in retrieved
            ],
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Advanced ask failed: {str(e)}")


@app.post("/ask")
def ask_manuals(req: AskRequest):
    try:
        print(f"Ask request received: {req.question}")

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
        )

        return {
            "question": req.question,
            "collection_name": req.collection_name,
            "result_count": len(retrieved),
            "answer": rag_result["answer"],
            "sources": [
                {
                    "source_file": r["metadata"].get("source_file"),
                    "section_title": r["metadata"].get("section_title"),
                    "page_start": r["metadata"].get("page_start"),
                    "page_end": r["metadata"].get("page_end"),
                }
                for r in retrieved
            ],
        }

    except Exception as e:
        print("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")
