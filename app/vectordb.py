import json
from pathlib import Path
from typing import Any, Dict, List

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from app.embeddings import get_embedding_model

CHROMA_DIR = Path("storage/chroma")


def load_chunked_json(chunked_file_path: str) -> List[Dict[str, Any]]:
    with open(chunked_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_to_document(chunk: Dict[str, Any]) -> Document:
    metadata = {
        "chunk_id": chunk.get("chunk_id"),
        "manual_id": chunk.get("manual_id"),
        "section_title": chunk.get("section_title"),
        "content_type": chunk.get("content_type"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "source_file": chunk.get("metadata", {}).get("source_file"),
        "parent_section": chunk.get("metadata", {}).get("parent_section"),
        "table_title": chunk.get("metadata", {}).get("table_title"),
    }
    return Document(
        page_content=chunk.get("text", ""),
        metadata=metadata,
    )


def index_chunks(
    chunked_file_path: str,
    collection_name: str = "engineering_manuals",
) -> int:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading chunks from: {chunked_file_path}")

    chunks = load_chunked_json(chunked_file_path)
    documents = [
        chunk_to_document(chunk)
        for chunk in chunks
        if chunk.get("text", "").strip()
    ]

    if not documents:
        return 0

    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )

    # Duplicate guard: reject if this manual_id is already indexed
    manual_id = documents[0].metadata.get("manual_id")
    if manual_id:
        existing = vectorstore.get(where={"manual_id": manual_id}, limit=1)
        if existing and existing.get("ids"):
            raise ValueError(
                f"Manual '{manual_id}' is already indexed in collection '{collection_name}'."
            )

    print(f"Prepared {len(documents)} documents for embedding")

    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Indexed {min(i + batch_size, len(documents))} / {len(documents)}")

    print("Indexing complete")
    return len(documents)


def get_vectorstore(collection_name: str = "engineering_manuals") -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embedding_model = get_embedding_model()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )
