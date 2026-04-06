"""
Thin HTTP client for the Engineering RAG FastAPI backend.
All functions raise requests.HTTPError on non-2xx responses.
"""

import os
import requests

BASE_URL = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")
TIMEOUT = 60


def _auth(token: str | None) -> dict:
    """Return Authorization header dict if token provided."""
    return {"Authorization": f"Bearer {token}"} if token else {}


def health() -> dict:
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def login(username: str, password: str) -> str:
    """Returns access_token string."""
    r = requests.post(
        f"{BASE_URL}/auth/token",
        data={"username": username, "password": password},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def list_manuals() -> list[dict]:
    r = requests.get(f"{BASE_URL}/manuals/list", timeout=10)
    r.raise_for_status()
    return r.json().get("manuals", [])


def upload_manual(file_bytes: bytes, filename: str, token: str) -> dict:
    r = requests.post(
        f"{BASE_URL}/manuals/upload",
        files={"file": (filename, file_bytes, "application/pdf")},
        headers=_auth(token),
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def chunk_manual(parsed_file_path: str, token: str) -> dict:
    r = requests.post(
        f"{BASE_URL}/manuals/chunk",
        json={"parsed_file_path": parsed_file_path},
        headers=_auth(token),
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def index_manual(chunked_file_path: str, token: str, collection_name: str = "engineering_manuals") -> dict:
    r = requests.post(
        f"{BASE_URL}/manuals/index",
        json={"chunked_file_path": chunked_file_path, "collection_name": collection_name},
        headers=_auth(token),
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def delete_manual(manual_id: str, token: str) -> dict:
    r = requests.delete(
        f"{BASE_URL}/manuals/{manual_id}",
        headers=_auth(token),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def ask(question: str, k: int = 8, model: str = "gpt-4.1-mini") -> dict:
    r = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question, "k": k, "model": model},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def ask_advanced(question: str, k: int = 8, model: str = "gpt-4.1-mini") -> dict:
    r = requests.post(
        f"{BASE_URL}/ask/advanced",
        json={"question": question, "k": k, "model": model},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def query(question: str, k: int = 8) -> list[dict]:
    r = requests.post(
        f"{BASE_URL}/query",
        json={"question": question, "k": k},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("results", [])
