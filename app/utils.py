import json
import shutil
from pathlib import Path

from fastapi import UploadFile

UPLOAD_DIR = Path("storage/uploads")
PARSED_DIR = Path("storage/parsed")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)


def save_upload_file(upload_file: UploadFile) -> Path:
    destination = UPLOAD_DIR / upload_file.filename
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination


def save_parsed_json(manual_id: str, payload: dict) -> Path:
    output_path = PARSED_DIR / f"{manual_id}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path
