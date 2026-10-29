from fastapi import APIRouter, UploadFile, File
from fastapi.exceptions import HTTPException
from pathlib import Path
from uuid import uuid4

from core import load_documents

router = APIRouter()
UPLOAD_DIR = Path("./dump")
CHUNK_SIZE = 2048


"""
Handles ingestion. 

Usually, uploading, and ingestion part should be handled on different api endpoints.
"""


@router.post("/document")
async def document(file: UploadFile = File(...)):
    # Check and download the file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    filename = file.filename

    if not filename.lower().endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt or .pdf is supported")

    document_id = uuid4()
    document_dir = UPLOAD_DIR / str(document_id)
    document_dir.mkdir(parents=True, exist_ok=True)

    file_path = document_dir / filename

    with open(file_path, "wb") as f:
        while chunk := await file.read(CHUNK_SIZE):
            f.write(chunk)

    # Load the file
    doc = load_documents(file_path)

    return {"Hello": "World"}
