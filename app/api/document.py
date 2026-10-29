from fastapi import APIRouter, UploadFile, File, Form
from fastapi.exceptions import HTTPException
from pathlib import Path
from uuid import uuid4

from app.core import load_documents, chunk_documents, generate_embedding
from app.core.qdrant import Collection

router = APIRouter()
UPLOAD_DIR = Path("./dump")
CHUNK_SIZE = 2048


"""
Handles ingestion. 

Usually, uploading, and ingestion part should be handled on different api endpoints.
"""


@router.post("/document")
async def document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("recursive"),
):
    # Check and download the file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    filename = Path(file.filename).name

    if not filename.lower().endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt or .pdf is supported")

    if chunking_strategy not in {"fixed", "recursive"}:
        raise HTTPException(
            status_code=422,
            detail="chunking_strategy must be 'fixed' or 'recursive'",
        )

    document_id = uuid4()
    document_dir = UPLOAD_DIR / str(document_id)
    document_dir.mkdir(parents=True, exist_ok=True)

    file_path = document_dir / filename

    with open(file_path, "wb") as f:
        while chunk := await file.read(CHUNK_SIZE):
            f.write(chunk)

    # Load the file
    docs = load_documents(file_path)

    # chunking
    chunks = chunk_documents(docs)

    chunk_texts = [
        chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        for chunk in chunks
    ]

    if not chunk_texts:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the document",
        )

    # embeddings
    embeddings = generate_embedding(chunk_texts)

    # store on qdrant
    collection = Collection()
    collection.add_vectors(str(document_id), chunk_texts, embeddings, filename)
    return {"Hello": "World"}
