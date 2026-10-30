from fastapi import APIRouter, UploadFile, File, status
from fastapi.exceptions import HTTPException
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel

from app.core import load_documents, chunk_documents, generate_embedding
from app.core.qdrant import Collection
from app.db.models import DocumentRecord
from app.db import AsyncSessionLocal
from app.core.chunk import ChunkingStrategy

router = APIRouter()
collection = Collection()
UPLOAD_DIR = Path("./dump")
CHUNK_SIZE = 2048


"""
Handles ingestion. 

Usually, uploading, and ingestion part should be handled on different api endpoints.
"""


class DocumentPayload(BaseModel):
    document_id: str
    filename: str
    chunking_strategy: str
    chunk_count: int
    embedding_dimension: int
    status: str


class DocumentResponse(BaseModel):
    status: str
    payload: DocumentPayload


@router.post(
    "/document", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED
)
async def document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = "recursive",
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

    try:
        # dump file
        # TODO: Reuse saved file on failure. Further enhancement
        with open(file_path, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                f.write(chunk)

        # Update the DB
        # as processing
        async with AsyncSessionLocal() as session:
            record = DocumentRecord(
                id=document_id,
                filename=filename,
                file_type=Path(file_path).suffix.lower(),
                file_path=str(file_path),
                chunking_strategy=chunking_strategy,
                status="processing",
            )
            await session.commit()

        # Load the file
        docs = load_documents(file_path)

        # chunking
        chunks = chunk_documents(docs, strategy=chunking_strategy)

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
        collection.add_vectors(str(document_id), chunk_texts, embeddings, filename)

        # Update back on DB
        # Processed
        async with AsyncSessionLocal() as session:
            record = await session.get(DocumentRecord, document_id)

            if record is None:
                raise RuntimeError("Document metadata record was not found")

            record.chunk_count = len(chunk_texts)
            record.status = "indexed"

            await session.commit()

        return {
            "status": "success",
            "payload": {
                "document_id": str(document_id),
                "filename": filename,
                "chunking_strategy": chunking_strategy,
                "chunk_count": len(chunk_texts),
                "embedding_dimension": len(embeddings[0]),
                "status": "indexed",
            },
        }
    finally:
        await file.close()
