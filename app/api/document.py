from fastapi import APIRouter, UploadFile, File, status
from fastapi.exceptions import HTTPException
from pathlib import Path
from uuid import UUID, uuid4
from pydantic import BaseModel
from typing import Tuple
import asyncio
import logging

from app.core import load_documents, chunk_documents, generate_embedding
from app.core.qdrant import Collection
from app.core.chunk import ChunkingStrategy
from app.db.models import DocumentRecord
from app.db import AsyncSessionLocal
from app.core.config import settings

router = APIRouter()
collection = Collection()
UPLOAD_DIR = settings.upload_dir
CHUNK_SIZE = settings.upload_chunk_size
logger = logging.getLogger(__name__)


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
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = "recursive",
):
    # Verify and download the file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    filename = Path(file.filename).name

    if not filename.lower().endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt or .pdf is supported")

    document_id = uuid4()
    document_dir = UPLOAD_DIR / str(document_id)
    document_dir.mkdir(parents=True, exist_ok=True)

    file_path = document_dir / filename

    try:
        # Dump file
        # TODO: Reuse saved file on failure. as Further enhancement
        with open(file_path, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                f.write(chunk)

        # Update metadata to the DB
        # status = processing
        async with AsyncSessionLocal() as session:
            record = DocumentRecord(
                id=document_id,
                filename=filename,
                file_type=Path(file_path).suffix.lower(),
                file_path=str(file_path),
                chunking_strategy=chunking_strategy,
                status="processing",
            )
            session.add(record)
            await session.commit()

        ## RAG pipeline
        try:
            (chunk_texts_len, top_embedding) = await asyncio.to_thread(
                rag_pipeline, file_path, chunking_strategy, document_id, filename
            )
        except ValueError as e:
            await _mark_failed(document_id)
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:
            await _mark_failed(document_id)
            logger.exception("RAG pipeline failed for document %s", document_id)
            raise HTTPException(
                status_code=500, detail="Pipeline failed during processing"
            )

        # Update back onto DB
        # status = Processed
        async with AsyncSessionLocal() as session:
            record = await session.get(DocumentRecord, ident=document_id)

            if record is None:
                raise RuntimeError("Document metadata record was not found")

            record.chunk_count = chunk_texts_len
            record.status = "indexed"

            await session.commit()

        return {
            "status": "success",
            "payload": {
                "document_id": str(document_id),
                "filename": filename,
                "chunking_strategy": chunking_strategy,
                "chunk_count": chunk_texts_len,
                "embedding_dimension": len(top_embedding),
                "status": "indexed",
            },
        }
    finally:
        await file.close()


def rag_pipeline(
    file_path: Path,
    chunking_strategy: ChunkingStrategy,
    document_id: UUID,
    filename: str,
) -> Tuple[int, list[float]]:
    # Load the file
    docs = load_documents(file_path)

    # 1. Chunking
    chunks = chunk_documents(docs, strategy=chunking_strategy)
    chunk_texts = [chunk.page_content for chunk in chunks]
    if not chunk_texts:
        raise ValueError("No text could be extracted from the document")

    # 2. Vector Embeddings
    embeddings = generate_embedding(chunk_texts)

    # 3. Store on qdrant
    collection.add_vectors(str(document_id), chunk_texts, embeddings, filename)

    return (len(chunk_texts), embeddings[0])


async def _mark_failed(document_id: UUID):
    try:
        async with AsyncSessionLocal() as session:
            record = await session.get(DocumentRecord, document_id)
            if record:
                record.status = "failed"
                await session.commit()
    except Exception:
        logger.exception("Could not mark document %s as failed", document_id)
