from typing import List, Tuple
from qdrant_client.models import (
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client import QdrantClient, models
from uuid import UUID, uuid5
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class Collection:
    VECTOR_SIZE = 384  # embedding models provide 384 dimensional vectors

    def __init__(
        self,
        collection_name: str = settings.qdrant_collection,
        url: str = settings.qdrant_url,
    ):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    def add_vectors(
        self,
        document_id: str,
        chunks: list[str],
        vectors: list[list[float]],
        filename: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("The number of chunks must match the number of vectors")

        if not chunks:
            return

        document_uuid = UUID(document_id)

        points = [
            models.PointStruct(
                id=str(uuid5(document_uuid, f"chunk-{index}")),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]

        self.client.upsert(
            collection_name=self.collection_name, points=points, wait=True
        )

    def query(
        self,
        vector_query: list[float],
        with_payload: bool = False,
        limit: int = 5,
        filter: Filter | None = None,
    ) -> List[ScoredPoint]:
        return self.client.query_points(
            collection_name=self.collection_name,
            query=vector_query,
            with_payload=with_payload,
            query_filter=filter,
            limit=limit,
        ).points

    def query_formatted(
        self,
        vector_query: list[float],
        document_id: str | None = None,
        with_payload: bool = True,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> Tuple[set[str], str]:
        query_filter: Filter | None = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        scored_points = self.query(
            vector_query=vector_query,
            with_payload=with_payload,
            limit=limit,
            filter=query_filter,
        )

        context_chunks = []
        sources = set()

        for point in scored_points:
            if point.score is not None and point.score < min_score:
                continue
            if point.payload:
                text = point.payload.get("text")
                filename = point.payload.get("filename")

                if text:
                    context_chunks.append(text)
                if filename:
                    sources.add(filename)

        context_str = "\n\n---\n\n".join(context_chunks)

        return (sources, context_str)
