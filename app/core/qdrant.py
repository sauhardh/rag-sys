from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient, models
from uuid import UUID, uuid5


class Collection:
    VECTOR_SIZE = 384  # embedding models provide 384 dimensional vectors

    def __init__(
        self,
        collection_name: str = "documents",
        url: str = "http://localhost:6333",
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
    ):
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
