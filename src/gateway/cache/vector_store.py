"""Qdrant vector-store abstraction."""

from typing import Any
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from gateway.config import settings


class VectorStore:
    """Stores and searches semantic-cache vectors in Qdrant."""

    def __init__(self) -> None:
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection_name = "queries"
        self.vector_size = 384  # all-MiniLM-L6-v2

    async def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = await self.client.get_collections()

        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def search(
        self,
        vector: list[float],
        threshold: float,
    ) -> dict[str, Any] | None:
        """Return the payload of the closest match above the threshold."""
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=1,
            score_threshold=threshold,
            with_payload=True,
        )

        if not results.points:
            return None

        return results.points[0].payload

    async def upsert(
        self,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert a vector and its payload into Qdrant."""
        point = PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload=payload,
        )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self.client.close()