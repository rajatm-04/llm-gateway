"""Semantic cache orchestration."""

from typing import Any

from gateway.cache.embedder import Embedder
from gateway.cache.vector_store import VectorStore
from gateway.config import settings


class SemanticCache:
    """Coordinates embedding, lookup, and storage of cached responses."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    async def initialize(self) -> None:
        """Ensure the Qdrant collection exists."""
        await self.vector_store.ensure_collection()

    async def lookup(self, query: str) -> dict[str, Any] | None:
        """
        Find a semantically similar cached query.

        Returns the cached payload on a hit, otherwise None.
        """
        query = query.strip()

        if not query:
            return None

        vector = self.embedder.embed(query)

        return await self.vector_store.search(
            vector=vector,
            threshold=settings.cache_similarity_threshold,
        )

    async def store(
        self,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Embed and store a response in Qdrant."""
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty")

        vector = self.embedder.embed(query)

        payload: dict[str, Any] = {
            "query": query,
            "response": response,
            **(metadata or {}),
        }

        await self.vector_store.upsert(
            vector=vector,
            payload=payload,
        )

    async def close(self) -> None:
        """Close the vector-store connection."""
        await self.vector_store.close()