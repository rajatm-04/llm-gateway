"""Scoped semantic cache orchestration; embedding work stays off the event loop."""

import asyncio
from typing import Any

from gateway.cache.embedder import Embedder
from gateway.cache.scope import CacheKey
from gateway.cache.vector_store import VectorStore
from gateway.config import Settings, settings


class SemanticCache:
    def __init__(self, embedder: Embedder | None = None, vector_store: VectorStore | None = None,
                 config: Settings | None = None):
        self.config = config if config is not None else settings
        self.embedder = embedder if embedder is not None else Embedder(self.config.embedding_model)
        self.vector_store = vector_store if vector_store is not None else VectorStore(self.config)
        self._embedding_lock = asyncio.Lock()

    async def initialize(self) -> None:
        await self.vector_store.ensure_collection()

    async def _embed(self, text: str) -> list[float]:
        # Serialize model use/lazy loading within this cache instance.
        async with self._embedding_lock:
            return await asyncio.to_thread(self.embedder.embed, text)

    async def lookup(self, key: CacheKey) -> dict[str, Any] | None:
        if key.exact:
            return await self.vector_store.find_exact(key.filters)
        vector = await self._embed(key.query)
        return await self.vector_store.search(
            vector=vector,
            threshold=self.config.cache_similarity_threshold,
            filters=key.filters,
        )

    async def store(self, key: CacheKey, response: str, metadata: dict[str, Any] | None = None) -> None:
        # Exact entries don't need embeddings, but the collection requires a vector.
        vector = ([0.0] * self.vector_store.vector_size if key.exact else await self._embed(key.query))
        payload = {
            **(metadata or {}),
            "query": key.query,
            "response": response,
            **key.fields,  # Caller metadata must never override cache boundaries.
        }
        await self.vector_store.upsert(vector=vector, payload=payload)

    async def close(self) -> None:
        await self.vector_store.close()
