"""Qdrant storage with mandatory exact scope filtering before similarity search."""

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from gateway.config import Settings, settings


def payload_filter(fields: dict[str, str]) -> Filter:
    if not fields.get("scope_id") or not fields.get("cache_schema"):
        raise ValueError("Cache queries require an exact scope and schema")
    return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in fields.items()])


class VectorStore:
    def __init__(self, config: Settings | None = None) -> None:
        config = config if config is not None else settings
        self.client = AsyncQdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        self.collection_name = "queries"
        self.vector_size = 384  # Must match the configured embedder; changing it requires migration.

    async def ensure_collection(self) -> None:
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    async def search(self, vector: list[float], threshold: float, filters: dict[str, str]) -> dict[str, Any] | None:
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=payload_filter(filters),
            limit=1,
            score_threshold=threshold,
            with_payload=True,
        )
        return results.points[0].payload if results.points else None

    async def find_exact(self, filters: dict[str, str]) -> dict[str, Any] | None:
        if not filters.get("request_id"):
            raise ValueError("Exact lookup requires request_id")
        points, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=payload_filter(filters),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return points[0].payload if points else None

    async def upsert(self, vector: list[float], payload: dict[str, Any]) -> None:
        # Repeated writes of the same scoped request replace rather than duplicate.
        identity = f"{payload['scope_id']}:{payload['request_id']}"
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=str(uuid5(NAMESPACE_URL, identity)), vector=vector, payload=payload)],
            wait=True,
        )

    async def close(self) -> None:
        await self.client.close()
