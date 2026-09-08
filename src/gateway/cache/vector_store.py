from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from gateway.config import settings
import uuid

class VectorStore:
    def __init__(self):
        self.client = AsyncQdrantClient(host=settings.qdrant_host,
                                        port=settings.qdrant_port)
        self.collection_name = "queries"
        self.vector_size = 384  # Size for all-MiniLM-L6-v2


    async def ensure_collection(self):
        """Create the collection if it doesn't exist."""
        collection_exists = await self.client.has_collection(self.collection_name)
        if not collection_exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )


    async def search(self, vector: list[float], threshold: float) -> dict | None:
        """Search for a similar vector. Return its payload if similarity > threshold."""
        # Use self.client.search
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=1,
            score_threshold=threshold
        )
        # Return the payload (the cached response) if a match is found, else None
        if results:
            return results[0].payload
        return None
    
        async def upsert(self, vector: list[float], payload: dict):
            """Save a new vector and its payload."""
            id = str(uuid.uuid4())  # Generate a unique ID (e.g., str(uuid.uuid4()))
            point = PointStruct(id=id, vector=vector, payload=payload)     # Create a PointStruct(id=..., vector=..., payload=...)

            # Use self.client.upsert
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )   
            pass