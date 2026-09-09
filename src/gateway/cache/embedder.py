"""Sentence-transformer embedding wrapper."""

from sentence_transformers import SentenceTransformer

from gateway.config import settings


class Embedder:
    """Creates vector embeddings for text."""

    def __init__(self) -> None:
        self.model_name = settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Load the model lazily on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        return self._model

    def embed(self, text: str) -> list[float]:
        """Convert one text string into an embedding vector."""
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple text strings into embedding vectors."""
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vectors.tolist()