from sentence_transformers import SentenceTransformer
from gateway.config import settings

class Embedder:
    def __init__(self):
        self.model_name = settings.embedding_model
        self._model = None  # Lazy loading

    @property
    def model(self) -> SentenceTransformer:
        """Load the model only when first needed to speed up startup."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Convert a single string into a vector."""
        vector = self.model.encode(text)
        return vector.tolist()