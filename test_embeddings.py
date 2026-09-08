from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model directly
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
v1 = model.encode("What is the capital of France?")
v2 = model.encode("What's France's capital city?")
v3 = model.encode("How do I bake a chocolate cake?")

print(len(v1))  # Will output 384

# Compute cosine similarity manually:
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_sim(v1, v2))  # ~0.90+ (semantically similar)
print(cosine_sim(v1, v3))  # ~0.08 (completely different)