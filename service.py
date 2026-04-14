from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embedding(text: str) -> List[float]:
    model = load_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    arr1 = np.asarray(vector1, dtype=np.float32)
    arr2 = np.asarray(vector2, dtype=np.float32)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))
