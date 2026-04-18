from typing import List, Sequence, Union

import numpy as np
from prometheus_client import Histogram

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
inference_latency = Histogram(
    "embedding_inference_latency_seconds",
    "Inference latency for embedding generation",
)


def load_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install runtime dependencies before starting the app."
            ) from exc

        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{MODEL_NAME}'. Ensure the model can be downloaded and the environment has network access. Error: {exc}"
            ) from exc
    return _model


def get_embedding(texts: Union[str, Sequence[str]]) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(texts, Sequence) or isinstance(texts, str):
        raise ValueError("get_embedding requires a string or list of strings")

    normalized_texts = [item for item in texts]
    if not normalized_texts or any(not isinstance(item, str) or not item.strip() for item in normalized_texts):
        raise ValueError("texts must be a non-empty list of non-empty strings")

    model = load_model()
    with inference_latency.time():
        vectors = model.encode(normalized_texts, convert_to_numpy=True)

    if vectors.ndim == 1:
        vectors = np.expand_dims(vectors, 0)
    return vectors.tolist()


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    arr1 = np.asarray(vector1, dtype=np.float32)
    arr2 = np.asarray(vector2, dtype=np.float32)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))
