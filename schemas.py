from pydantic import BaseModel, validator


class EmbedRequest(BaseModel):
    texts: list[str]

    @validator("texts")
    def texts_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value or not isinstance(value, list):
            raise ValueError("texts must be a non-empty list of strings")
        if any(not item or not item.strip() for item in value):
            raise ValueError("texts must contain non-empty strings")
        return value


class SimilarityRequest(BaseModel):
    text1: str
    text2: str

    @validator("text1")
    def text1_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text1 must not be empty")
        return value

    @validator("text2")
    def text2_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text2 must not be empty")
        return value


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class SimilarityResponse(BaseModel):
    similarity: float
