from pydantic import BaseModel, validator


class EmbedRequest(BaseModel):
    text: str

    @validator("text")
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text must not be empty")
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
    embedding: list[float]


class SimilarityResponse(BaseModel):
    similarity: float
