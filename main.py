from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import EmbedRequest, SimilarityRequest, EmbedResponse, SimilarityResponse
from service import load_model, get_embedding, cosine_similarity

app = FastAPI(title="Sentence Embeddings API")


@app.on_event("startup")
def startup_event() -> None:
    """Load the sentence-transformers model once when the app starts."""
    load_model()


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid input. Please provide valid JSON with the required fields."},
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    embedding = get_embedding(request.text)
    return EmbedResponse(embedding=embedding)


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(request: SimilarityRequest) -> SimilarityResponse:
    embedding1 = get_embedding(request.text1)
    embedding2 = get_embedding(request.text2)
    similarity_score = cosine_similarity(embedding1, embedding2)
    return SimilarityResponse(similarity=similarity_score)
