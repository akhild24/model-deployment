from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from schemas import EmbedRequest, SimilarityRequest, EmbedResponse, SimilarityResponse
from service import load_model, get_embedding, cosine_similarity

app = FastAPI(title="Sentence Embeddings API")
instrumentator = Instrumentator()


@app.on_event("startup")
def startup_event() -> None:
    """Load the sentence-transformers model and attach Prometheus metrics."""
    load_model()
    instrumentator.instrument(app).expose(app)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid input. Please provide valid JSON with the required fields."},
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    embeddings = get_embedding(request.texts)
    return EmbedResponse(embeddings=embeddings)


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(request: SimilarityRequest) -> SimilarityResponse:
    embedding1 = get_embedding(request.text1)[0]
    embedding2 = get_embedding(request.text2)[0]
    similarity_score = cosine_similarity(embedding1, embedding2)
    return SimilarityResponse(similarity=similarity_score)
