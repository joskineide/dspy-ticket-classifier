from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.router import classify as classify_router
from app.router import health as health_router
from app.router import pipeline as pipeline_router

from app.services.classifier import configure_dspy
from app.services.pipeline import initialize_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_dspy()
    # Build the FAISS retrieval index once at startup. Embedding 30 documents
    # takes a few seconds; doing it per-request would add that cost to every call.
    initialize_retriever()
    yield


app = FastAPI(title="DSPy Ticket Classifier", lifespan=lifespan)

app.include_router(health_router.router)
app.include_router(classify_router.router, prefix="/v1")
app.include_router(pipeline_router.router, prefix="/v1")
