from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.router import classify as classify_router
from app.router import health as health_router

# TODO: import configure_dspy once the DSPy service is implemented
# from app.services.classifier import configure_dspy


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: call configure_dspy() here so the LM is wired before the first request.
    #       Everything before yield runs on startup, after yield on shutdown.
    # configure_dspy()
    yield


app = FastAPI(title="DSPy Ticket Classifier", lifespan=lifespan)

app.include_router(health_router.router)
app.include_router(classify_router.router, prefix="/v1")
