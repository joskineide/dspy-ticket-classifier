from fastapi import APIRouter, HTTPException

from app.schemas.pipeline import PipelineRequest, PipelineResponse
from app.schemas.ticket import VALID_LABELS
from app.services.pipeline import run_pipeline

router = APIRouter()


@router.post("/pipeline", response_model=PipelineResponse, status_code=200)
async def pipeline(request: PipelineRequest) -> PipelineResponse:
    result = await run_pipeline(request.ticket)

    invalid = [l for l in result.labels if l not in VALID_LABELS]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Pipeline returned unexpected labels {invalid}. Valid: {sorted(VALID_LABELS)}",
        )

    return result
