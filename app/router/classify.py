from fastapi import APIRouter, HTTPException

from app.schemas.ticket import ClassifyRequest, ClassifyResponse, VALID_LABELS
from app.services.classifier import classify_ticket

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse, status_code=200)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    result = await classify_ticket(request.ticket)

    # Last-resort guard against a model that ignores the label constraint in
    # the Signature's desc=. In practice the desc= handles this, but if the
    # model starts returning unexpected labels frequently, tighten the desc=
    # or add an Assert in the DSPy module (see M19).
    invalid = [l for l in result.labels if l not in VALID_LABELS]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Model returned unexpected labels {invalid}. Valid: {sorted(VALID_LABELS)}",
        )

    return result
