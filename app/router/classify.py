from fastapi import APIRouter, HTTPException

from app.schemas.ticket import ClassifyRequest, ClassifyResponse, VALID_LABELS
from app.services.classifier import classify_ticket

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse, status_code=200)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    # TODO: once classify_ticket() is implemented, remove the NotImplementedError
    #       handler below — it's only here so the server starts during scaffolding.
    try:
        result = await classify_ticket(request.ticket)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))

    # Guard against a model that ignores the label constraint entirely.
    # This is a last-resort check; the desc= on the OutputField should handle
    # it in practice. If you're seeing this error often, tighten the desc=.
    if result.label not in VALID_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"Model returned unexpected label '{result.label}'. Valid: {sorted(VALID_LABELS)}",
        )

    return result
