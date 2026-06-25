from pydantic import BaseModel, Field


VALID_LABELS = {"bug", "feature", "feedback", "out_of_scope"}


class ClassifyRequest(BaseModel):
    ticket: str = Field(..., min_length=1, description="Raw customer support message to classify")


class ClassifyResponse(BaseModel):
    labels: list[str] = Field(..., description="Exactly one of: bug, feature, feedback, out_of_scope")
    reasoning: str | None = Field(
        default=None,
        description="Chain-of-thought reasoning produced by the model, if available",
    )
    model: str = Field(..., description="The model that produced this classification")
