from pydantic import BaseModel, Field


VALID_LABELS = {"bug", "billing", "feature", "security"}


class ClassifyRequest(BaseModel):
    ticket: str = Field(..., min_length=1, description="Raw customer support message to classify")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="One of: bug, billing, feature, security")
    reasoning: str | None = Field(
        default=None,
        description="Chain-of-thought reasoning produced by the model, if available",
    )
    model: str = Field(..., description="The model that produced this classification")
