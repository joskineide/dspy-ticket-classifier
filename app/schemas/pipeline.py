from pydantic import BaseModel, Field, field_validator

from app.services.output_guard import validate_reply

# Field() attaches metadata to a model attribute beyond its type annotation.
# It takes two kinds of arguments:
#
#   First positional argument — the default value:
#     ...   (Ellipsis) means the field is REQUIRED. Pydantic will reject any
#           model instance that omits it entirely.
#     None  (or any value) means the field is OPTIONAL with that as the default.
#
#   description= — a string that serves two consumers simultaneously:
#     1. FastAPI reads it and renders it in the auto-generated OpenAPI docs at
#        /docs, so API callers know what each field contains.
#     2. It documents intent for anyone reading the code.
#
# Compare with dspy.InputField(desc=...) in the Signatures: same idea, different
# consumer. Pydantic sends description= to the API client via OpenAPI; DSPy
# injects desc= directly into the prompt the model sees. Both use the string as
# a hint — just to different audiences.


class PipelineRequest(BaseModel):
    # min_length=1 is an extra Pydantic validator on top of the type.
    # An empty string is technically a valid str, but not a valid ticket.
    ticket: str = Field(..., min_length=1, description="Raw customer support message")


class PipelineResponse(BaseModel):
    labels: list[str] = Field(..., description="Classification result: exactly one of bug, feature, feedback")

    # reasoning is optional because it only exists when the pipeline uses
    # ChainOfThought. If you swap to dspy.Predict, there is no reasoning trace
    # and the field comes back as null in the JSON instead of causing a
    # validation error. `str | None` is the type; `default=None` is the value
    # Pydantic uses when the field is absent.
    reasoning: str | None = Field(
        default=None,
        description="Chain-of-thought reasoning from the classify step",
    )

    # reply is required (... as default) — the pipeline always produces one.
    reply: str = Field(..., description="Suggested support reply drafted for the ticket")

    @field_validator("reply")
    @classmethod
    def reply_content_is_safe(cls, v: str) -> str:
        return validate_reply(v)

    # retrieved_context exposes what the retriever fetched so callers (and M25 traces)
    # can see exactly what grounded the reply. Optional: absent when RAG is disabled.
    retrieved_context: list[str] | None = Field(
        default=None,
        description="Past resolutions retrieved from the knowledge base for this ticket",
    )

    model: str = Field(..., description="The model that produced this response")
