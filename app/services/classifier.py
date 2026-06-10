# This module is the DSPy layer. The FastAPI router calls `classify_ticket()`;
# everything below that is DSPy-specific and left for you to implement.
#
# READING ORDER:
#   1. configure_dspy()   — wires the LM to Ollama or the gateway
#   2. ClassifyTicket     — the Signature: declares inputs and outputs
#   3. TicketClassifier   — the Module: wraps the predictor
#   4. classify_ticket()  — the public function the router calls

from app.config import settings
from app.schemas.ticket import ClassifyResponse

import dspy  # type: ignore[import-untyped]

dspy.LM("openai/ollama/", api_base=settings.gateway_url, api_key=settings.gateway_api_key)

# ---------------------------------------------------------------------------
# 1. LM configuration
#
# DSPy needs to know which LM to talk to. This is called once at startup
# from the FastAPI lifespan in main.py.
#
# The lm_mode setting (from .env / Settings) chooses the path:
#   "direct"  → dspy.LM("ollama/...", api_base=settings.ollama_api_base)
#   "gateway" → dspy.LM("openai/ollama/...", api_base=settings.gateway_url,
#                        api_key=settings.gateway_api_key)
#               The "openai/" prefix tells DSPy to use the OpenAI-compatible
#               client — which is exactly what the gateway exposes.
# ---------------------------------------------------------------------------

def configure_dspy() -> None:
    # TODO: build the dspy.LM(...) object based on settings.lm_mode
    #       and call dspy.configure(lm=lm)
    #
    # Hint: the model string for Option B is "openai/" + settings.classifier_model
    #       because DSPy needs to know to use the OpenAI client protocol.
    pass


# ---------------------------------------------------------------------------
# 2. Signature
#
# A Signature is a type contract. It declares what goes in and what must
# come out. DSPy reads this and constructs the actual prompt automatically.
#
# The class docstring becomes the task description in the generated prompt.
# The desc= on each field is the primary lever for output quality — without
# it the model has no constraint on format or valid values.
# ---------------------------------------------------------------------------

# TODO: define ClassifyTicket(dspy.Signature)
#
# class ClassifyTicket(dspy.Signature):
#     """..."""   ← TODO: write a one-sentence task description
#
#     ticket: str = dspy.InputField(desc="...")   ← TODO: describe what the field contains
#     label: str = dspy.OutputField(desc="...")   ← TODO: constrain to the four valid labels


# ---------------------------------------------------------------------------
# 3. Module
#
# A Module is a composable unit that wraps one or more predictors.
# In __init__ you choose the predictor type:
#   dspy.Predict(ClassifyTicket)        — single LLM call
#   dspy.ChainOfThought(ClassifyTicket) — adds a hidden "reasoning" field before
#                                         the answer; forces the model to reason
#                                         before committing to a label
#
# In forward() you call the predictor and return its output.
# ---------------------------------------------------------------------------

# TODO: define TicketClassifier(dspy.Module)
#
# class TicketClassifier(dspy.Module):
#     def __init__(self):
#         # TODO: initialise self.predictor with Predict or ChainOfThought
#         pass
#
#     def forward(self, ticket: str) -> dspy.Prediction:
#         # TODO: call self.predictor and return the result
#         pass


# ---------------------------------------------------------------------------
# 4. Public interface
#
# This is the only function the router imports. It runs the DSPy module and
# maps the prediction to the ClassifyResponse schema.
#
# classify_ticket is async so the router can await it uniformly with other
# async service calls. DSPy's LM calls are synchronous under the hood, so
# we run them with asyncio.to_thread to avoid blocking the event loop.
# ---------------------------------------------------------------------------

async def classify_ticket(ticket: str) -> ClassifyResponse:
    # TODO: instantiate TicketClassifier (or keep a module-level singleton)
    #       call it via asyncio.to_thread, and return a ClassifyResponse
    #
    # The reasoning field should be populated only when the predictor has it
    # (i.e. ChainOfThought was used). You can check with hasattr(pred, "reasoning").
    #
    # import asyncio
    # classifier = TicketClassifier()
    # pred = await asyncio.to_thread(classifier, ticket=ticket)
    # return ClassifyResponse(
    #     label=pred.label,
    #     reasoning=getattr(pred, "reasoning", None),
    #     model=settings.classifier_model,
    # )
    raise NotImplementedError("Implement classify_ticket() — see TODOs above")
