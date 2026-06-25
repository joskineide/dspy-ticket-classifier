"""
Milestone 22 — RAG Pipeline: Grounded Reply Generation

OBJECTIVE:
    Extend TicketPipeline with a retrieval step so replies are grounded in
    past resolutions from the knowledge base rather than model priors alone.

HOW TO RUN (from the project root, with venv active):
    python experiments/m22_rag_pipeline.py

WHAT TO OBSERVE:
    - The retrieved context printed before each reply — this is exactly what
      the model receives in the `context` field of the prompt.
    - dspy.inspect_history(n=4) shows TWO LLM prompts (classify + reply).
      The reply prompt now contains the context field with the retrieved
      resolutions. Retrieval itself is not an LLM call so it has no history entry.
    - Compare the RAG reply against the M20 (no context) reply for the same
      ticket: the grounded reply should reference specific actions or outcomes
      from the retrieved resolutions rather than generic support phrasing.

RAG vs FEW-SHOT — what changed:
    M20 reply predictor: ticket + category → reply
    M22 reply predictor: ticket + category + context → reply

    The `context` field is dynamic — different for every ticket, fetched live
    from the knowledge base. Few-shot demonstrations (injected by the optimizer)
    are static — the same examples baked into every prompt at compile time.
    Both improve quality but fix different problems:
      Few-shot → teaches the model HOW to reason and format
      RAG      → gives the model the FACTS to reason over

PRODUCTION NOTES (M22):
    The retriever should be a separate service so it can be scaled, updated,
    and monitored independently of the LLM classify and reply steps. Retrieval
    latency already has its own span in M27 — a latency spike there indicates
    a vector DB problem, not an LLM problem.

    Consider adding a second retrieval step before classification using a
    product feature KB, so the classifier can resolve wrong-premise tickets
    with grounded data rather than Signature instructions alone. See insights.md
    for the full rationale and the architecture diagram.

    Track the retrieved context length per request. If KB entries grow longer
    over time, they inflate the reply prompt and increase token cost silently.
    Set a max_context_chars limit in retrieve() to bound this.

CRITICAL — set lm_mode=direct in .env for experiment runs.

FINDINGS FROM EXECUTION:
    Three tickets were tested: a bug (export crash), a feature request (scheduled
    reports), and a pricing complaint (feedback). Results showed RAG working at
    different quality levels depending on retrieval relevance.

    Bug — "The export button crashes every time I click it."
      Retrieved: PDF export docs [irrelevant], CSV null-pointer fix [directly relevant],
                 status page incident [irrelevant].
      Reply grounded the response in resolution [2], correctly referencing "empty fields
      that might trigger the crash" and the v2.3.1 patch by version number. This is
      the clearest example of RAG working: the model used a specific fact it could not
      have known from training alone.

    Feature — "I would love to schedule automatic weekly reports."
      Retrieved: scheduled report feature [directly relevant], PDF export [loosely related],
                 monthly aggregation bug [loosely related].
      Best result of the three. The reply gave the exact navigation path
      (Settings > Reports > Schedule), delivery options, and plan availability —
      all sourced verbatim from resolution [1]. Without context this would have been
      a generic "we'll consider it" feature-request response.

    Feedback — "Your pricing feels expensive compared to last year."
      Retrieved: loyalty discount offer [relevant], misleading pricing copy [relevant],
                 performance regression bug [completely irrelevant — noise].
      The reply was vague despite having two good resolutions available. The model
      acknowledged the concern but did not reference the loyalty discount or escalation
      path from resolution [1]. The irrelevant third result (a bug fix) diluted the
      context and the model did not extract the most actionable facts.

KEY FINDING — the K tradeoff is real:
    With K=3 and only 30 entries in the knowledge base, at least one retrieved result
    was noise in two of the three tests. The model sometimes uses the noise to hedge
    ("we'll explore options") rather than commit to the specific action in the good
    results. Reducing K to 2 or expanding the knowledge base would both help.

KEY FINDING — RAG grounds facts, not tone:
    The bug and feature replies were measurably more specific than a no-context reply
    would be. The pricing reply had correct tone but missed the specific resolution
    action (loyalty discount). Tone is learned from training; specific facts require
    the model to actively use the retrieved context — which it does more reliably when
    the context is unambiguous and the noise-to-signal ratio is low.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dspy

from app.services.classifier import configure_dspy
from app.services.pipeline import TicketPipeline, initialize_retriever


def run_rag_smoke_test(ticket: str) -> None:
    pipeline = TicketPipeline()
    pred = pipeline(ticket=ticket)

    # pred.retrieved_context comes from TicketPipeline.forward() — the same
    # resolutions that were passed into the reply prompt as context.
    print(f"Ticket:  {ticket!r}")
    print("Retrieved context:")
    for i, r in enumerate(pred.retrieved_context or [], 1):
        print(f"  [{i}] {r[:110]}{'...' if len(r) > 110 else ''}")
    print()

    print(f"Label:  {pred.categories}")
    print(f"Reply:  {pred.reply}")
    print()


if __name__ == "__main__":
    configure_dspy()
    initialize_retriever()

    print("=== M22 RAG Pipeline ===\n")

    run_rag_smoke_test("The export button crashes every time I click it.")
    run_rag_smoke_test("I would love to schedule automatic weekly reports.")
    run_rag_smoke_test("Your pricing feels expensive compared to last year.")

    # inspect_history shows the reply prompt — find the `context` field in it
    # and confirm the retrieved resolutions appear verbatim inside the prompt.
    print("--- Prompt inspection (reply step) ---")
    dspy.inspect_history(n=2)
