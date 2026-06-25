"""
Milestone 25 — RAG Triad Evaluation

OBJECTIVE:
    Evaluate the RAG pipeline across three independent dimensions — Context Relevance,
    Faithfulness, and Answer Relevance — to isolate which component is responsible
    when reply quality is low. M24 measured end-to-end output quality. M25 opens the
    pipeline and measures each step separately.

HOW TO RUN (from the project root, with venv active):
    python experiments/m25_rag_triad.py

THE THREE DIMENSIONS:
    Each dimension has a deliberately different input set. This is the key design
    decision: conflating them into a single prompt makes it impossible to diagnose
    which component failed.

    1. Context Relevance — ticket + context → score
       Is the retrieved context actually relevant to this ticket?
       Low score means the retriever found the wrong passages. Everything downstream
       of a bad retrieval step is working with bad inputs; fixing K, expanding the
       knowledge base, or improving the embedding model are the right levers here.

    2. Faithfulness — reply + context → score (no ticket)
       Does the reply only claim things the context supports?
       The ticket is intentionally excluded: whether a reply is grounded in context
       is a relationship between the reply and the context alone. Low score means the
       model hallucinated facts or ignored the retrieved passages. The M22 pricing
       ticket finding is the canonical example: context contained the loyalty discount
       offer, the reply didn't mention it — that is a faithfulness failure.

    3. Answer Relevance — ticket + reply → score (no context, no expected reply)
       Does the reply actually address what the user asked?
       Context is excluded because whether a reply helps the user is independent of
       how it was generated. Expected reply is excluded because that would make this
       re-measure M24 quality, not RAG groundedness. Low score means the reply is
       technically coherent but off-topic to the user's actual need.

DIAGNOSTIC TABLE — reading the three scores together:
    CR=low,  F=any,  AR=low  → retriever is broken; fix K or knowledge base content
    CR=high, F=low,  AR=low  → model hallucinated; fix the grounding instruction
    CR=high, F=high, AR=low  → model used context correctly but missed the user's need;
                               fix the reply Signature (RetrievedReply docstring)
    CR=high, F=high, AR=high → pipeline is working end-to-end

WHAT TO LOOK FOR:
    - Compare Context Relevance scores against M22 findings: the pricing ticket
      had two relevant resolutions and one noise result. CR should be moderate (3-4)
      not high (5) for entries where the third retrieved result was clearly off-topic.
    - Compare Faithfulness against Answer Relevance: if F is high but AR is low,
      the model is faithfully using context that wasn't relevant to start with (the
      CR failure cascaded). If F is low but AR is high, the model gave a good answer
      from its own priors — which means RAG is not actually helping on those entries.
    - The two category misses from M23/M24 ([22] mobile slowness, [24] status page)
      should score well on all three dimensions despite the wrong label, since the
      context for performance tickets is relevant and the model used it faithfully.

PRODUCTION NOTES (M25):
    CR, F, and AR each require a separate LLM call, tripling evaluation cost.
    Run them offline on sampled production traffic — not in the request path.
    A weekly batch job that evaluates a random 2–5% sample is enough to detect
    systematic drift before it reaches users.

    Track CR, F, and AR as time-series metrics, not one-off scores. A downward
    trend in CR means KB quality is degrading (entries becoming stale, or new
    ticket types appearing that the KB doesn't cover). A downward F trend means
    the reply Signature is drifting away from grounded responses, possibly due
    to a Signature change that removed grounding instructions.

    The diagnostic table (CR/F/AR pattern → root cause) in this file is the
    right mental model for triage. Build it into a dashboard so on-call can read
    the pattern without re-deriving it from first principles.

CRITICAL — set lm_mode=direct in .env for experiment runs.

FINDINGS FROM EXECUTION (25 golden examples, RAG pipeline, qwen2.5:14b):

    Context Relevance avg : 4.5 / 5
    Faithfulness avg      : 4.4 / 5
    Answer Relevance avg  : 4.6 / 5

    All three dimensions score high on happy-path inputs, validating the pipeline
    baseline before adversarial entries are added in the robustness milestone.

    DIVERGENCE: CR≥4 but F≤2 (retriever worked, model contradicted context):
      [06] Bulk delete — CR=5, F=2, AR=4
      The knowledge base resolution says the bug was already fixed (JSON parsing
      error, patch released). The model's reply presented it as ongoing and invented
      a workaround not in the context. Perfect retrieval, hallucinated claim. This
      is the canonical faithfulness failure: the model contradicted what it was given.

    DIVERGENCE: F≥4 but AR≤2 (faithful to context, but missed the user's actual need):
      [12] Keyboard shortcuts — CR=3, F=4, AR=2
      A cascade failure. The knowledge base says "keyboard shortcuts already exist,
      press ? to see them." CR=3 because retrieval was only partial. The model
      faithfully used that partial context (F=4) but produced a reply that deferred
      the feature to the roadmap — treating a request as if the feature doesn't exist,
      when the context says it does. Root cause: knowledge base coverage gap that
      cascaded into a wrong reply through faithful but misleading grounding.

    CONTRADICTION BETWEEN M24 AND M25 — [07] Dark mode:
      M24 score=2 (judge said reply claimed resolution of an ongoing bug)
      M25 scores CR=5, F=5, AR=5 (all dimensions passing)
      Both are correct. The knowledge base has a "fixed" resolution for dark mode;
      the model faithfully used it. But the golden dataset expected reply was written
      assuming the bug is still ongoing. The Triad measures correctness against the
      knowledge base; M24 measures it against the human-authored reference. When the
      two sources disagree, the Triad and M24 will contradict each other. This is
      a data consistency issue: the knowledge base and the golden dataset need to
      agree on whether a given issue is resolved or ongoing.

    FAITHFULNESS CONFIRMS M22 PRICING FINDING — [25] Three-year customer:
      CR=5 (loyalty discount resolution retrieved correctly)
      F=3  (model acknowledged loyalty but didn't extract the specific discount offer)
      AR=4 (reply was still helpful)
      The retriever found the right context; the model didn't use the most actionable
      fact in it. Faithfulness score isolates this as a grounding failure, not a
      retrieval failure — a distinction impossible to make from M24's end-to-end score.

FINDINGS FROM ROBUSTNESS RUN (34 golden examples — 25 original + 6 out-of-scope + 3 wrong-premise):

    Context Relevance avg : 4.0 / 5  (was 4.5 — dropped 0.5)
    Faithfulness avg      : 4.6 / 5  (was 4.4 — rose slightly, partly artificial)
    Answer Relevance avg  : 4.1 / 5  (was 4.6 — dropped 0.5)

    The CR and AR drops are real and meaningful. This is exactly the signal M26
    CI/CD gates are designed to catch: a dataset change caused measurable quality
    regression that would be invisible without automated evaluation.

    FINDING — Wrong-premise corrections worked perfectly:
      [32] Mobile app claim: CR=5, F=5, AR=5
      [33] Dark mode claim:  CR=5, F=5, AR=5
      [34] Bulk delete claim: CR=5, F=5, AR=5
      All three retrieved the correction entries from the expanded knowledge base
      and the model used them to redirect the user correctly.

    FINDING — KB pollution caused [03] retrieval regression (CR: 3 -> 1):
      The new correction entries (mobile, dark mode, bulk delete, feature discovery)
      contaminated FAISS retrieval. For "images not showing up after saving", the
      retriever now pulls a feature-discovery entry instead of the CDN/S3 resolution.
      The model answered from priors (AR=4) but context was useless (CR=1, F=1).
      Adding entries changes the embedding space and shifts nearest neighbours for
      existing queries. M26 gate: CR average below 4.2 would fail the suite here.

    FINDING — Salad [30] misclassified as feedback instead of out_of_scope:
      "I really love this salad I had for lunch today" triggered full RAG instead
      of the short-circuit. The classifier docstring described feedback as "praise,
      complaints, rants, or opinions" — a salad compliment fits linguistically.
      Fix applied: out_of_scope description now explicitly includes "anything that
      has no connection to software, accounts, or support", making the product
      boundary explicit rather than relying on the model to infer it.

    FINDING — Vacuous faithfulness for out-of-scope entries [26-29, 31]:
      All clear out-of-scope entries scored F=5 with context=[]. When context is
      empty, faithfulness is trivially true and meaningless. The divergence detector
      flags these as "F>=4 but AR<=2 -> faithful but unhelpful" — the wrong
      diagnosis. M26 should filter out_of_scope entries from CR and F averages,
      since those dimensions are undefined for empty context.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dspy
from pydantic import BaseModel, Field

from app.services.classifier import configure_dspy
from app.services.pipeline import TicketPipeline, initialize_retriever


def load_golden_dataset() -> list[dict]:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "golden_dataset.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TriadScore — the structured verdict each dimension judge returns
#
# Reusing the same shape for all three dimensions: a score from 1-5 and a
# one-sentence reason. The reason is especially important here because the
# three scores only become actionable when you know *why* each one was assigned.
# A CR=3 with reason "second result was unrelated to the ticket's error type"
# tells you exactly what to fix; CR=3 alone tells you nothing.
# ---------------------------------------------------------------------------

class TriadScore(BaseModel):
    score: int = Field(..., ge=1, le=5, description="Quality rating 1 (poor) to 5 (excellent)")
    reason: str = Field(..., description="One sentence explaining the score")


# ---------------------------------------------------------------------------
# ContextRelevance — does the retrieved context actually address this ticket?
#
# Penalizes context that is generic, off-topic, or from the wrong category
# even if it is internally coherent. A bug resolution about CSV export is
# internally coherent but useless as context for a login redirect issue.
#
# Scoring rubric (baked into the docstring so the judge sees it):
#   5 — all retrieved passages directly address the ticket's specific issue
#   4 — most passages are relevant, one is loosely related
#   3 — mixed: at least one passage is directly relevant, others are off-topic
#   2 — passages are from the right category but miss this specific issue
#   1 — retrieved content is entirely unrelated to the ticket
#
# Note: out_of_scope tickets will always score 1 here because context=[] is
# empty by design (the short-circuit fires before retrieval). Filter these
# entries before computing CR averages — see M26.
# ---------------------------------------------------------------------------

class ContextRelevance(dspy.Signature):
    """Evaluate the relevance of the retrieved context to the ticket. Consider
    whether the passages directly address the user's issue, are loosely related,
    or are off-topic. A high score means the context is well-targeted to the
    ticket; a low score means it is not useful for generating a helpful reply."""

    ticket: str = dspy.InputField(desc="The original customer support message")
    context: str = dspy.InputField(desc="The retrieved resolutions provided as context for reply generation")
    score: int = dspy.OutputField(desc="Relevance rating from 1 (irrelevant) to 5 (highly relevant)")
    reason: str = dspy.OutputField(desc="One sentence explaining the relevance score")


# ---------------------------------------------------------------------------
# Faithfulness — does the reply only claim things the context supports?
#
# Ticket is intentionally absent. Faithfulness is a relationship between the
# reply and the context alone. Whether the original question is relevant to
# that relationship is a separate concern (AnswerRelevance handles that).
#
# Concrete example of what this catches:
#   Context says: "bug caused by empty date fields, fixed in v2.3.1"
#   Reply says:   "we released a fix" → F=5 (grounded)
#   Reply says:   "we're investigating" → F=2 (the context says it's fixed;
#                 the reply introduces an unsupported "ongoing" claim)
#
# Scoring rubric:
#   5 — every factual claim is directly supported by the context
#   4 — mostly grounded, one minor unsupported detail
#   3 — reply uses some context but introduces unsupported claims
#   2 — reply mostly ignores context and draws on model priors
#   1 — reply contradicts context or fabricates facts not in it
#
# Note: out_of_scope tickets score F=5 vacuously (empty context cannot be
# contradicted). This is a false signal — filter before computing averages.
# ---------------------------------------------------------------------------

class Faithfulness(dspy.Signature):
    """Evaluate the faithfulness of the reply to the retrieved context. A high
    score means the reply is well-grounded in the provided context, with every
    factual claim supported by it. A low score means the reply contains
    hallucinated information not found in the context or ignores key facts from it."""

    context: str = dspy.InputField(desc="The retrieved resolutions provided as context for reply generation")
    predicted_reply: str = dspy.InputField(desc="The support reply generated by the pipeline")
    score: int = dspy.OutputField(desc="Faithfulness rating from 1 (not faithful) to 5 (fully faithful)")
    reason: str = dspy.OutputField(desc="One sentence explaining the faithfulness score")

# ---------------------------------------------------------------------------
# AnswerRelevance — does the reply address what the user actually needed?
#
# Both context and expected_reply are intentionally absent:
#   - Context is excluded because whether a reply helps the user is independent
#     of how it was generated (priors vs retrieved facts). A model can answer
#     correctly from its own knowledge with zero retrieved context.
#   - Expected reply is excluded because including it would re-measure M24
#     quality (does it match the reference?) rather than standalone usefulness.
#
# The diagnostic value: F-high + AR-low means the model faithfully used
# context that wasn't relevant to the user's actual need. That is a retriever
# problem disguised as a reply problem. AR-high + F-low means the model
# answered well from priors — RAG is not contributing on that entry.
#
# Scoring rubric:
#   5 — reply directly and fully addresses the user's need
#   4 — reply mostly addresses the need, minor omission
#   3 — reply partially addresses the need but misses a key point
#   2 — reply is on-topic but answers the wrong question
#   1 — reply does not address what the user asked
# ---------------------------------------------------------------------------

class AnswerRelevance(dspy.Signature):
    """Evaluate the relevance of the reply to the user's actual need as expressed in the ticket. 
    A high score means the reply directly and fully addresses the user's issue or request. 
    A low score means the reply is off-topic, misses key points, or answers a different question than what was asked.""" 

    ticket: str = dspy.InputField(desc="The original customer support message")
    predicted_reply: str = dspy.InputField(desc="The support reply generated by the pipeline")
    score: int = dspy.OutputField(desc="Answer relevance rating from 1 (not relevant) to 5 (highly relevant)")
    reason: str = dspy.OutputField(desc="One sentence explaining the answer relevance score")

# ---------------------------------------------------------------------------
# evaluate_triad — runs all three judges and returns three TriadScores
#
# Three separate predictor instances, one per Signature. The input sets differ
# across dimensions — context is not passed to AnswerRelevance, ticket is not
# passed to Faithfulness. This enforces the separation of concerns at the call
# site: each judge literally cannot consider information it's not given.
#
# Parse failure handling follows the same pattern as M24: int() on a DSPy
# string output, with a ValueError raise on failure. For a three-dimension
# evaluation, letting a parse failure surface immediately is preferable to
# clamping — a bad score on one dimension will skew all three averages in
# the summary table.
# ---------------------------------------------------------------------------

def evaluate_triad(
    ticket: str,
    context: str,
    predicted_reply: str,
) -> tuple[TriadScore, TriadScore, TriadScore]:
    # Each Signature is wrapped in ChainOfThought — Signatures are field declarations,
    # not callable modules. ChainOfThought(Signature) is the module that knows how to
    # build and send the prompt. The returned dspy.Prediction has string fields even
    # when the OutputField is typed as int, so int() is required on every score.
    cr = dspy.ChainOfThought(ContextRelevance)(ticket=ticket, context=context)
    f  = dspy.ChainOfThought(Faithfulness)(context=context, predicted_reply=predicted_reply)
    ar = dspy.ChainOfThought(AnswerRelevance)(ticket=ticket, predicted_reply=predicted_reply)

    return (
        TriadScore(score=int(cr.score), reason=cr.reason),
        TriadScore(score=int(f.score),  reason=f.reason),
        TriadScore(score=int(ar.score), reason=ar.reason),
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_triad_evaluation() -> None:
    dataset = load_golden_dataset()
    pipeline = TicketPipeline()

    cr_scores: list[int] = []
    f_scores: list[int] = []
    ar_scores: list[int] = []

    print(f"Running RAG Triad evaluation on {len(dataset)} golden examples...\n")
    print(f"{'─' * 75}")
    print(f"{'':4} {'CR':>4}  {'F':>4}  {'AR':>4}  ticket")
    print(f"{'─' * 75}")

    for i, example in enumerate(dataset, 1):
        pred = pipeline(ticket=example["ticket"])

        context = "\n\n".join(
            f"Past resolution {j+1}: {r}"
            for j, r in enumerate(pred.retrieved_context or [])
        )

        cr_score, f_score, ar_score = evaluate_triad(
            ticket=example["ticket"],
            context=context,
            predicted_reply=pred.reply
        )

        print(f"[{i:02d}] CR={cr_score.score}  F={f_score.score}  AR={ar_score.score}  | {example['ticket'][:60]}")
        print(f"      CR reason: {cr_score.reason}")
        print(f"      F  reason: {f_score.reason}")
        print(f"      AR reason: {ar_score.reason}")
        print()
        cr_scores.append(cr_score.score)
        f_scores.append(f_score.score)
        ar_scores.append(ar_score.score)

    print(f"{'─' * 75}\n")

    print(f"Context Relevance avg : {sum(cr_scores)/len(cr_scores):.1f} / 5")
    print(f"Faithfulness avg      : {sum(f_scores)/len(f_scores):.1f} / 5")
    print(f"Answer Relevance avg  : {sum(ar_scores)/len(ar_scores):.1f} / 5")

    # Divergence cases are the most actionable findings — they point to a specific
    # pipeline component rather than a general quality drop.
    triad = list(zip(cr_scores, f_scores, ar_scores))

    def divergence(label: str, matches: list) -> None:
        if matches:
            print(f"\n{label}:")
            for i, cr, f, ar in matches:
                print(f"  [{i:02d}] CR={cr}  F={f}  AR={ar}  | {dataset[i-1]['ticket'][:60]}")
        else:
            print(f"\n{label}: none")

    divergence(
        "CR≥4 but F≤2 → retriever worked, model ignored or contradicted context",
        [(i, cr, f, ar) for i, (cr, f, ar) in enumerate(triad, 1) if cr >= 4 and f <= 2],
    )
    divergence(
        "F≥4 but AR≤2 → model faithfully used context that wasn't helpful to the user",
        [(i, cr, f, ar) for i, (cr, f, ar) in enumerate(triad, 1) if f >= 4 and ar <= 2],
    )
    divergence(
        "CR≤2 but AR≥4 → model answered well from priors; RAG not contributing",
        [(i, cr, f, ar) for i, (cr, f, ar) in enumerate(triad, 1) if cr <= 2 and ar >= 4],
    )


if __name__ == "__main__":
    configure_dspy()
    initialize_retriever()
    run_triad_evaluation()
