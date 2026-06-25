"""
Milestone 24 — LLM-as-a-Judge

OBJECTIVE:
    Replace exact-string evaluation with a semantic judge: an LLM that reads both
    the predicted reply and the expected reply in context, then returns a structured
    score explaining its reasoning. This solves the paraphrase problem that made
    keyword coverage an unreliable signal in M23.

HOW TO RUN (from the project root, with venv active):
    python experiments/m24_llm_judge.py

WHY THIS SOLVES THE M23 PROBLEM:
    M23 results: 77.3% keyword coverage, 0% exact reply match.
    Most of those keyword misses were not wrong replies — they were correct replies
    in different words. "We'll investigate" vs "Our team will look into this" carry
    identical intent; keyword matching scores them differently, a judge scores them
    the same.

    The judge also catches failure modes keywords can't:
      - A reply that contains all expected keywords but is factually wrong
      - A reply that acknowledges the wrong issue (misread the ticket)
      - A reply that is technically correct but has the wrong tone (dismissive for feedback)

THE JUDGE IS ALSO A DSPy MODULE:
    The evaluator here is itself a dspy.Signature — not a hardcoded prompt, not
    a raw string passed to the LLM. This means:
      1. You can inspect its prompt with dspy.inspect_history()
      2. You could optimize it with MIPROv2 if you had human-rated score data
      3. Its behavior changes when you swap the underlying LM — same interface, new judge

    This is the recursive insight of DSPy: even your evaluation logic can be a
    declarative module, not a fixed script.

PRODUCTION NOTES (M24):
    Running the judge on every ticket in production is expensive — it adds one
    full LLM call per request. Sample instead: evaluate 5–10% of production
    traffic continuously, and evaluate 100% of tickets that triggered a Refine
    retry (already a signal of difficulty).

    Store judge scores in the trace (M27 span metadata) so you can correlate
    quality with latency, cost, and label. "Bug tickets consistently score lower
    than feature tickets" or "judge quality drops on Monday mornings" are
    findings that only emerge from production data at scale.

    Consider using a smaller, cheaper model for judging. The judge does not need
    to be the same model as the pipeline — a faster 7B model that understands
    the rubric is often sufficient, and its cost is fixed regardless of pipeline
    model choice.

BIASES TO WATCH FOR:
    1. Verbosity bias — longer replies tend to score higher even when shorter is better.
       A reply that says "We sincerely apologize for the inconvenience this has caused
       and our team is actively working..." may outscore "We're on it" even if brevity
       is more appropriate. Watch for it in the reasons.

    2. Position bias — always pass (predicted, expected) in the same order. If you swap
       them, scores can shift. Keep the order fixed; note it in your analysis.

    3. Leniency bias — on a 1–5 scale, judges rarely use 1 or 5. Your effective range
       is 2–4. A 3/5 is "mediocre" to a human but "average" to the judge. Keep this
       in mind when you set quality gate thresholds in M26.

    4. The judge can be wrong — it is probabilistic, not deterministic. Running the
       same pair twice may return different scores. This is expected behavior. The
       *average* judge score across many examples is reliable; individual scores are not.

WHAT TO LOOK FOR IN THE OUTPUT:
    - Entries where judge_score is high but keyword_score was low in M23
      → These are the paraphrase wins. The judge correctly ignores word choice.
    - Entries where judge_score is low — read the reason carefully.
      → Either the judge is wrong (verbose vs accurate), or there's a real quality
         problem the keyword evaluator missed (the reply addressed the wrong issue).
    - The two M23 category misses ([22] mobile slowness, [24] status page) — the judge
      should still score the replies for those fairly: even a bug-framed reply to a
      feedback ticket can be empathetic and helpful, just slightly off-topic.

CRITICAL — set lm_mode=direct in .env for experiment runs.

FINDINGS FROM EXECUTION (25 golden examples, RAG pipeline, qwen2.5:14b):

    Judge avg score : 4.1 / 5
    Score ≥ 4       : 21/25  (84.0%)
    Score = 3       : 3/25   (12.0%)
    Score ≤ 2       : 1/25   (4.0%)

    Compare against M23 keyword coverage (77.3%): the 6.7% gap is the paraphrase wins —
    entries that scored low on keywords because the model used different words, but scored
    4 or 5 on the judge because the meaning was equivalent.

    KEY FINDING — the judge caught a real quality failure keywords missed:
      [07] Dark mode toggle (score=2) — the pipeline told the user the bug was fixed,
      when the correct reply acknowledges it is ongoing and offers a workaround. Keyword
      matching found "dark mode" and gave partial credit. The judge read both replies,
      detected the factual contradiction, and correctly scored it as wrong. This is the
      core value of LLM-as-judge: not just paraphrase tolerance, but semantic understanding
      of what is actually being claimed.

    KEY FINDING — category error does not equal reply quality failure:
      [22] mobile slowness: cat=✘ (classified as bug, expected feedback), score=5
      [24] status page:     cat=✘ (classified as bug, expected feedback), score=4
      Both category misses produced replies that were still helpful and empathetic. The
      pipeline generated good support content even when the label was wrong. This separates
      two failure modes: classification quality and reply quality.

    KEY FINDING — the judge catches missing actions, not just wrong words:
      [03] score=3 — reply acknowledged the image upload issue but didn't ask for
                     diagnostic info (file format, size) that would help reproduce it.
      [16] score=3 — reply explained PDF export and Share Link but missed the critical
                     detail that Share Link requires no account — the user's actual need.
      [21] score=3 — reply acknowledged outdated docs but didn't ask which specific
                     pages, preventing the team from prioritizing the fix.
      In all three, keywords were present but a key *action* was absent. This is a content
      gap, not a phrasing gap. Keyword evaluation would have scored these as passing.

    LENIENCY BIAS observed: no score of 1 was assigned. The effective range was 2–5,
    consistent with what the bias section in this file predicts. The single score=2
    ([07]) was for a genuinely wrong reply, not just a mediocre one.
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
# ReplyJudgement — structured verdict from the judge
#
# Field(..., ge=1, le=5) on score enforces the range at Pydantic validation
# time, before the value reaches the evaluation loop. If the judge returns
# "4.5" or "seven", Pydantic raises ValidationError immediately. This is the
# same boundary-validation pattern used in the gateway for user input — applied
# here to LLM output instead.
#
# `reason` matters as much as `score`. When a score looks wrong, read the
# reason first — it reveals whether the judge misunderstood the task or
# correctly identified a real quality problem you hadn't noticed.
# ---------------------------------------------------------------------------

class ReplyJudgement(BaseModel):
    score: int = Field(..., ge=1, le=5, description="Quality rating 1 (poor) to 5 (excellent)")
    reason: str = Field(..., description="One sentence explaining the score")

# ---------------------------------------------------------------------------
# EvaluateReply — the judge's Signature
#
# The docstring is the scoring rubric injected verbatim into the prompt.
# Two design decisions worth noting:
#
# 1. "The expected reply is one example of a good response" — this instruction
#    prevents the judge from treating word choice as a quality signal. Without
#    it, the judge exhibits strong paraphrase bias: it penalizes "We'll look
#    into this" vs "Our team will investigate" even when they mean the same thing.
#
# 2. "avoid padding or unnecessary verbosity" — this partially mitigates
#    verbosity bias. The judge's training data skewed toward rating longer
#    replies higher; this instruction pushes back against that learned prior.
#    It reduces but doesn't eliminate the bias — it's fighting a weight, not
#    a rule.
#
# The `ticket` input field is context for judgment, not just decoration. Without
# the original ticket, the judge cannot evaluate whether the reply addresses the
# *right* issue — it could only compare phrasing against the reference, which is
# exactly what keyword matching already does.
# ---------------------------------------------------------------------------

class EvaluateReply(dspy.Signature):
    """
    You are a helpful and precise customer support quality evaluator.
    Given a support ticket, a predicted reply, and an expected reply, rate the quality of the predicted reply on a scale from 1 to 5, where:

    5 — conveys the same meaning and tone as the expected reply, even if worded differently
    4 — mostly correct, minor omission or slightly different tone
    3 — partially addresses the ticket but misses a key point or action
    2 — addresses the wrong issue or has a significantly wrong tone
    1 — wrong, irrelevant, or harmful

    When scoring, consider both the content (does it address the customer's issue?) and the tone (is it empathetic for feedback, appropriately formal for bugs?).
    The expected reply is one example of a good response, but different wording can still be excellent as long as it captures the same intent and tone.
    Answers should be objective and based on the ticket and expected reply, it should avoid padding or unnecessary verbosity that doesn't add value.

    Provide a one-sentence reason explaining your score."""

    ticket: str = dspy.InputField(desc="The original customer support message")
    predicted_reply: str = dspy.InputField(desc="The reply generated by the pipeline")
    expected_reply: str = dspy.InputField(desc="The human-written reference reply")
    score: int = dspy.OutputField(desc="Quality rating from 1 (poor) to 5 (excellent)")
    reason: str = dspy.OutputField(desc="One sentence explaining the score")


class EvaluateReplyWithHint(dspy.Signature):
    """
    You are a helpful and precise customer support quality evaluator.
    A previous attempt to score this reply returned a value that could not be parsed as an integer.
    The hint field describes the error. Return only a single integer from 1 to 5 as the score.

    5 — conveys the same meaning and tone as the expected reply, even if worded differently
    4 — mostly correct, minor omission or slightly different tone
    3 — partially addresses the ticket but misses a key point or action
    2 — addresses the wrong issue or has a significantly wrong tone
    1 — wrong, irrelevant, or harmful"""

    ticket: str = dspy.InputField(desc="The original customer support message")
    predicted_reply: str = dspy.InputField(desc="The reply generated by the pipeline")
    expected_reply: str = dspy.InputField(desc="The human-written reference reply")
    hint: str = dspy.InputField(desc="Description of the parse error from the previous attempt")
    score: int = dspy.OutputField(desc="A single integer from 1 to 5, nothing else")
    reason: str = dspy.OutputField(desc="One sentence explaining the score")

# ---------------------------------------------------------------------------
# evaluate_response — runs the judge and returns a validated verdict
#
# DSPy returns all OutputField values as strings even when the field is typed
# as `int`, so pred.score is always "4" not 4 — int() is required.
#
# On parse failure, this function retries using EvaluateReplyWithHint rather
# than retrying blindly. A blind retry at temperature=0 reproduces the same
# malformed output because there is no new signal for the model to act on.
# The hint describes the specific error, giving the judge something to correct
# against. If the second attempt also fails, it raises — two parse failures on
# the same input means the output desc in the Signature needs to be stricter.
#
# This only helps with format failures ("four" instead of "4"). If the judge
# returns a valid integer but a wrong score (5 for a clearly bad reply), no
# retry strategy helps — that is a semantic failure requiring a better rubric.
# ---------------------------------------------------------------------------

def evaluate_response(ticket: str, predicted_reply: str, expected_reply: str) -> ReplyJudgement:
    judge = dspy.ChainOfThought(EvaluateReply)
    pred = judge(ticket=ticket, predicted_reply=predicted_reply, expected_reply=expected_reply)

    try:
        score = int(pred.score)
        reason = pred.reason
        return ReplyJudgement(score=score, reason=reason)
    except ValueError:
        print(f"Judge score parse failed. Raw value: '{pred.score}'. Retrying with hint...")
        # Option D — retry using a separate Signature that includes the hint as an explicit
        # InputField. Extra kwargs passed to a DSPy predictor that aren't in its Signature
        # raise TypeError, so the hint must be declared as a field on a dedicated Signature.
        hint = f"Your previous score '{pred.score}' could not be parsed as an integer. Return only a number from 1 to 5."
        judge_retry = dspy.ChainOfThought(EvaluateReplyWithHint)
        pred_retry = judge_retry(ticket=ticket, predicted_reply=predicted_reply, expected_reply=expected_reply, hint=hint)
        try:
            score_retry = int(pred_retry.score)
            return ReplyJudgement(score=score_retry, reason=pred_retry.reason)
        except ValueError:
            # Two parse failures — the Signature's output desc needs to be stricter.
            raise ValueError(f"Judge score unparseable after two attempts. Last raw value: '{pred_retry.score}'")

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_judge_evaluation() -> None:
    dataset = load_golden_dataset()
    pipeline = TicketPipeline()

    judge_scores: list[int] = []

    print(f"Running judge evaluation on {len(dataset)} golden examples...\n")
    print(f"{'─' * 70}")

    for i, example in enumerate(dataset, 1):
        pred = pipeline(ticket=example["ticket"])
        pred_reply = pred.reply

        judgement = evaluate_response(
            ticket=example["ticket"],
            predicted_reply=pred_reply,
            expected_reply=example["expected_reply"],
        )
        print(f"[{i:02d}] score={judgement.score}  | cat={'✔' if pred.categories and pred.categories[0] == example['expected_category'] else '✘'}  | {example['ticket'][:60]}")
        print(f"      reason: {judgement.reason}")
        print()
        judge_scores.append(judgement.score)
    print(f"{'─' * 70}\n")
    n = len(dataset)
    print(f"Judge avg score : {sum(judge_scores)/n:.1f} / 5")
    print(f"Score ≥ 4       : {sum(1 for s in judge_scores if s >= 4)}/{n}  ({100 * sum(1 for s in judge_scores if s >= 4)/n:.1f}%)")
    print(f"Score = 3       : {sum(1 for s in judge_scores if s == 3)}/{n}  ({100 * sum(1 for s in judge_scores if s == 3)/n:.1f}%)")
    print(f"Score ≤ 2       : {sum(1 for s in judge_scores if s <= 2)}/{n}  ({100 * sum(1 for s in judge_scores if s <= 2)/n:.1f}%)")

    # inspect_history shows the exact prompt DSPy sent to the judge — the scoring
    # rubric from the docstring appears verbatim in the system message, and the three
    # input fields are formatted as labeled sections. Useful for verifying that the
    # rubric is reaching the model as intended and for debugging unexpected scores.
    dspy.inspect_history(n=1)


if __name__ == "__main__":
    configure_dspy()
    initialize_retriever()
    run_judge_evaluation()
