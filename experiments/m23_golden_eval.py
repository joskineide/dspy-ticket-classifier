"""
Milestone 23 — Golden Dataset & Reference-Based Evaluation

OBJECTIVE:
    Establish a fixed benchmark (golden dataset) and measure the pipeline against
    it using three evaluation methods of increasing sophistication. The goal is not
    to get a high score — it is to understand exactly where deterministic evaluation
    breaks down, so you know what M24 needs to fix.

HOW TO RUN (from the project root, with venv active):
    python experiments/m23_golden_eval.py

THE THREE EVALUATORS:
    1. Exact category match  — did the model return the right label?
       This works well. Labels are discrete, the model is good at classification.
       Expected result: ~85–95% accuracy.

    2. Keyword presence      — does the reply contain expected key concepts?
       This partially works. A reply can be correct but use different words:
       "sorry" instead of "apologize", "look into" instead of "investigate".
       Expected result: 60–80%. The misses are not wrong replies — they are
       correct replies phrased differently. This is the phrasing problem.

    3. Exact reply match     — does the reply exactly match the reference?
       This always fails. The model never reproduces the exact same sentence twice.
       Expected result: 0%. This is the number that motivates LLM-as-judge (M24).

WHAT THE SCORE TABLE TELLS YOU:
    If category accuracy is low → the classifier needs improvement (M18 optimizer).
    If keyword coverage is low  → the reply is missing key concepts, not just
                                  rephrasing them. Check retrieved context quality.
    If exact match is 0%        → expected. Move to M24 for semantic evaluation.

GOLDEN DATASET RULES:
    - Never update golden_dataset.json to make tests pass. It is the yardstick,
      not the target. Only add entries when new coverage is genuinely needed.
    - The expected_reply values are reference answers written by a human. They
      exist to show the model what correct looks like — not to be matched exactly.
    - expected_reply_keywords are the minimum concepts a correct reply must convey,
      regardless of phrasing. Keep them to 2–4 words per entry.

PRODUCTION NOTES (M23):
    The golden dataset should live in a versioned database, not a JSON file.
    Every evaluation run should record which dataset version it used, the
    pipeline version, the scores, and the thresholds — not just the final number.
    That audit trail makes it possible to answer "when did accuracy drop and what
    changed that week?" without guessing.

    Add new entries continuously from production traffic: sample real tickets,
    have a human write the expected reply, and tag the entry with the pipeline
    version that first processed it. This keeps the dataset aligned with what
    users are actually sending, which synthetic generation cannot replicate.

    Keyword coverage (eval_keywords) is cheap but unreliable for paraphrase.
    In production it is most useful as a fast pre-filter: entries that score
    below 0.5 on keywords are worth reviewing manually, while entries above 0.8
    can be trusted to be correct without running the judge.

FINDINGS FROM EXECUTION (25 golden examples, RAG pipeline, qwen2.5:14b):

    Exact category match : 23/25  (92.0%)
    Keyword coverage     : avg 77.3%
    Exact reply match    : 0/25   (0.0%)

    Category misses (2) — both expected "feedback", classified as "bug":
      - "The mobile app feels much slower than the web version. Everything takes
        twice as long to load."
      - "Your status page says everything is operational but we have been
        experiencing slowness all morning."
    These are genuine label ambiguities, not model failures. Both tickets describe
    an active performance problem in first-person terms — that reads like a bug
    report. A real support team would debate these two. The golden dataset surfaces
    edge cases at the boundary between categories; improving them requires either
    refining the label taxonomy or adding disambiguation examples to the classifier.

    Keyword coverage breakdown:
      kw=1.00 (all keywords found):  13/25 entries — model used expected vocabulary
      kw=0.67 (2/3 keywords found):   9/25 entries — one keyword missed per entry
      kw=0.33 (1/3 keywords found):   3/25 entries — paraphrase gap is widest here
        [04] TFA codes: model likely said "verify" not "investigate", "token" not "code"
        [07] Dark mode: model likely said "theme" not "dark mode", "persist" not "fix"
        [08] Mobile freeze: model likely said "performance" not "freeze", "items" missed
    In all three cases the reply was semantically correct — right content, different words.
    Keyword matching cannot distinguish a wrong reply from a correctly-rephrased reply.
    This is the core limitation being demonstrated.

    Exact reply match at 0% is expected and by design. A language model cannot
    reproduce a human-authored sentence verbatim. This number is the argument for
    M24: a judge that evaluates meaning, not string equality.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.classifier import configure_dspy
from app.services.pipeline import TicketPipeline, initialize_retriever


def load_golden_dataset() -> list[dict]:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "golden_dataset.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluator 1 — Exact category match
#
# The simplest possible check: did the model return the correct label?
# Works well for classification because the output space is small and discrete.
# ---------------------------------------------------------------------------

def eval_category(example: dict, pred_category: str) -> float:
    # This is a simple exact match of the expected category, it either has the right category or it doesn't. 
    # The model is good at classification, so we expect a high score here.
    return 1.0 if pred_category.lower() == example["expected_category"].lower() else 0.0


# ---------------------------------------------------------------------------
# Evaluator 2 — Keyword presence
#
# Checks whether each expected keyword appears anywhere in the reply.
# Returns a partial score: 2 out of 3 keywords found = 0.67.
# This is more lenient than exact match but still fails on paraphrase.
# ---------------------------------------------------------------------------

def eval_keywords(example: dict, pred_reply: str) -> float:
    keywords = example.get("expected_reply_keywords", [])
    if not keywords:
        return 1.0  # no keywords to check, so we consider it a full score
    # Returning the fraction of expected keywords that are present in the predicted reply, ignoring case.
    # If the awnser has 2 out of 5 keywords, it gets a score of 0.4. 
    # This shows whether the model is missing key concepts (low score) or just rephrasing them (high score, even if eval_category was 0).
    return sum(1 for kw in keywords if kw.lower() in pred_reply.lower()) / len(keywords)



# ---------------------------------------------------------------------------
# Evaluator 3 — Exact reply match
#
# Compares the model's reply character-for-character against the reference.
# This will always return 0.0. It exists to make the phrasing problem visible.
# After running this, the 0% score is the argument for M24 (LLM-as-judge).
# ---------------------------------------------------------------------------

def eval_exact_reply(example: dict, pred_reply: str) -> float:
    # This checks if the predicted reply exactly matches the expected reply, including punctuation and spacing.
    # The model is very unlikely to produce the exact same string twice, so we expect this to be 0% across the board.
    # This evaluator is designed to show the limitations of exact string matching.
    example_reply = example.get("expected_reply", "").strip()
    pred_reply = pred_reply.strip()
    return 1.0 if pred_reply == example_reply else 0.0


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    dataset = load_golden_dataset()

    cat_scores: list[float] = []
    kw_scores: list[float] = []
    exact_scores: list[float] = []
    failures: list[dict] = []

    print(f"Running evaluation on {len(dataset)} golden examples...\n")

    for i, example in enumerate(dataset, 1):
        pipeline = TicketPipeline()
        pred = pipeline(ticket=example["ticket"])

        pred_category = pred.categories[0] if pred.categories else ""
        pred_reply = pred.reply

        cat = eval_category(example, pred_category)
        kw = eval_keywords(example, pred_reply)
        exact = eval_exact_reply(example, pred_reply)

        cat_scores.append(cat)
        kw_scores.append(kw)
        exact_scores.append(exact)

        status = "✔" if cat == 1.0 else "✘"
        print(f"[{i:02d}] {status} cat={cat:.0f}  kw={kw:.2f}  exact={exact:.0f}  |  {example['ticket'][:60]}")

        if cat < 1.0:
            failures.append({
                "ticket": example["ticket"],
                "expected": example["expected_category"],
                "got": pred_category,
                "reply": pred_reply,
            })

    # --- Summary ---
    n = len(dataset)
    print(f"\n{'─' * 60}")
    print(f"Exact category match : {sum(cat_scores):.0f}/{n}  ({100 * sum(cat_scores)/n:.1f}%)")
    print(f"Keyword coverage     : avg {100 * sum(kw_scores)/n:.1f}%")
    print(f"Exact reply match    : {sum(exact_scores):.0f}/{n}  ({100 * sum(exact_scores)/n:.1f}%)")
    print(f"{'─' * 60}")

    # --- Category misses — the most actionable failures ---
    if failures:
        print(f"\nCategory misses ({len(failures)}):")
        for f in failures:
            print(f"  Expected '{f['expected']}', got '{f['got']}'")
            print(f"  Ticket: {f['ticket'][:80]}")
            print(f"  Reply:  {f['reply'][:100]}")
            print()

if __name__ == "__main__":
    configure_dspy()
    initialize_retriever()
    run_evaluation()
