"""
Milestone 26 — CI/CD Quality Gating

OBJECTIVE:
    Turn every evaluation metric established in M23–M25 into a pytest assertion
    that fails automatically when quality drops. Running this file is the answer
    to "is it safe to ship this change?"

HOW TO RUN:
    pytest tests/test_quality_gates.py -v
    pytest tests/test_quality_gates.py -v -k test_category_accuracy   # single gate
    pytest tests/test_quality_gates.py -v --tb=short                  # compact output

PRODUCTION NOTES (M26):
    Run this suite automatically in CI on every PR that touches a Signature,
    KB content, the golden dataset, or the optimized program artifact — not
    only on application code changes. Prompt changes are code changes.

    Raise thresholds when new baselines are established. A threshold that never
    gets raised has stopped catching regressions — it becomes a false safety net.
    The commit that raises a threshold should cite the new baseline score and the
    change that produced it (e.g. "raise CR_AVG_MIN 4.0 → 4.3 after KB expansion").

    Shadow mode gating: before promoting a new pipeline version, run it alongside
    the current production version on the same live requests. Require that the new
    version passes this suite AND scores at least as well as the current version
    on the live sample. The suite alone is necessary but not sufficient — a version
    can pass all gates on the golden dataset and still regress on production traffic
    that the golden dataset doesn't cover.

WHY pytest AND NOT ANOTHER SCRIPT:
    The experiment scripts (m23–m25) are for exploration: you run them, read the
    output, and think. This file is for enforcement: it passes or fails, and the
    failure message tells you exactly what broke and by how much. pytest also
    integrates with any CI system (GitHub Actions, Jenkins, etc.) by returning a
    non-zero exit code on failure, which stops the deployment pipeline automatically.

THE SESSION FIXTURE PATTERN:
    Running the pipeline on 34 examples takes time — each example triggers at least
    two LLM calls (classify + reply). Without the session fixture, each of the four
    test functions below would re-run the full pipeline: 4 × 34 × 2 = 272 LLM calls
    per pytest run. With `scope="session"`, the pipeline runs once and all test
    functions share the results: 34 × 2 = 68 calls. The fixture is the most
    important performance decision in this file.

HOW THRESHOLDS WERE SET:
    Each threshold is set slightly below the baseline score established by the
    experiment runs, to tolerate natural model variance without masking real
    regressions. The baseline, the current threshold, and the tolerance gap are
    documented inline. When you improve the pipeline and baselines rise, raise
    the thresholds — a threshold that's too easy to pass stops catching regressions.

    Baseline (25 clean examples, M25 first run):
        Category accuracy  : 92%    → gate at 88%
        Judge avg score    : 4.1/5  → gate at 3.8
        Judge ≥4 rate      : 84%    → gate at 75%
        CR avg (no oos)    : 4.5/5  → gate at 4.0  (post-expansion baseline: 4.0)
        F  avg (no oos)    : 4.4/5  → gate at 4.0
        AR avg (all)       : 4.6/5  → gate at 3.8  (post-expansion baseline: 4.1)
        Out-of-scope rate  : ?      → gate at 70%  (set after first run with expanded data)

OUT-OF-SCOPE FILTERING:
    out_of_scope entries have context=[] by design (the short-circuit fires before
    retrieval). This makes CR=1 (trivially irrelevant) and F=5 (vacuously faithful)
    for all of them — both are noise, not signal. The fixture tags every prediction
    with its expected category so test functions can filter before computing averages.
    AR is NOT filtered: the canned redirect reply should still be evaluated as
    relevant-or-not to the request (even if "not relevant" is the correct answer for
    a greeting — that's a real AR=1 that belongs in the average).

WHAT EACH GATE CATCHES:
    test_category_accuracy  → classifier Signature or label taxonomy changes
    test_judge_quality      → reply Signature or knowledge base content changes
    test_out_of_scope_rate  → classifier boundary for out_of_scope label eroding
    test_rag_triad          → retriever changes (CR), grounding changes (F), reply drift (AR)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.classifier import configure_dspy
from app.services.pipeline import TicketPipeline, initialize_retriever

# Import evaluation functions from the experiment scripts so logic lives in one place.
# The experiment scripts remain runnable standalone for exploration; this file only
# adds assertions on top of them.
from experiments.m24_llm_judge import load_golden_dataset, evaluate_response
from experiments.m25_rag_triad import evaluate_triad


# ---------------------------------------------------------------------------
# Thresholds
#
# Change these only when you have a new, higher baseline to justify the move.
# Lowering a threshold to make a failing test pass is the same as deleting the gate.
# ---------------------------------------------------------------------------

CATEGORY_ACCURACY_MIN   = 0.88   # baseline 0.92; gap absorbs natural variance
JUDGE_AVG_MIN           = 3.8    # baseline 4.1
JUDGE_GE4_RATE_MIN      = 0.75   # baseline 0.84
OUT_OF_SCOPE_RATE_MIN   = 0.70   # at least 70% of out_of_scope entries correctly classified
CR_AVG_MIN              = 4.0    # baseline 4.5 (clean), 4.0 (post-expansion)
F_AVG_MIN               = 4.0    # baseline 4.4
AR_AVG_MIN              = 3.8    # baseline 4.6 (clean), 4.1 (post-expansion)


# ---------------------------------------------------------------------------
# Session fixture — runs the pipeline ONCE and shares results across all tests
#
# Returns a list of result dicts, one per golden example:
#   {
#     "ticket":            str,
#     "expected_category": str,
#     "pred_category":     str,
#     "pred_reply":        str,
#     "context":           str,   # formatted string passed to the reply step
#   }
#
# Judge scores and triad scores are NOT pre-computed here — each test function
# calls its own evaluator so that a failure in one evaluator doesn't prevent
# others from running. The pipeline output (the expensive part) is shared;
# the evaluation (the interpretive part) is kept per-test.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pipeline_results() -> list[dict]:
    configure_dspy()
    initialize_retriever()
    pipeline = TicketPipeline()
    results = []
    for example in load_golden_dataset():
        pred = pipeline(ticket=example["ticket"])
        context = "\n\n".join(
            f"Past resolution {j+1}: {r}"
            for j, r in enumerate(pred.retrieved_context or [])
        )
        results.append({
            "ticket":            example["ticket"],
            "expected_category": example["expected_category"],
            "expected_reply":    example["expected_reply"],
            "pred_category":     pred.categories[0] if pred.categories else "",
            "pred_reply":        pred.reply,
            "context":           context,
        })
    return results


# ---------------------------------------------------------------------------
# Gate 1 — Category accuracy
#
# Filters out out_of_scope entries before computing accuracy: those entries
# test the classifier boundary, not the classification quality of real tickets.
# out_of_scope detection is a separate gate (test_out_of_scope_rate).
# ---------------------------------------------------------------------------

def test_category_accuracy(pipeline_results):
    real_tickets = [r for r in pipeline_results if r["expected_category"] != "out_of_scope"]
    correct = sum(1 for r in real_tickets if r["pred_category"] == r["expected_category"])
    accuracy = correct / len(real_tickets)
    assert accuracy >= CATEGORY_ACCURACY_MIN, (
        f"Category accuracy {accuracy:.1%} below gate {CATEGORY_ACCURACY_MIN:.1%} "
        f"({correct}/{len(real_tickets)} correct)"
    )

# ---------------------------------------------------------------------------
# Gate 2 — LLM judge quality (from M24)
#
# Two sub-checks: average score and the fraction scoring ≥4. Both are needed
# because average score masks bimodal distributions (many 3s and many 5s can
# produce the same average as all 4s, but those are different quality profiles).
# ---------------------------------------------------------------------------

def test_judge_quality(pipeline_results):
    real_tickets = [r for r in pipeline_results if r["expected_category"] != "out_of_scope"]
    scores = [
        evaluate_response(r["ticket"], r["pred_reply"], r["expected_reply"]).score
        for r in real_tickets
    ]
    avg = sum(scores) / len(scores)
    ge4_rate = sum(1 for s in scores if s >= 4) / len(scores)
    assert avg >= JUDGE_AVG_MIN, f"Judge avg {avg:.2f} below gate {JUDGE_AVG_MIN}"
    assert ge4_rate >= JUDGE_GE4_RATE_MIN, (
        f"Judge ≥4 rate {ge4_rate:.1%} below gate {JUDGE_GE4_RATE_MIN:.1%}"
    )


# ---------------------------------------------------------------------------
# Gate 3 — Out-of-scope detection rate
#
# Measures how many out_of_scope entries the classifier actually identifies
# as out_of_scope. A drop here means the classifier boundary is eroding —
# lifestyle/unrelated messages are leaking into the real ticket pipeline.
# ---------------------------------------------------------------------------

def test_out_of_scope_rate(pipeline_results):
    oos_entries = [r for r in pipeline_results if r["expected_category"] == "out_of_scope"]
    detected = sum(1 for r in oos_entries if r["pred_category"] == "out_of_scope")
    rate = detected / len(oos_entries)
    assert rate >= OUT_OF_SCOPE_RATE_MIN, (
        f"Out-of-scope detection {rate:.1%} below gate {OUT_OF_SCOPE_RATE_MIN:.1%} "
        f"({detected}/{len(oos_entries)} detected)"
    )


# ---------------------------------------------------------------------------
# Gate 4 — RAG Triad (from M25)
#
# CR and F are computed only on non-out_of_scope entries (empty context makes
# both scores meaningless for those entries). AR is computed on all entries —
# even a redirect reply to a greeting should score low AR, and that is a real
# and correct signal that belongs in the average.
# ---------------------------------------------------------------------------

def test_rag_triad(pipeline_results):
    cr_scores, f_scores, ar_scores = [], [], []
    for r in pipeline_results:
        cr, f, ar = evaluate_triad(r["ticket"], r["context"], r["pred_reply"])
        if r["expected_category"] != "out_of_scope":
            cr_scores.append(cr.score)
            f_scores.append(f.score)
        ar_scores.append(ar.score)
    cr_avg = sum(cr_scores) / len(cr_scores)
    f_avg  = sum(f_scores)  / len(f_scores)
    ar_avg = sum(ar_scores) / len(ar_scores)
    assert cr_avg >= CR_AVG_MIN, f"Context Relevance avg {cr_avg:.2f} below gate {CR_AVG_MIN}"
    assert f_avg  >= F_AVG_MIN,  f"Faithfulness avg {f_avg:.2f} below gate {F_AVG_MIN}"
    assert ar_avg >= AR_AVG_MIN, f"Answer Relevance avg {ar_avg:.2f} below gate {AR_AVG_MIN}"
