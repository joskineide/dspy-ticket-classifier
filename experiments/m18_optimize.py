"""
Milestone 18 — The Optimizer

OBJECTIVE:
    Use DSPy's BootstrapFewShot and MIPROv2 optimizers to automatically find
    few-shot demonstrations that improve the classifier's Jaccard score above
    the 91.67% baseline established in M17.

HOW TO RUN (from the project root, with venv active):
    python experiments/m18_optimize.py

CRITICAL — set lm_mode=direct in .env (same reason as m17_baseline.py):
    Gateway semantic cache collapses all scores to 1/N. Direct mode required.

WHAT TO OBSERVE:
    - The optimized score vs the 91.67% baseline
    - optimized_classifier.json — open it and read the demonstrations DSPy chose
    - That you wrote zero prompt engineering to get the improvement

FINDINGS FROM EXPERIMENTATION:
    The optimizer only works when three conditions are met simultaneously:
    1. The task is within the model's native capability (baseline > ~70%)
    2. The labels are structurally distinct with minimal semantic overlap
    3. Cache is bypassed so every evaluation call is a fresh LLM inference

    Datasets we tried and why they failed to show optimizer improvement:

    dair-ai/emotion (6 labels: joy, love, sadness, anger, fear, surprise)
      Baseline 62%. Optimizer flat at 60%. Root cause: joy/love and
      anger/sadness are inherently ambiguous in short text. The optimizer
      injected more ambiguous examples as demonstrations, which confused
      the model further. No prompt can fix semantic label overlap.

    dair-ai/emotion collapsed to 3 labels (positive, negative, surprise)
      Baseline 83%. Optimizer made things slightly worse (80%). Root cause:
      remaining errors were sarcasm and irony — "Oh great, another broken
      update!" reads as positive on the surface. That is a training data
      problem, not a prompt problem. Human inter-annotator agreement on
      social media sentiment sits at ~83%, so 83% was already near ceiling.

    Synthetic customer support (bug, feature, feedback) — THIS FILE:
      Baseline 91.67%. Bootstrap 95.83%. MIPROv2 100%.
      Works because the categories are structurally distinct (broken thing
      vs wanted thing vs opinion), the model has strong priors from
      developer/support text in its training data, and the synthetic
      generator produced clean, unambiguous examples.

    Key insight: label design matters more than optimizer choice.
    A well-matched 3-class problem showed full optimizer benefit.
    An ambiguous 6-class problem showed none, regardless of optimizer.

PRODUCTION NOTES (M18):
    Run optimization as a CI/CD step, not in the application. The optimizer
    should be triggered automatically when the dataset or Signatures change,
    not manually.

    Store the optimized .json artifact in versioned artifact storage (S3 or
    equivalent) alongside the git commit that produced it. The artifact and
    the Signature that generated it must be versioned together — a Signature
    change invalidates saved demonstrations.

    Before promoting a new optimized program to production, run it in shadow
    mode: serve both the current and new programs on live traffic, compare
    evaluation scores on the same requests, and only cut over when the new
    program is demonstrably better. Never promote based on devset score alone.

    Use auto="medium" or auto="heavy" for a final production optimization run.
    auto="light" is for iteration speed during development.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dspy

from app.services.classifier import configure_dspy, TicketClassifier

# Re-use the dataset and metric from M17 — no duplication needed.
# The optimizer and the evaluator must use the same metric so the scores
# are directly comparable to the 91.67% baseline.
from m17_baseline import TRAINSET, DEVSET, jaccard_metric

M17_BASELINE = 91.67  # synthetic customer support dataset, 3 labels, direct mode


# ---------------------------------------------------------------------------
# 1. Optimizer — BootstrapFewShot
#
# BootstrapFewShot is the simplest DSPy optimizer. It works in two steps:
#
#   Step 1 — Bootstrap:
#     Runs the unoptimized module on every example in TRAINSET.
#     For each example, it checks the metric. If the model got it right
#     (metric > 0), that (input, output) pair becomes a candidate demonstration.
#
#   Step 2 — Select:
#     Picks the best `max_bootstrapped_demos` candidates and injects them
#     as few-shot examples directly into the prompt.
#
# The result is a new module with the same Signature, but with demonstrations
# baked into every prompt. The model now sees "here are examples of correct
# behaviour" before each real ticket.
#
# What gets injected looks like this (with ChainOfThought):
#   Ticket: "The login button does nothing on Firefox."
#   Reasoning: The user is describing unexpected behavior on a specific browser...
#   Categories: ["bug"]
#
# max_bootstrapped_demos — model-generated (ticket + reasoning + prediction).
#   The reasoning trace can be verbose or subtly wrong — read the saved JSON
#   to verify what actually got injected. 2–3 is a safe starting point for
#   14B models; more risks context overflow or pattern-matching over reasoning.
#
# max_labeled_demos — raw (ticket, label) pairs from TRAINSET with no reasoning.
#   Less noise than bootstrapped demos, useful when the model struggles to
#   bootstrap clean examples on its own. 2 works well for this dataset size.
#
# Result on this dataset: 91.67% → 95.83%
# ---------------------------------------------------------------------------

def run_bootstrap() -> dspy.Module:
    optimizer = dspy.BootstrapFewShot(
        metric=jaccard_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
    )
    unoptimized = TicketClassifier()
    # compile() runs on TRAINSET and returns a NEW module — the original is
    # unchanged. The returned module has demonstrations injected into its
    # predictor's prompt template.
    optimized = optimizer.compile(student=unoptimized, trainset=TRAINSET)
    return optimized


# ---------------------------------------------------------------------------
# 2. Optimizer — MIPROv2
#
# MIPROv2 is the current state-of-the-art DSPy optimizer. It does two things
# BootstrapFewShot does not:
#
#   a) Instruction rewriting — generates multiple candidate task descriptions
#      (the Signature docstring equivalent), scores each on TRAINSET, and keeps
#      the best. This is why MIPROv2 can change *what the model is asked to do*,
#      not just which examples it sees.
#
#   b) Joint optimisation — searches over (instruction, demonstrations) pairs
#      together, not independently. This usually finds a better overall solution.
#
# The tradeoff: MIPROv2 makes many more LLM calls than BootstrapFewShot and
# can take 10–30 minutes on a local model. It also requires a model that can
# follow DSPy's JSON response format reliably during instruction generation —
# smaller or less instruction-tuned models (llama3.1) fail at this step.
# qwen2.5:14b handles it cleanly.
#
# auto="light" runs fewer candidates and trials — good for iteration.
# Use "medium" or "heavy" for a final production-quality optimization run.
#
# Result on this dataset: 91.67% → 100%
# ---------------------------------------------------------------------------

def run_mipro() -> dspy.Module:
    optimizer = dspy.MIPROv2(
        metric=jaccard_metric,
        auto="light",
    )
    unoptimized = TicketClassifier()
    optimized = optimizer.compile(
        unoptimized,
        trainset=TRAINSET,
        requires_permission_to_run=False,  # suppresses the interactive cost prompt
    )
    return optimized


# ---------------------------------------------------------------------------
# 3. Evaluate and compare
#
# Runs dspy.Evaluate on DEVSET — the same held-out set used for the M17
# baseline. Using the same devset is what makes the comparison honest.
# The optimizer only ever saw TRAINSET; a score improvement on DEVSET means
# the optimizer found something that genuinely generalises.
# ---------------------------------------------------------------------------

def evaluate(module: dspy.Module) -> float:
    evaluate_fn = dspy.Evaluate(
        devset=DEVSET,
        metric=jaccard_metric,
        num_threads=1,
        display_progress=True,
        display_table=5,
    )
    return float(evaluate_fn(module))


# ---------------------------------------------------------------------------
# 4. Save / load
#
# save() serialises the optimized program — injected demonstrations and
# (if MIPROv2) the rewritten instructions — to a JSON file.
# Model weights are NOT saved; this is prompt state only.
#
# To reload in production or in M20's multi-module pipeline:
#   module = TicketClassifier()
#   module.load("experiments/optimized_classifier.json")
#   pred = module(ticket="...")
# ---------------------------------------------------------------------------

SAVE_PATH = os.path.join(os.path.dirname(__file__), "optimized_classifier.json")


if __name__ == "__main__":
    configure_dspy()

    # Switch between run_bootstrap() and run_mipro() to compare.
    # Bootstrap is faster; MIPROv2 searches instruction + demo space jointly.
    optimized = run_mipro()

    print("\nEvaluating optimized module on devset...")
    optimized_score = evaluate(optimized)

    print(f"\nBaseline  (M17):  {M17_BASELINE:.2f}%")
    print(f"Optimized (M18):  {optimized_score:.2f}%")
    print(f"Delta:            {optimized_score - M17_BASELINE:+.2f}%")

    optimized.save(SAVE_PATH)
    print(f"\nSaved to {SAVE_PATH}")
    print("Open that file — read the demonstrations DSPy chose to inject.")
