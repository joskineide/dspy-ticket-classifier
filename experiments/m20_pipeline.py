"""
Milestone 20 — Multi-Module Pipelines

OBJECTIVE:
    Chain ClassifyTicket → SuggestReply inside a single dspy.Module so the
    optimizer can improve both predictors jointly from one compile() call.

HOW TO RUN (from the project root, with venv active):
    python experiments/m20_pipeline.py

WHAT TO OBSERVE:
    - dspy.inspect_history(n=4) after a pipeline call shows TWO prompts:
      one for classify, one for reply — both built by DSPy automatically.
    - After compile(), open optimized_pipeline.json and look at the
      demonstrations injected into EACH predictor independently.
    - The optimizer improved the full pipeline from a single metric,
      without you writing prompt engineering for either step.

THE END-TO-END METRIC CHALLENGE:
    For classification (M17/M18), the metric was easy: is the label correct?
    For reply generation, there is no ground truth — you can't say "the
    correct reply is exactly these words."

    Two practical options:
    a) Evaluate only the classification leg — metric = Jaccard(true_label, pred_label).
       Simple and honest, but tells the optimizer nothing about reply quality.
       The reply predictor still gets demonstrations injected based on when
       the classification was correct, which indirectly improves its examples.

    b) LLM-as-judge — ask a second LM to score the reply on a 0–1 scale given
       the ticket and category. More informative, but slower and adds a second
       model dependency. The judge score must be fast and stable enough for the
       optimizer to use across dozens of evaluations.

    Option (a) was used here and produced 100% after Bootstrap with only 3 traces.
    The 87.5% baseline (vs 91.67% in M18) reflects the added failure mode of a
    two-step chain: a misclassification on step 1 propagates and scores 0 on step 2.

RESULTS:
    Dataset:   synthetic customer support, 168 train / 72 dev
    Metric:    Jaccard on classification leg only (option a above)
    Baseline:  87.50%
    Bootstrap: 100.00%  (only 3 traces needed — found in first 3 examples)
    Delta:     +12.50%

KEY FINDING — composed modules are cheap to optimise:
    Bootstrap found 100% accuracy after just 3 successful traces — one per label.
    The demonstrations in optimized_pipeline.json show the optimizer injected
    examples into EACH predictor separately (look for "classify" and "reply"
    sections). One compile() call, one metric, two predictors improved.

CRITICAL — set lm_mode=direct in .env for experiment runs.
    Same reason as M17/M18: the gateway semantic cache collapses scores.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dspy

from app.services.classifier import configure_dspy
from app.services.pipeline import TicketPipeline

from m17_baseline import TRAINSET, DEVSET, jaccard_metric

M20_BASELINE = 87.50


# ---------------------------------------------------------------------------
# 1. Smoke test — run one ticket through the pipeline
#
# Run this first to confirm the chain works end-to-end.
# inspect_history(n=4) shows both prompts: classify then reply.
# Note they share no state — DSPy builds each from its own Signature.
# ---------------------------------------------------------------------------

def smoke_test() -> None:
    pipeline = TicketPipeline()
    pred = pipeline(ticket="The export button crashes every time I click it.")

    print(f"Label:     {pred.categories}")
    print(f"Reasoning: {getattr(pred, 'reasoning', None)}")
    print(f"Reply:     {pred.reply}")
    print()

    dspy.inspect_history(n=4)


# ---------------------------------------------------------------------------
# 2. End-to-end metric
#
# Scores only the classification leg. The reply predictor still benefits:
# Bootstrap injects demonstrations from the examples where the full pipeline
# succeeded, so the reply step sees examples paired with correct labels.
# ---------------------------------------------------------------------------

def pipeline_metric(example, pred, trace=None) -> float:
    return jaccard_metric(example, pred, trace)


# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------

def evaluate(module: dspy.Module) -> float:
    evaluate_fn = dspy.Evaluate(
        devset=DEVSET,
        metric=pipeline_metric,
        num_threads=1,
        display_progress=True,
        display_table=5,
    )
    return float(evaluate_fn(module))


# ---------------------------------------------------------------------------
# 4. Optimize with BootstrapFewShot
#
# compile() recurses into TicketPipeline and finds both self.classify and
# self.reply. It injects demonstrations into each independently, choosing
# examples where the full pipeline metric was highest.
#
# One metric. One compile() call. Both predictors improved.
# ---------------------------------------------------------------------------

SAVE_PATH = os.path.join(os.path.dirname(__file__), "optimized_pipeline.json")


def run_bootstrap() -> dspy.Module:
    optimizer = dspy.BootstrapFewShot(
        metric=pipeline_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
    )
    optimized = optimizer.compile(student=TicketPipeline(), trainset=TRAINSET)
    return optimized


if __name__ == "__main__":
    configure_dspy()

    print("=== M20 Multi-Module Pipeline ===\n")

    print("--- Smoke test ---")
    smoke_test()

    print("--- Baseline ---")
    baseline = evaluate(TicketPipeline())
    print(f"Baseline: {baseline:.2f}%\n")

    print("--- Optimizing ---")
    optimized = run_bootstrap()
    optimized_score = evaluate(optimized)
    print(f"\nBaseline:  {baseline:.2f}%")
    print(f"Optimized: {optimized_score:.2f}%")
    print(f"Delta:     {optimized_score - baseline:+.2f}%")
    optimized.save(SAVE_PATH)
    print(f"\nSaved to {SAVE_PATH}")
    print("Open it — find the demonstrations for EACH predictor separately.")
