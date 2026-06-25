"""
Milestone 19 — Constraint Enforcement

OBJECTIVE:
    Enforce output constraints so the model self-corrects on bad outputs
    instead of propagating them to the router.

HOW TO RUN (from the project root, with venv active):
    python experiments/m19_constraints.py

THE MODERN APPROACH — dspy.Refine:
    dspy.Assert and dspy.Suggest are legacy constructs deprecated in DSPy 3.x.
    The modern replacement is dspy.Refine, which acts as a best-of-N wrapper:

      dspy.Refine(module, N=3, reward_fn=fn, threshold=1.0)

    On each attempt it runs the module at temperature=1.0 (for variation) and
    scores the output with reward_fn. If the score is below threshold it calls
    an internal dspy.Predict(OfferFeedback) to generate targeted advice, which
    is injected as a hint into the next attempt automatically. It returns either
    the first prediction that meets the threshold or the best one seen so far.

    The reward function signature is (inputs: dict, pred: Prediction) -> float:
      inputs — the keyword arguments that were passed to the module
      pred   — the Prediction the module returned
      return — a scalar score; 1.0 = fully valid, 0.0 = fully invalid

    This is different from the metric functions used in Evaluate/Bootstrap, which
    take (example, pred). The reward_fn receives live call inputs, not dataset rows.

WHAT TO OBSERVE:
    - A well-formed ticket classifies correctly on the first attempt (reward = 1.0,
      Refine stops immediately without using the remaining N-1 slots).
    - Call dspy.inspect_history(n=6) after the test to see the classify prompt.
      If Refine needed to retry, you will see the hint injected by OfferFeedback
      in the subsequent attempt.

PRODUCTION NOTES (M19):
    Monitor the retry rate as a dashboard metric. Export it from the traces
    written in M27 (check whether a trace's "classify" span was reached more
    than once, or add a retry_count field to span metadata).

    A sustained high retry rate signals that the Signature boundary conditions
    are poorly specified — the model frequently produces outputs the reward_fn
    rejects. Raising N to 5 masks the symptom; fixing the Signature docstring
    or the reward_fn threshold addresses the cause.

    Each retry adds one full LLM call to the request latency. At N=3 and a
    25% retry rate, the average classify step costs 1.25 LLM calls instead of 1.
    That multiplier compounds with the reply step's own latency.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dspy

from app.services.classifier import configure_dspy, TicketClassifier


def test_valid_ticket() -> None:
    """A normal ticket — reward_fn should return 1.0 on the first attempt."""
    classifier = TicketClassifier()
    pred = classifier(ticket="I ate a tasty fish yesterday.")
    print(f"Valid ticket → {pred.categories}")
    assert len(pred.categories) == 1, "Expected exactly one label"
    print("Constraint satisfied on first attempt.\n")


if __name__ == "__main__":
    configure_dspy()

    print("=== M19 Constraint Enforcement ===\n")

    test_valid_ticket()

    # n=6 gives enough headroom to see all attempts if Refine retried (N=3
    # means up to 3 classify calls, each producing one history entry).
    dspy.inspect_history(n=6)
