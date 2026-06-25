"""
Milestone 28 — Two-Retriever Architecture: Feature KB

OBJECTIVE:
    Add a product feature knowledge base (feature_kb.json) that the classifier
    retrieves from BEFORE assigning a label. This gives the classifier grounded
    evidence about what features actually exist, resolving the wrong-premise
    classification failures identified in M25.5.

HOW TO RUN (from the project root, with venv active):
    python experiments/m28_feature_kb.py

THE CORE INSIGHT — retrieve at the step that needs the information:
    M22 retrieves past resolutions BEFORE the reply step — so the reply is grounded.
    M28 retrieves product features BEFORE the classify step — so the label is grounded.
    The pattern is the same; the KB and the step are different.

    Without feature context, "I can't believe there's no dark mode" is ambiguous:
    the model has no way to know whether dark mode exists, so it may classify it
    as 'feature' (genuine request) instead of 'feedback' (wrong premise).
    With feature context, the model sees the dark mode entry before committing
    to a label and can recognise the wrong premise directly.

WHAT THIS EXPERIMENT DOES:
    Runs the three wrong-premise tickets from the golden dataset through the pipeline
    twice — once with the feature retriever active (M28 path) and once bypassing it
    (M22 path, feature_context="") — and prints the label each time. A correct M28
    run should label all three as 'feedback'; a correct M22 run labels them as 'feature'.

WRONG-PREMISE TICKETS IN THE GOLDEN DATASET:
    [32] "Your app does not even have a mobile version."        → expected: feedback
    [33] "I cannot believe you still do not have dark mode."   → expected: feedback
    [34] "There is no way to delete multiple items at once."   → expected: feedback

WHAT TO LOOK FOR:
    - All three should flip from 'feature' (M22) to 'feedback' (M28) if the
      feature retriever retrieves the right entry and the classifier uses it.
    - If a ticket still classifies as 'feature' with context, read the retrieved
      features: the wrong entry may have been retrieved (low similarity) or the
      Signature docstring may need a stronger instruction to use the context.
    - Check the retrieve_features span in the trace to see which KB entries were
      pulled and whether they actually matched the ticket semantically.

CRITICAL — set lm_mode=direct in .env for this experiment.

FINDINGS (run 2026-06-23):
    All three tickets classified as 'feedback' on BOTH paths (M22 and M28).

    This reveals an important nuance: the M26 Signature fix (tightening the
    'feature' definition and explicitly including wrong-premise complaints under
    'feedback') was already sufficient to correct these three cases without
    feature KB grounding. M28 still adds value:

    1. Robustness — subtly worded tickets that don't obviously signal a
       wrong-premise assumption would still fool the Signature-only path.
       Feature KB grounding gives the model evidence, not just instructions.

    2. Retrieval quality confirmed — the correct KB entry was the top hit in
       all three cases (Mobile App, Dark Mode, Bulk Delete), demonstrating
       that nomic-embed-text maps complaint-style language to the right
       feature descriptions reliably.

    3. Layered defence — Signature instructions handle clear cases; KB
       grounding handles ambiguous ones. Having both is stronger than
       either alone.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.classifier import configure_dspy, TicketClassifier, ClassifyTicket
from app.services.retriever import FeatureRetriever
from app.services.pipeline import TicketPipeline, initialize_retriever

import dspy

WRONG_PREMISE_TICKETS = [
    "Your app does not even have a mobile version. It is desktop only and that is a dealbreaker for me.",
    "I cannot believe you still do not have dark mode in 2024. It is the most basic feature and it is missing.",
    "There is no way to delete multiple items at once. I have been removing them one by one for weeks and it is exhausting.",
]


def classify_without_features(classifier: TicketClassifier, ticket: str) -> str:
    """M22 path — classify with no feature context (empty string)."""
    pred = classifier(ticket=ticket)
    return pred.categories[0] if pred.categories else "unknown"


def classify_with_features(
    classifier: TicketClassifier,
    feature_retriever: FeatureRetriever,
    ticket: str,
) -> tuple[str, list[str]]:
    """M28 path — retrieve relevant features first, then classify with context."""
    features = feature_retriever.retrieve(ticket, k=2)
    feature_context = "\n\n".join(features)
    pred = classifier(ticket=ticket, feature_context=feature_context)
    label = pred.categories[0] if pred.categories else "unknown"
    return label, features


def main() -> None:
    configure_dspy()

    feature_retriever = FeatureRetriever()
    classifier = TicketClassifier()
    for ticket in WRONG_PREMISE_TICKETS:
        label_no_features = classify_without_features(classifier, ticket)
        label_with_features, features = classify_with_features(classifier, feature_retriever, ticket)
        print(f"Ticket: {ticket[:70]}")
        print(f"M22 (no context): {label_no_features}")
        print(f"M28 (with context): {label_with_features}")
        feature_names = [f.splitlines()[0].removeprefix("Feature: ") for f in features]
        print(f"Retrieved: {', '.join(feature_names)}")
        print(f"{'✔' if label_with_features == 'feedback' else '✘'}\n")


if __name__ == "__main__":
    main()
