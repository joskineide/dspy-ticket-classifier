"""
Milestone 21 — RAG Foundation: Document Store & Retrieval

OBJECTIVE:
    Build a semantic retriever that, given a new support ticket, finds the
    most similar past resolutions from the knowledge base.

HOW TO RUN (from the project root, with venv active):
    python experiments/m21_retriever.py

WHAT TO OBSERVE:
    - The cosine similarity scores printed alongside each result.
      A score > 0.85 is a strong semantic match; < 0.60 is weak.
    - How the same query returns different results than keyword search would:
      "app freezes on save" retrieves crash-related resolutions even if
      the word "crash" doesn't appear in the query.
    - What happens when K is too large: the lower-ranked results become
      increasingly irrelevant. There is a point where adding more context
      hurts rather than helps the reply generator.
    - Test 4 (out-of-domain query): watch the scores drop sharply compared
      to tests 1-3. This is the retriever telling you it has no good match —
      the model will have to generate without grounded context.
      NOTE: the query must be genuinely outside the KB's domain. Navigation
      and settings queries now have KB coverage; account cancellation and
      mobile apps do not — those are the clean out-of-domain signals.

BEFORE RUNNING:
    Make sure nomic-embed-text is pulled in Ollama:
        ollama pull nomic-embed-text
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.retriever import Retriever


def print_results(query: str, results: list[tuple[str, float]]) -> None:
    print(f"Query: {query!r}")
    for i, (resolution, score) in enumerate(results, 1):
        print(f"  [{i}] score={score:.3f}  {resolution[:110]}{'...' if len(resolution) > 110 else ''}")
    print()


if __name__ == "__main__":
    print("Building retrieval index...")
    retriever = Retriever()
    print("Index ready.\n")

    # --- Test 1: bug query ---
    print_results(
        "The export button crashes every time I click it.",
        retriever.retrieve_with_scores("The export button crashes every time I click it.", k=3),
    )

    # --- Test 2: feature query ---
    print_results(
        "I would love to schedule automatic weekly reports.",
        retriever.retrieve_with_scores("I would love to schedule automatic weekly reports.", k=3),
    )

    # --- Test 3: feedback query ---
    print_results(
        "Your pricing feels expensive compared to last year.",
        retriever.retrieve_with_scores("Your pricing feels expensive compared to last year.", k=3),
    )

    # --- Test 4: out-of-domain query — observe score drop ---
    # The KB has no resolutions about account cancellation or mobile apps.
    # All three returned results will be low-confidence matches. Compare the
    # scores here against tests 1-3 to see the retriever signal clearly
    # distinguish "good match" from "best available but still weak."
    # (Previously used "I cannot find the settings page" — that query is now
    # in-domain because M28 added Settings/Appearance/Theme KB entries.)
    print_results(
        "How do I cancel my subscription and get a refund for unused months?",
        retriever.retrieve_with_scores("How do I cancel my subscription and get a refund for unused months?", k=3),
    )
