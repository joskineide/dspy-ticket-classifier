"""
Confusion Analysis

Runs the classifier against DEVSET and shows which labels are being confused
with which. Use this to diagnose why the optimizer isn't improving the score —
if one label is systematically predicted as another, the problem is semantic
(the model can't distinguish them) not prompt-related (optimizer can't fix it).

HOW TO RUN (from the project root, with venv active):
    python experiments/confusion_analysis.py
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from m17_baseline import DEVSET
from app.services.classifier import configure_dspy, TicketClassifier


if __name__ == "__main__":
    configure_dspy()
    module = TicketClassifier()

    confusion: dict[str, list[str]] = defaultdict(list)
    correct = 0
    total = 0

    print(f"Running classifier on {len(DEVSET)} devset examples...\n")

    for ex in DEVSET:
        total += 1
        expected = ex.categories[0]
        try:
            pred = module(ticket=ex.ticket)
            predicted = pred.categories[0] if pred.categories else "none"
        except Exception:
            predicted = "parse_error"

        if expected == predicted:
            correct += 1
        else:
            confusion[expected].append(predicted)

    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)\n")

    if not confusion:
        print("No misclassifications found.")
    else:
        print("Misclassifications by true label:")
        print(f"{'True label':<22} {'Misses':>6}  Predicted as")
        print("-" * 60)
        for true_label, wrong_preds in sorted(confusion.items(), key=lambda x: -len(x[1])):
            counts: dict[str, int] = defaultdict(int)
            for p in wrong_preds:
                counts[p] += 1
            breakdown = ", ".join(f"{label} ({n}x)" for label, n in sorted(counts.items(), key=lambda x: -x[1]))
            print(f"{true_label:<22} {len(wrong_preds):>6}  → {breakdown}")

        print("\nWorst confused pairs (true → predicted):")
        pairs: dict[tuple[str, str], int] = defaultdict(int)
        for true_label, preds in confusion.items():
            for p in preds:
                pairs[(true_label, p)] += 1
        for (true_label, pred_label), count in sorted(pairs.items(), key=lambda x: -x[1])[:5]:
            print(f"  {true_label:<22} → {pred_label:<22} ({count}x)")
