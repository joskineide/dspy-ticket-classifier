"""
Synthetic Dataset Generator

Generates realistic customer support tickets using the gateway LM, saves them
to synthetic_tickets.json. Run this once — the baseline and optimizer scripts
load from the saved file rather than re-generating on every run.

HOW TO RUN (from the project root, with venv active):
    python experiments/generate_dataset.py

LABELS (3 classes, single-label):
    bug      — something is broken or behaving unexpectedly
    feature  — user wants new functionality that doesn't exist yet
    feedback — praise, complaints, rants, or noise that needs no action

Collapsing positive_feedback / negative_feedback / troll into one "feedback"
bucket removes the ambiguous overlap that was confusing the model. The key
signal is now bug vs feature vs everything-else — a much cleaner decision.
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dspy

from app.services.classifier import configure_dspy

SAVE_PATH = os.path.join(os.path.dirname(__file__), "synthetic_tickets.json")

# 80 examples per class → 240 total, shuffled before saving.
# 240 examples gives MIPROv2 a proper internal valset (~50 examples) while
# leaving ~120 for TRAINSET and ~70 for DEVSET after the 70/30 split.
CATEGORIES = {
    "bug":      80,
    "feature":  80,
    "feedback": 80,
}

# Style variation prevents the generator from producing near-identical tickets.
# Without this, all "bug" examples sound like the same person retyping the same
# complaint — the optimizer has nothing to learn from.
STYLES = [
    "brief and frustrated",
    "brief and polite",
    "detailed and technical, with steps to reproduce",
    "detailed and emotional, expressing real impact",
    "casual and informal",
    "formal and professional",
    "very short (one sentence only)",
    "sarcastic",
    "confused, asking questions rather than stating a problem",
    "angry and demanding immediate action",
    "positive and appreciative tone despite the issue",
    "comparing unfavourably to a competitor",
]

# Per-category guidance injected into the generation prompt so the model
# understands what "feedback" covers (it's intentionally broad).
CATEGORY_HINTS = {
    "bug":     "The user is reporting something that is broken, crashing, returning wrong results, or behaving in an unexpected way.",
    "feature": "The user is requesting new functionality, an improvement, or a capability that does not currently exist.",
    "feedback": "The user is expressing an opinion — praise, disappointment, frustration, complaints about pricing or support, or general noise. There is no specific bug or feature request.",
}


class GenerateTicket(dspy.Signature):
    """Generate a single realistic customer support ticket matching the given category, style, and category hint."""
    category: str = dspy.InputField(desc="The target label: bug, feature, or feedback")
    category_hint: str = dspy.InputField(desc="What this category means — use this to stay on-label")
    style: str = dspy.InputField(desc="Writing style for the ticket")
    ticket: str = dspy.OutputField(desc="A realistic customer support message, 1–4 sentences, that clearly fits the category and style. No meta-commentary, just the ticket text itself.")


class TicketGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict(GenerateTicket)

    def forward(self, category: str, style: str) -> dspy.Prediction:
        return self.generate(
            category=category,
            category_hint=CATEGORY_HINTS[category],
            style=style,
        )


def generate_all() -> list[dict]:
    generator = TicketGenerator()
    examples = []
    total = sum(CATEGORIES.values())
    generated = 0

    for category, count in CATEGORIES.items():
        print(f"\nGenerating {count} tickets for [{category}]...")
        for _ in range(count):
            style = random.choice(STYLES)
            try:
                pred = generator(category=category, style=style)
                ticket_text = pred.ticket.strip()
                if ticket_text:
                    examples.append({"ticket": ticket_text, "categories": [category]})
            except Exception as e:
                print(f"  Skipped (generation failed): {e}")

            generated += 1
            if generated % 20 == 0:
                print(f"  {generated}/{total} generated...")

    return examples


if __name__ == "__main__":
    configure_dspy()

    print(f"Generating {sum(CATEGORIES.values())} synthetic tickets (3 classes)...")
    print(f"Output: {SAVE_PATH}\n")

    examples = generate_all()
    random.shuffle(examples)

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(examples)} tickets saved to {SAVE_PATH}")
    print("Spot-check the file before running m17_baseline.py — look for tickets")
    print("where the label doesn't obviously match the text.")
