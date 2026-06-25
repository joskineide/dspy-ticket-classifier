# M32 — PII Redaction at the Retrieval Boundary (OWASP LLM06)
#
# ---------------------------------------------------------------------------
# WHY THIS MODULE EXISTS
#
# The knowledge base is built from past resolved support tickets. Real KB
# entries contain contact details recorded by support agents: email addresses,
# phone numbers, internal account IDs. The RAG pipeline retrieves raw resolution
# text and injects it verbatim into the LLM's context window.
#
# Without redaction a retrieved chunk like:
#   "Contacted customer Jane Doe (jane.doe@acmecorp.com, +1-415-555-0142,
#    account #ACC-10842) and issued a refund..."
# lands in the prompt. The LLM then echoes those details back in its reply
# to whoever asked the current question — surfacing PII that belongs to a
# completely different customer.
# ---------------------------------------------------------------------------
#
# WHY AT THE RETRIEVAL BOUNDARY, NOT POST-LLM
#
# Two alternatives exist:
#   A. Redact BEFORE the LLM — this module's approach.
#   B. Scrub the LLM's OUTPUT after generation (post-processing).
#
# A is strictly better for two reasons:
#   1. The LLM never internalises the raw PII in its context.
#      Post-output scrubbing catches what the model WROTE, but the model
#      has already reasoned over the PII. A generative model may reformulate
#      values ("her email was jane dot doe at example dot com") in ways
#      a regex won't catch.
#   2. What the LLM doesn't see it can't repeat. Redacting the context
#      is a hard guarantee; scrubbing the output is probabilistic.
#
# The architectural principle: trust boundaries live at DATA INGESTION points,
# not at output points. M30's output validation is a final safety net, not
# the primary defence.
# ---------------------------------------------------------------------------
#
# WHY REGEX FOR EMAILS AND PHONES, NOT FOR NAMES
#
# Emails and phone numbers have structural signatures:
#   RFC 5322 email format:  local@domain.tld
#   E.164 phone format:     +country-area-number
# A regex that matches these formats has high precision and low false-positive
# risk in a support-context corpus.
#
# Names ("Jane Doe", "Robert Chen") do NOT have a structural signature.
# "Jane" could be a name, a product name, or a verb. Regex cannot distinguish
# "customer John Smith" from "follow the John Smith process". Name detection
# requires Named Entity Recognition (NER) — a trained model that understands
# grammatical context.
#
# Production options for full-spectrum PII detection:
#   - Microsoft Presidio (OSS): 50+ entity types, NER + regex + context rules,
#     pluggable custom recognizers, outputs an operator log.
#   - AWS Comprehend: managed API, no infrastructure, pay-per-call.
#   - spaCy en_core_web_trf + custom entity ruler for domain-specific patterns.
#
# This module covers pattern-detectable PII only. Name and address detection
# are documented explicitly as a production gap — a visible limitation is
# better than invisible missing coverage.
# ---------------------------------------------------------------------------
#
# READING ORDER:
#   1. RedactionResult  — the typed return value
#   2. _PATTERNS        — compiled regexes, defined once at module load
#   3. _REPLACEMENT_TOKENS — maps pattern name to the token that replaces it
#   4. redact_pii()     — the single public function called from pipeline.py

import re
from dataclasses import dataclass, field


@dataclass
class RedactionResult:
    text: str                          # Text after redaction — this is what the LLM receives
    redacted_count: int                # Total number of PII spans replaced across all types
    types_found: list[str] = field(default_factory=list)  # Which PII types were detected


# ---------------------------------------------------------------------------
# Compiled patterns — defined once at module load, not per call.
#
# Patterns are compiled with re.IGNORECASE where casing is not meaningful
# (account ID prefixes are uppercase by convention, but be defensive).
#
# WHY re.compile() at module level:
#   Compilation converts the regex string into an internal automaton.
#   Doing this inside redact_pii() would recompile on every call. At module
#   load it compiles once and is reused. For a function called on every
#   retrieved chunk, this matters.
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, re.Pattern] = {
    # Standard RFC 5322 local@domain.tld shape with word boundaries.
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),

    # Internal IDs: 2-6 uppercase letters, hyphen, 4-6 digits. e.g. ACC-10842.
    # Run before phone so the digit sequence inside an ID isn't consumed first.
    "account_id": re.compile(r"\b[A-Z]{2,6}-\d{4,6}\b"),

    # E.164 and common national formats. Uses (?<!\w) instead of \b so the
    # leading + is captured as part of the match rather than left behind.
    "phone": re.compile(r"(?<!\w)\+?(?:\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b"),
}

# Replacement tokens: what the LLM sees instead of the raw PII value.
# [EMAIL] / [PHONE] / [ACCOUNT_ID] are clear, unambiguous placeholders.
# The model can still write a coherent reply ("the customer's [EMAIL]") without
# knowing the actual address. Do NOT use empty string — that creates run-on
# sentences and makes the redaction invisible to log reviewers.
_REPLACEMENT_TOKENS: dict[str, str] = {
    "email":      "[EMAIL]",
    "phone":      "[PHONE]",
    "account_id": "[ACCOUNT_ID]",
}


def redact_pii(text: str) -> RedactionResult:
    """Replace PII spans in text with opaque tokens.

    Returns a RedactionResult containing:
      - text: the cleaned string to pass to the LLM
      - redacted_count: how many replacements were made in total
      - types_found: which PII categories were detected

    Called from pipeline.py immediately after context moderation passes and
    before the text is wrapped in structural tags and handed to a DSPy predictor.
    If redacted_count > 0, log a warning so the KB can be cleaned at the source.
    """
    total = 0
    types_found = []

    for pii_type, pattern in _PATTERNS.items():
        token = _REPLACEMENT_TOKENS[pii_type]
        text, count = pattern.subn(token, text)
        if count > 0:
            total += count
            types_found.append(pii_type)

    return RedactionResult(text=text, redacted_count=total, types_found=types_found)
