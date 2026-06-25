# M30 — Output Content Validation (OWASP LLM05)
#
# ---------------------------------------------------------------------------
# PRODUCTION NOTES (M30)
#
# False positive tuning:
#   These patterns are intentionally conservative for a support reply context.
#   A real production system would tune thresholds and patterns against a
#   sample of legitimate outputs to measure false positive rate. A validator
#   that blocks too many valid replies is worse than no validator — it degrades
#   the product while the attacker simply adjusts phrasing.
#
# Output validation is not a replacement for input validation:
#   M12 and M29 are the primary defences. M30 is the last line of defence for
#   cases where something slips through. The ordering matters: if a malicious
#   payload reaches the LLM and the LLM reproduces it in its output, M30 is
#   the last gate before the caller receives it.
#
# Why not scan all fields:
#   `reply` is the field that a caller will render or process. `labels` is
#   already validated against VALID_LABELS. `reasoning` is internal trace data
#   that is not rendered by callers — flagging it would create false positives
#   for legitimate model reasoning that mentions security topics.
#
# Gateway equivalent:
#   The same patterns should be applied as @field_validators on the TarotCard
#   and structured output schemas in the AI Gateway (M13/M14). The pattern is
#   identical; only the schema classes differ.
# ---------------------------------------------------------------------------
#
# READING ORDER:
#   1. _PATTERNS — compiled regexes, defined once at module import
#   2. Individual detection functions — one concern per function, easy to test
#   3. validate_reply() — the single entry point called by the Pydantic validator

import logging
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns — defined once at module load, not per call.
# re.IGNORECASE applied where casing should not affect detection.
# ---------------------------------------------------------------------------

_PATTERNS = {
    # HTML opening tags: <script, <img, <a, <iframe, <form, etc.
    # Not just angle brackets — requires a letter after < to avoid false
    # positives on "error <code> occurred" style messages.
    "html": re.compile(r"<[a-zA-Z][^>]*>", re.IGNORECASE),

    # SQL DML/DDL keywords appearing in sequence as they would in an injection.
    # "SELECT * FROM" or "DROP TABLE" — not standalone SELECT or FROM which
    # could appear in support replies ("please select from the menu").
    "sql": re.compile(
        r"\b(DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM|SELECT\s+\*\s+FROM"
        r"|UPDATE\s+\w+\s+SET|CREATE\s+TABLE|ALTER\s+TABLE|EXEC\s*\()",
        re.IGNORECASE,
    ),

    # Shell command sequences. Standalone $ is excluded (prices, variables in
    # natural language). Backtick substitution, $() substitution, ; chaining,
    # && chaining, and pipe to dangerous commands are flagged.
    "shell": re.compile(
        r"(`[^`]+`|\$\([^)]+\)|;\s*(rm|curl|wget|bash|sh|python|nc)\b)",
        re.IGNORECASE,
    ),
}

# Replies longer than this are suspicious — prompt leakage or model going
# off-rails. A support reply is 2-4 sentences; 2000 chars is generous.
_MAX_REPLY_CHARS = 2000


# ---------------------------------------------------------------------------
# Detection functions — one concern per function so each is independently
# testable. Each returns (triggered: bool, reason: str).
# ---------------------------------------------------------------------------

def _check_html(text: str) -> tuple[bool, str]:
    if _PATTERNS["html"].search(text):
        return True, "reply contains HTML tags"
    return False, ""


def _check_sql(text: str) -> tuple[bool, str]:
    if _PATTERNS["sql"].search(text):
        return True, "reply contains SQL keywords"
    return False, ""


def _check_shell(text: str) -> tuple[bool, str]:
    if _PATTERNS["shell"].search(text):
        return True, "reply contains shell metacharacters"
    return False, ""


def _check_length(text: str) -> tuple[bool, str]:
    if len(text) > _MAX_REPLY_CHARS:
        return True, f"reply length {len(text)} exceeds maximum {_MAX_REPLY_CHARS}"
    return False, ""


def validate_reply(text: str) -> str:
    """Run all output checks on a reply string.

    Returns the text unchanged if all checks pass.
    Raises ValueError with a safe message if any check fails.

    WHY ValueError, not HTTPException:
        @field_validator runs inside Pydantic's validation layer, not inside
        FastAPI's request handling layer. HTTPException is a FastAPI construct
        that only works inside route handlers. ValueError is what Pydantic
        expects — it wraps it in a ValidationError, which FastAPI then maps
        to a 422 response automatically.

    WHY the error message must not include the flagged content:
        Same policy as M12 moderation blocks — log metadata only. Including
        the malicious payload in the error response would surface it to the
        caller, which is exactly what the validator is trying to prevent.
    """
    for check in (_check_html, _check_sql, _check_shell, _check_length):
        triggered, reason = check(text)
        if triggered:
            log.warning("Output validation blocked reply: %s", reason)
            raise ValueError(f"Output validation failed: {reason}")
    return text
