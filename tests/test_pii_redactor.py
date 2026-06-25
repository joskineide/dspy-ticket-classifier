# M32 — Unit tests for pii_redactor.py
#
# Pure unit tests — no Ollama, no DSPy, no network calls.
# redact_pii() is a deterministic regex function; its behaviour is fully
# verifiable without any external dependencies.
#
# WHY UNIT TESTS HERE INSTEAD OF INTEGRATION TESTS (contrast with test_content_guard.py):
#   content_guard.py wraps an LLM call — mocking httpx.post() would prove the
#   plumbing but not the model's detection quality, so integration tests are
#   the right call there. pii_redactor.py is pure regex: the only thing to
#   verify is that the patterns match what they should and nothing else.
#   A unit test is faster, deterministic, and sufficient.
#
# Run with:
#   pytest tests/test_pii_redactor.py -v

from app.services.pii_redactor import redact_pii, RedactionResult


EMAIL_SAMPLE = "Contact support at jane.doe@acmecorp.com for follow-up."
PHONE_SAMPLE = "Spoke with account owner at +1-312-555-8823 to confirm the issue."
ACCOUNT_ID_SAMPLE = "Workspace permissions for account #ACC-10842 were scoped incorrectly."
MIXED_SAMPLE = (
    "Contacted customer Jane Doe (r.chen@techstartup.io, +1-415-555-0142, "
    "account #CUST-00312) and issued a refund."
)
CLEAN_SAMPLE = (
    "Identified a null-pointer exception in the CSV exporter when the dataset "
    "contains empty date fields. Released a patch in v2.3.1."
)


def test_email_is_redacted():
    result = redact_pii(EMAIL_SAMPLE)
    assert "[EMAIL]" in result.text
    assert "jane.doe@acmecorp.com" not in result.text
    assert result.redacted_count >= 1
    assert "email" in result.types_found


def test_phone_is_redacted():
    result = redact_pii(PHONE_SAMPLE)
    assert "[PHONE]" in result.text
    assert "+1-312-555-8823" not in result.text
    assert result.redacted_count >= 1
    assert "phone" in result.types_found


def test_account_id_is_redacted():
    result = redact_pii(ACCOUNT_ID_SAMPLE)
    assert "[ACCOUNT_ID]" in result.text
    assert "ACC-10842" not in result.text
    assert result.redacted_count >= 1
    assert "account_id" in result.types_found


def test_clean_text_is_unchanged():
    # False-positive guard: version strings like "v2.3.1" must not match account_id.
    result = redact_pii(CLEAN_SAMPLE)
    assert result.text == CLEAN_SAMPLE
    assert result.redacted_count == 0
    assert result.types_found == []


def test_multiple_pii_types_in_one_string():
    # Verifies patterns compose correctly: tokens like [EMAIL] don't accidentally
    # trigger the phone or account_id pattern on a subsequent pass.
    result = redact_pii(MIXED_SAMPLE)
    assert result.redacted_count == 3
    assert "email" in result.types_found
    assert "phone" in result.types_found
    assert "account_id" in result.types_found
    assert "r.chen@techstartup.io" not in result.text
    assert "+1-415-555-0142" not in result.text
    assert "CUST-00312" not in result.text


def test_redacted_count_is_accurate():
    # Verifies subn() is used (not sub()) so each individual match increments
    # the counter, not just the presence of a PII type.
    two_emails_one_phone = (
        "Primary contact: alice@example.com, "
        "secondary: bob@example.org, phone +44-20-7946-0958."
    )
    result = redact_pii(two_emails_one_phone)
    assert result.redacted_count == 3


def test_return_type_is_redaction_result():
    # Guards against a refactor that accidentally returns a bare string.
    assert isinstance(redact_pii(CLEAN_SAMPLE), RedactionResult)
    assert isinstance(redact_pii(EMAIL_SAMPLE), RedactionResult)
