from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from saliencegate.domain import (
    EventPhase,
    EventType,
    NormalizedTraceEventDraft,
    PayloadDigestAlgorithm,
    TrustLabel,
    canonical_json,
    validate_normalized_trace_event_draft,
)
from saliencegate.security import (
    AmbiguousDigestModeError,
    AmbiguousFieldNameError,
    InstallationKey,
    RedactionPolicy,
    Redactor,
    SecretInFieldNameError,
    SyntheticDigestDisabledError,
    verify_redacted_event,
)
from saliencegate.security.digests import RedactedPayload, hmac_payload_digest

RUN_ID = UUID("00000000-0000-4000-8000-000000000001")
NOW = datetime(2026, 7, 11, 12, 30, tzinfo=UTC)
FULLWIDTH_ACCESS_TOKEN = "\uff21\uff23\uff23\uff25\uff33\uff33\uff3f\uff34\uff2f\uff2b\uff25\uff2e"


@pytest.mark.parametrize(
    ("raw", "forbidden", "preserved"),
    [
        (
            "Authorization: Bearer fixture_token_1234567890",
            "fixture_token_1234567890",
            "Bearer [REDACTED]",
        ),
        ("key=sk-proj-fixture1234567890abcdef", "sk-proj-fixture", "[REDACTED]"),
        ("token=sk-aaaaaaaaaaaaaaa-", "sk-aaaaaaaaaaaaaaa-", "[REDACTED]"),
        ("token ghp_fixture12345678901234567890", "ghp_fixture", "[REDACTED]"),
        ("google=AIzaFixture1234567890123456789012345", "AIzaFixture", "[REDACTED]"),
        ("slack=xoxb-1234567890-fixturetoken", "xoxb-1234567890", "[REDACTED]"),
        ("api_key = fixture-value-123456", "fixture-value-123456", "api_key"),
        ("api key: fixture-value-123456", "fixture-value-123456", "api key"),
        ('{"password": "fixture-value-123456"}', "fixture-value-123456", "password"),
        ('{"X-API-Key":"fixture-value-123456"}', "fixture-value-123456", "X-API-Key"),
        (
            "postgresql://alice:fixture-password@localhost/db",
            "fixture-password",
            "postgresql://alice:[REDACTED]@localhost/db",
        ),
        (
            "-----BEGIN PRIVATE KEY-----\nfixture-private-material\n-----END PRIVATE KEY-----",
            "fixture-private-material",
            "[REDACTED PRIVATE KEY]",
        ),
    ],
)
def test_common_inline_secret_formats_are_redacted(
    raw: str,
    forbidden: str,
    preserved: str,
) -> None:
    result = Redactor().redact_payload({"message": raw})
    encoded = canonical_json(result.payload.root).decode()

    assert forbidden not in encoded
    assert preserved in encoded
    assert result.findings


@pytest.mark.parametrize(
    ("raw", "forbidden", "expected"),
    [
        (
            "mongodb://alice:p@ss@[2001:db8::1]:27017/db",
            "p@ss",
            "mongodb://alice:[REDACTED]@[2001:db8::1]:27017/db",
        ),
        (
            "postgres://alice:p%40ss@localhost/db",
            "p%40ss",
            "postgres://alice:[REDACTED]@localhost/db",
        ),
        (
            "//alice:rawsecret@host/path",
            "rawsecret",
            "//alice:[REDACTED]@host/path",
        ),
        (
            "postgres://alice%3Arawsecret@host/db",
            "rawsecret",
            "postgres://[REDACTED]@host/db",
        ),
        (
            "lowercase bearer\u00a0fixture_token_1234567890",
            "fixture_token_1234567890",
            "Bearer [REDACTED]",
        ),
        (
            "Bearer\u200bfixture_token_1234567890",
            "fixture_token_1234567890",
            "Bearer [REDACTED]",
        ),
        (
            "Authorization: Bearer\r\n fixture_token_1234567890",
            "fixture_token_1234567890",
            "Bearer [REDACTED]",
        ),
        (
            "Bearer fixture_\u200btoken_1234567890",
            "token_1234567890",
            "Bearer [REDACTED]",
        ),
    ],
)
def test_uri_and_bearer_edge_cases(raw: str, forbidden: str, expected: str) -> None:
    result = Redactor().redact_payload({"message": raw})
    redacted = result.payload.root["message"]
    assert forbidden not in redacted
    assert expected in redacted


@pytest.mark.parametrize(
    "label",
    [
        "PRIVATE KEY",
        "RSA PRIVATE KEY",
        "EC PRIVATE KEY",
        "OPENSSH PRIVATE KEY",
        "ENCRYPTED PRIVATE KEY",
        "PGP PRIVATE KEY BLOCK",
    ],
)
def test_private_key_variants_are_fully_redacted(label: str) -> None:
    raw = f"-----BEGIN {label}-----\r\nfixture-material\r\n-----END {label}-----"
    result = Redactor().redact_payload({"message": raw})
    assert result.payload.root["message"] == "[REDACTED PRIVATE KEY]"


def test_truncated_private_key_is_redacted_conservatively() -> None:
    raw = "prefix -----BEGIN PRIVATE KEY-----\nfixture-material"
    result = Redactor().redact_payload({"message": raw})
    assert "fixture-material" not in result.payload.root["message"]


@pytest.mark.parametrize(
    "text",
    [
        "sk-short",
        "Bearer economy is a phrase, not a token",
        "https://alice@example.com/path",
        "-----BEGIN PUBLIC KEY-----\nfixture\n-----END PUBLIC KEY-----",
        "ordinary configuration value",
    ],
)
def test_near_matches_are_not_redacted(text: str) -> None:
    result = Redactor().redact_payload({"message": text})

    assert result.payload.root["message"] == text
    assert result.findings == ()


def test_redaction_preserves_safe_scalars_and_already_redacted_markers() -> None:
    payload = {
        "marker": "[REDACTED]",
        "url": "https://example.com/path",
        "count": 3,
        "ratio": 0.5,
        "enabled": True,
        "missing": None,
    }
    result = Redactor().redact_payload(payload)
    assert result.payload.root == payload
    assert result.findings == ()


def test_secret_named_fields_are_replaced_recursively() -> None:
    payload = {
        "safe": "unchanged",
        "password": "fixture-password",
        "nested": [
            {"apiKey": "fixture-api-key"},
            {FULLWIDTH_ACCESS_TOKEN: {"deep": "fixture-token"}},
        ],
    }
    result = Redactor().redact_payload(payload)

    assert result.payload.root == {
        "nested": (
            {"apiKey": "[REDACTED]"},
            {FULLWIDTH_ACCESS_TOKEN: "[REDACTED]"},
        ),
        "password": "[REDACTED]",
        "safe": "unchanged",
    }
    assert {finding.path for finding in result.findings} == {
        "/nested/0/apiKey",
        f"/nested/1/{FULLWIDTH_ACCESS_TOKEN}",
        "/password",
    }


def test_prefixed_secret_header_names_are_redacted() -> None:
    secret = "fixture-secret-123456"
    result = Redactor().redact_payload({"headers": {"X-API-Key": secret}})

    assert result.payload.root["headers"]["X-API-Key"] == "[REDACTED]"
    assert secret not in canonical_json(result.payload.root).decode()


@pytest.mark.parametrize(
    "field_name",
    [
        "pass\u200bword",
        "pass\u202eword",
        "pwd",
        "client.secret",
        "REFRESH TOKEN",
        "\uff30\uff21\uff33\uff33\uff37\uff2f\uff32\uff24",
    ],
)
def test_secret_field_matching_resists_unicode_and_separator_bypasses(
    field_name: str,
) -> None:
    result = Redactor().redact_payload({field_name: "fixture-secret"})
    assert result.payload.root[field_name] == "[REDACTED]"


@pytest.mark.parametrize(
    "field_name",
    [
        "p\u0430ssword",
        "pa\u0455\u0455word",
        "\u0455ecret",
        "pa\ua731\ua731word",
        "\ua731ecret",
    ],
)
def test_mixed_script_field_names_are_rejected_conservatively(field_name: str) -> None:
    with pytest.raises(AmbiguousFieldNameError):
        Redactor().redact_payload({field_name: "rawsecret"})


@pytest.mark.parametrize(
    "field_name",
    ["token_count", "password_policy", "secretary", "monkey", "authorization_status"],
)
def test_secret_field_near_matches_are_preserved(field_name: str) -> None:
    result = Redactor().redact_payload({field_name: "safe-value"})
    assert result.payload.root[field_name] == "safe-value"
    assert result.findings == ()


def test_non_confusable_international_field_names_are_preserved() -> None:
    result = Redactor().redact_payload({"日本語": "safe-value"})

    assert result.payload.root["日本語"] == "safe-value"
    assert result.findings == ()


def test_secret_material_in_a_mapping_key_is_rejected_without_echoing_it() -> None:
    secret = "fixture-key-secret"
    redactor = Redactor(literal_secrets=(secret,))

    with pytest.raises(SecretInFieldNameError) as error:
        redactor.redact_payload({f"prefix-{secret}": "value"})

    assert secret not in str(error.value)


def test_obfuscated_configured_literal_in_a_mapping_key_is_rejected() -> None:
    with pytest.raises(SecretInFieldNameError):
        Redactor(literal_secrets=("café-secret",)).redact_payload(
            {"cafe\u200b\u0301-secret": "value"}
        )


def test_custom_field_names_and_literal_secrets_are_supported() -> None:
    redactor = Redactor(
        literal_secrets=("project-codename",),
        structured_field_names=("vaultReference",),
    )
    result = redactor.redact_payload(
        {
            "note": "Use project-codename twice: project-codename.",
            "vaultReference": "secret/path",
        }
    )

    encoded = canonical_json(result.payload.root).decode()
    assert "project-codename" not in encoded
    assert "secret/path" not in encoded
    assert encoded.count("[REDACTED]") == 3


@pytest.mark.parametrize(
    "configuration",
    [
        {"literal_secrets": ("",)},
        {"literal_secrets": ("[REDACTED]",)},
        {"literal_secrets": ("ACT",)},
        {"structured_field_names": ("",)},
    ],
)
def test_invalid_redaction_configuration_is_rejected(
    configuration: dict[str, tuple[str, ...]],
) -> None:
    with pytest.raises(ValueError):
        Redactor(**configuration)


def test_redactor_repr_reports_counts_without_literal_values() -> None:
    secret = "fixture-secret"
    rendered = repr(Redactor(literal_secrets=(secret,), structured_field_names=("vault",)))
    assert secret not in rendered
    assert "literal_secrets=" in rendered


def test_redaction_policy_copies_inputs_and_hides_literal_values() -> None:
    secret = "fixture-policy-secret"
    literals = [secret]
    fields = ["vault"]
    policy = RedactionPolicy(
        literal_secrets=literals,
        structured_field_names=fields,
    )
    literals.clear()
    fields.clear()

    assert policy.literal_secrets == (secret,)
    assert policy.structured_field_names == ("vault",)
    assert secret not in repr(policy)
    assert "literal_secrets=1" in repr(policy)


def test_non_json_payload_value_is_rejected_without_echoing_contents() -> None:
    secret = "fixture-secret"
    with pytest.raises(ValueError) as error:
        Redactor().redact_payload({"value": {secret}})
    assert secret not in str(error.value)


@pytest.mark.parametrize(
    "raw",
    [
        r'password="abc\"rawsecret"',
        'password="rawsecret',
        r"secret='abc\'rawsecret'",
        "secret='rawsecret",
        'password="abc\nrawsecret"',
        'password="abc\nrawsecret',
    ],
)
def test_quoted_and_truncated_assigned_secrets_are_fully_redacted(raw: str) -> None:
    result = Redactor().redact_payload({"message": raw})
    redacted = result.payload.root["message"]
    assert "rawsecret" not in redacted
    assert redacted.endswith("[REDACTED]")


def test_configured_literals_cover_canonical_unicode_variants() -> None:
    composed = "café-secret"
    decomposed = "cafe\u0301-secret"
    result = Redactor(literal_secrets=(composed,)).redact_payload({"value": decomposed})

    assert decomposed not in canonical_json(result.payload.root).decode()


@pytest.mark.parametrize(
    "obfuscated",
    [
        "\uff50\uff52\uff4f\uff4a\uff45\uff43\uff54\uff0d"
        "\uff43\uff4f\uff44\uff45\uff4e\uff41\uff4d\uff45",
        "project-\u200bcodename",
        "cafe\u200b\u0301-secret",
    ],
)
def test_configured_literals_reject_compatibility_and_format_bypasses(
    obfuscated: str,
) -> None:
    configured = "café-secret" if "cafe" in obfuscated else "project-codename"
    result = Redactor(literal_secrets=(configured,)).redact_payload({"value": obfuscated})
    assert result.payload.root["value"] == "[REDACTED]"


@given(
    prefix=st.text(max_size=40),
    suffix=st.text(max_size=40),
    secret=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=4,
        max_size=24,
    ).filter(
        lambda value: all(value not in fixed for fixed in ("[REDACTED]", "nested", "message"))
    ),
)
def test_configured_secret_never_survives_recursive_redaction(
    prefix: str,
    suffix: str,
    secret: str,
) -> None:
    result = Redactor(literal_secrets=(secret,)).redact_payload(
        {"nested": [{"message": prefix + secret + suffix}]}
    )

    assert secret not in result.payload.root["nested"][0]["message"]


def test_redaction_is_idempotent() -> None:
    redactor = Redactor(literal_secrets=("fixture-secret",))
    first = redactor.redact_payload({"token": "fixture-secret"})
    second = redactor.redact_payload(first.payload.root)

    assert second.payload == first.payload
    assert second.findings == ()


def test_event_redaction_computes_hmac_only_after_masking() -> None:
    draft = NormalizedTraceEventDraft(
        run_id=RUN_ID,
        source_event_id="source-1",
        timestamp=NOW,
        event_type=EventType.OBSERVATION,
        phase=EventPhase.POST_ACTION,
        payload={"message": "credential=fixture-secret"},
        source_adapter="fixture",
        trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
    )
    key = InstallationKey(b"k" * 32)
    redactor = Redactor(literal_secrets=("fixture-secret",))
    result = redactor.redact_event(draft, key=key)

    assert result.event.payload_digest.algorithm is PayloadDigestAlgorithm.HMAC_SHA256
    assert "fixture-secret" not in canonical_json(result.event).decode()
    assert verify_redacted_event(result.event, redactor=redactor, key=key)

    tampered_values = result.event.model_dump(mode="python")
    tampered_values["payload_digest"]["value"] = "0" * 64
    tampered = type(result.event).model_validate(tampered_values)
    assert not verify_redacted_event(tampered, redactor=redactor, key=key)
    assert draft.payload["message"] == "credential=fixture-secret"
    assert result.event.run_id == draft.run_id
    assert result.event.source_event_id == draft.source_event_id
    assert result.event.timestamp == draft.timestamp
    assert result.event.event_type is draft.event_type
    assert result.event.phase is draft.phase
    assert result.event.parent_ids == draft.parent_ids
    assert result.event.source_adapter == draft.source_adapter
    assert result.event.trust_label is draft.trust_label


def test_forged_redacted_draft_with_valid_hmac_is_rejected() -> None:
    draft = NormalizedTraceEventDraft(
        run_id=RUN_ID,
        source_event_id="source-1",
        timestamp=NOW,
        event_type=EventType.OBSERVATION,
        phase=EventPhase.POST_ACTION,
        payload={"password": "rawsecret"},
        source_adapter="fixture",
        trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
    )
    key = InstallationKey(b"k" * 32)
    raw_payload = RedactedPayload(root=draft.payload)
    values = draft.model_dump(mode="python")
    values.update(
        record_type="redacted_trace_event_draft",
        payload_digest=hmac_payload_digest(raw_payload, key),
    )
    forged = type(Redactor().redact_event(draft, key=key).event).model_validate(values)

    assert not verify_redacted_event(forged, redactor=Redactor(), key=key)


def test_normalized_draft_repr_and_validation_errors_hide_raw_payload() -> None:
    secret = "fixture-secret-that-must-not-echo"
    draft = NormalizedTraceEventDraft(
        run_id=RUN_ID,
        source_event_id="source-1",
        timestamp=NOW,
        event_type=EventType.OBSERVATION,
        phase=EventPhase.POST_ACTION,
        payload={"message": secret},
        source_adapter="fixture",
        trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
    )
    assert secret not in repr(draft)

    values = draft.model_dump(mode="python")
    values["payload"] = {"message": {secret}}
    with pytest.raises(ValidationError) as error:
        validate_normalized_trace_event_draft(values)
    assert secret not in str(error.value)
    assert secret not in repr(error.value)
    assert secret not in str(error.value.errors())
    assert secret not in error.value.json()
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_redactor_validation_errors_drop_structured_input_values() -> None:
    secret = "fixture-secret-that-must-not-echo"

    with pytest.raises(ValidationError) as error:
        Redactor().redact_payload({"message": secret, "ratio": float("inf")})

    assert secret not in str(error.value)
    assert secret not in repr(error.value)
    assert secret not in str(error.value.errors())
    assert secret not in error.value.json()
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_synthetic_event_digest_is_explicit_and_cannot_mix_with_a_key() -> None:
    draft = NormalizedTraceEventDraft(
        run_id=RUN_ID,
        source_event_id="source-1",
        timestamp=NOW,
        event_type=EventType.OBSERVATION,
        phase=EventPhase.POST_ACTION,
        payload={"message": "synthetic fixture"},
        source_adapter="fixture",
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )
    redactor = Redactor()

    with pytest.raises(SyntheticDigestDisabledError):
        redactor.redact_event(draft)
    with pytest.raises(AmbiguousDigestModeError):
        redactor.redact_event(
            draft,
            key=InstallationKey(b"k" * 32),
            synthetic_benchmark=True,
        )

    result = redactor.redact_event(draft, synthetic_benchmark=True)
    assert result.event.payload_digest.algorithm is PayloadDigestAlgorithm.SYNTHETIC_SHA256
    assert verify_redacted_event(
        result.event,
        redactor=redactor,
        synthetic_benchmark=True,
    )
