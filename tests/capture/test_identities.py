from __future__ import annotations

import copy
import hmac
import pickle
import re
from hashlib import sha256

import pytest

from saliencegate.capture.identities import CaptureDigestContext, CaptureIdentityError
from saliencegate.domain import canonical_json
from saliencegate.security import InstallationKey

KEY_MATERIAL = b"capture-identity-test-key-value!"
OTHER_KEY_MATERIAL = b"other-capture-test-key-material!"
LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")

DOMAINS = {
    "session_id": b"saliencegate:capture:session-id:v1",
    "call_ref": b"saliencegate:capture:call-ref:v1",
    "producer_event": b"saliencegate:capture:producer-event:v1",
    "action_identity": b"saliencegate:capture:action-identity:v1",
    "unavailable_action_identity": b"saliencegate:capture:unavailable-action-identity:v1",
    "workspace_identity": b"saliencegate:capture:workspace-identity:v1",
    "environment_identity": b"saliencegate:capture:environment-identity:v1",
    "failure_signature": b"saliencegate:capture:failure-signature:v1",
    "subagent_id": b"saliencegate:capture:subagent-id:v1",
    "turn_id": b"saliencegate:capture:turn-id:v1",
    "transport_batch_ref": b"saliencegate:capture:transport-batch-ref:v1",
    "transport_chunk_digest": b"saliencegate:capture:transport-chunk-digest:v1",
    "integrity_tag": b"saliencegate:capture:integrity-tag:v1",
}


def _expected_hmac(material: bytes, *, domain: bytes, value: bytes) -> str:
    framed = (
        len(domain).to_bytes(8, byteorder="big", signed=False)
        + domain
        + len(value).to_bytes(8, byteorder="big", signed=False)
        + value
    )
    return hmac.new(material, framed, sha256).hexdigest()


def _context(material: bytes = KEY_MATERIAL) -> CaptureDigestContext:
    return CaptureDigestContext(InstallationKey(material))


def test_named_capture_digests_are_deterministic_hmac_sha256_known_vectors() -> None:
    value = b"same-provider-native-value"
    first = _context()
    second = _context()

    for method_name, domain in DOMAINS.items():
        method = getattr(first, method_name)
        digest = method(value)
        assert digest == getattr(second, method_name)(value)
        assert digest == _expected_hmac(KEY_MATERIAL, domain=domain, value=value)
        assert LOWER_SHA256.fullmatch(digest)


def test_every_identity_purpose_has_a_distinct_domain() -> None:
    context = _context()
    value = b"one-reused-provider-identifier"

    digests = {getattr(context, method_name)(value) for method_name in DOMAINS}

    assert len(digests) == len(DOMAINS)
    assert context.session_id(value) != _context(OTHER_KEY_MATERIAL).session_id(value)


def test_pseudonymous_session_call_subagent_and_turn_ids_are_locally_linkable_only() -> None:
    context = _context()
    native = b"provider-secret-session-id"

    session_id = context.session_id(native)
    assert context.session_id(native) == session_id
    assert context.session_id(native + b"-other") != session_id
    assert native.decode() not in session_id
    assert session_id not in context.call_ref(native)
    assert session_id not in context.subagent_id(native)
    assert session_id not in context.turn_id(native)


def test_action_identity_commits_exact_canonical_bytes_without_order_artifacts() -> None:
    context = _context()
    first = canonical_json(
        {
            "tool_name": "exec_command",
            "native_input": {"argv": ["pytest", "-q"], "cwd": "/synthetic"},
        }
    )
    second = canonical_json(
        {
            "native_input": {"cwd": "/synthetic", "argv": ["pytest", "-q"]},
            "tool_name": "exec_command",
        }
    )
    changed = canonical_json(
        {
            "tool_name": "exec_command",
            "native_input": {"argv": ["pytest", "-x"], "cwd": "/synthetic"},
        }
    )

    assert first == second
    assert context.action_identity(first) == context.action_identity(second)
    assert context.action_identity(first) != context.action_identity(changed)
    assert b"pytest" not in context.action_identity(first).encode()


def test_unavailable_action_identity_is_per_call_and_never_matches_exact_identity() -> None:
    context = _context()
    first_call = b"call-1"
    second_call = b"call-2"

    first = context.unavailable_action_identity(first_call)
    assert first == context.unavailable_action_identity(first_call)
    assert first != context.unavailable_action_identity(second_call)
    assert first != context.action_identity(first_call)


def test_digest_context_never_exposes_or_serializes_the_installation_key() -> None:
    context = _context()
    marker = KEY_MATERIAL.decode()

    assert repr(context) == "CaptureDigestContext(<redacted>)"
    assert str(context) == repr(context)
    assert marker not in repr(context)
    assert not hasattr(context, "key")
    assert not hasattr(context, "material")
    assert not hasattr(context, "hmac_sha256")
    with pytest.raises(TypeError):
        vars(context)
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(context)
    with pytest.raises(TypeError):
        copy.copy(context)
    with pytest.raises(AttributeError):
        context.key = InstallationKey(OTHER_KEY_MATERIAL)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "invalid",
    (
        b"",
        "provider-native-secret",
        bytearray(b"provider-native-secret"),
        memoryview(b"provider-native-secret"),
    ),
)
def test_digest_inputs_are_exact_nonempty_bytes_and_errors_are_content_free(
    invalid: object,
) -> None:
    context = _context()

    with pytest.raises(CaptureIdentityError) as raised:
        context.action_identity(invalid)  # type: ignore[arg-type]

    assert str(raised.value) == "capture identity is invalid"
    assert "provider-native-secret" not in str(raised.value)
    assert "provider-native-secret" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_digest_context_requires_an_exact_installation_key() -> None:
    for invalid in (KEY_MATERIAL, object(), None):
        with pytest.raises(CaptureIdentityError):
            CaptureDigestContext(invalid)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        CaptureIdentityError("provider-native-secret")  # type: ignore[call-arg]
