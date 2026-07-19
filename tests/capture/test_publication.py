from __future__ import annotations

import pytest
from tests.capture.store_support import ZERO_TAG, capture_context, unauthenticated_intake

from saliencegate.capture.publication import (
    CaptureIntakeAuthenticationError,
    authenticate_capture_intake,
    verify_capture_intake_authentication,
)

AUTHENTICATION_FAILED_MESSAGE = "capture intake authentication failed"
SECRET_MARKER = "provider-native-secret-marker"


def _assert_content_free_authentication_error(
    error: BaseException,
    *,
    marker: str,
) -> None:
    assert type(error) is CaptureIntakeAuthenticationError
    assert str(error) == AUTHENTICATION_FAILED_MESSAGE
    assert error.args == (AUTHENTICATION_FAILED_MESSAGE,)
    assert marker not in str(error)
    assert marker not in repr(error)
    assert error.__cause__ is None


def test_authentication_rejects_an_invalid_context_without_mutating_the_intake() -> None:
    valid_context = capture_context()
    intake = unauthenticated_intake("session_started", context=valid_context)
    before = intake.model_dump(mode="python")

    with pytest.raises(CaptureIntakeAuthenticationError) as captured:
        authenticate_capture_intake(
            intake,
            context=SECRET_MARKER,  # type: ignore[arg-type]
        )

    _assert_content_free_authentication_error(captured.value, marker=SECRET_MARKER)
    assert intake.model_dump(mode="python") == before
    assert intake.intake_tag == ZERO_TAG


def test_authentication_normalizes_a_malformed_intake_and_preserves_context_use() -> None:
    context = capture_context()
    valid = unauthenticated_intake("session_started", context=context)
    malformed = valid.model_dump(mode="python")
    malformed["producer_event_digest"] = SECRET_MARKER
    before = dict(malformed)

    with pytest.raises(CaptureIntakeAuthenticationError) as captured:
        authenticate_capture_intake(
            malformed,  # type: ignore[arg-type]
            context=context,
        )

    _assert_content_free_authentication_error(captured.value, marker=SECRET_MARKER)
    assert malformed == before
    authenticated = authenticate_capture_intake(valid, context=context)
    assert authenticated.intake_tag != ZERO_TAG
    assert valid.intake_tag == ZERO_TAG


def test_verification_rejects_an_invalid_context_without_mutating_the_intake() -> None:
    valid_context = capture_context()
    authenticated = authenticate_capture_intake(
        unauthenticated_intake("session_started", context=valid_context),
        context=valid_context,
    )
    before = authenticated.model_dump(mode="python")

    with pytest.raises(CaptureIntakeAuthenticationError) as captured:
        verify_capture_intake_authentication(
            authenticated,
            context=SECRET_MARKER,  # type: ignore[arg-type]
        )

    _assert_content_free_authentication_error(captured.value, marker=SECRET_MARKER)
    assert authenticated.model_dump(mode="python") == before
    verified = verify_capture_intake_authentication(authenticated, context=valid_context)
    assert verified == authenticated
    assert verified is not authenticated
