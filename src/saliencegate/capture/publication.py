"""Authenticated publication boundaries for pseudonymized capture intake."""

from __future__ import annotations

import hmac

from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.schema import CaptureIntake, validate_capture_intake
from saliencegate.domain import canonical_json


class CaptureIntakeAuthenticationError(ValueError):
    """An intake could not be authenticated without exposing its values."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture intake authentication failed")


def _authentication_preimage(intake: CaptureIntake) -> bytes:
    return canonical_json(
        {
            "schema_version": "capture-intake-integrity/v1",
            "intake": intake.model_dump(
                mode="json",
                exclude={"intake_tag"},
                warnings="error",
            ),
        }
    )


def authenticate_capture_intake(
    intake: CaptureIntake,
    *,
    context: CaptureDigestContext,
) -> CaptureIntake:
    """Return a defensively copied intake authenticated over its semantic fields."""

    try:
        if type(context) is not CaptureDigestContext:
            raise CaptureIntakeAuthenticationError()
        validated = validate_capture_intake(intake)
        tag = context.integrity_tag(_authentication_preimage(validated))
        return validate_capture_intake(validated.model_copy(update={"intake_tag": tag}))
    except CaptureIntakeAuthenticationError:
        raise
    except Exception:
        raise CaptureIntakeAuthenticationError() from None


def verify_capture_intake_authentication(
    intake: object,
    *,
    context: CaptureDigestContext,
) -> CaptureIntake:
    """Validate and return one intake only when its integrity tag is exact."""

    try:
        if type(context) is not CaptureDigestContext:
            raise CaptureIntakeAuthenticationError()
        validated = validate_capture_intake(intake)
        expected = context.integrity_tag(_authentication_preimage(validated))
        if not hmac.compare_digest(validated.intake_tag, expected):
            raise CaptureIntakeAuthenticationError()
        return validated
    except CaptureIntakeAuthenticationError:
        raise
    except Exception:
        raise CaptureIntakeAuthenticationError() from None


__all__ = [
    "CaptureIntakeAuthenticationError",
    "authenticate_capture_intake",
    "verify_capture_intake_authentication",
]
