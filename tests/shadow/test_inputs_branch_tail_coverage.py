"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

import saliencegate.shadow.inputs as inputs_module
from saliencegate.shadow.inputs import (
    ShadowControllerErrorInput,
    ShadowEventRef,
    ShadowStartInput,
    ShadowTestResultInput,
    ShadowToolResultInput,
)
from saliencegate.signals import TestFailureEvidence as FailureEvidence

_RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def test_input_scalar_validators_reject_non_exact_aliases() -> None:
    calls = (
        lambda: inputs_module._require_exact_source_event_id(1),
        lambda: ShadowEventRef.require_exact_schema_version(1),
        lambda: ShadowStartInput.require_exact_schema_version(1),
        lambda: ShadowStartInput.require_exact_kind(1),
        lambda: ShadowToolResultInput.require_exact_optional_tool_text(1),
        lambda: ShadowTestResultInput.require_exact_framework(1),
        lambda: ShadowTestResultInput.require_exact_test_status(1),
        lambda: ShadowControllerErrorInput.require_exact_error_code(1),
    )

    for call in calls:
        with pytest.raises(ValueError):
            call()


def test_failure_copy_preflight_rejects_every_invalid_member_shape() -> None:
    valid = FailureEvidence(
        schema_version="1.0",
        test_id="tests/test_coverage.py::test_tail",
        failure_type="AssertionError",
        signature="coverage-signature",
    )
    corrupted = valid.model_copy()
    object.__setattr__(corrupted, "schema_version", 1)

    invalid = (
        [],
        (corrupted,),
        (object(),),
        (
            {
                "schema_version": "1.0",
                "test_id": "tests/test_coverage.py::test_tail",
                "extra": True,
            },
        ),
        (
            {
                "schema_version": "1.0",
                "test_id": "tests/test_coverage.py::test_tail",
                "failure_type": 1,
            },
        ),
    )
    for value in invalid:
        with pytest.raises(ValueError, match="test failure evidence is invalid"):
            ShadowTestResultInput.defensively_copy_failures(value)


def test_failure_copy_accounts_for_optional_text_before_copying() -> None:
    failure = {
        "schema_version": "1.0",
        "test_id": "tests/test_coverage.py::test_tail",
        "failure_type": "AssertionError",
        "signature": "coverage-signature",
    }

    copied = ShadowTestResultInput.defensively_copy_failures((failure,))

    assert copied == (FailureEvidence.model_validate(failure),)
    assert copied[0] is not failure


def test_projection_internal_boundaries_reject_unknown_models_and_adapter_aliases() -> None:
    with pytest.raises(ValueError, match="unsupported shadow input"):
        inputs_module._validated_input(object())

    start = ShadowStartInput(
        source_event_id="coverage-start",
        occurred_at=datetime(2026, 7, 21, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="source adapter is invalid"):
        inputs_module._project_validated_input(
            start,
            run_id=_RUN_ID,
            source_adapter=1,  # type: ignore[arg-type]
            start_payload={"schema_version": "shadow-run/v1"},
            finish_payload=None,
        )
