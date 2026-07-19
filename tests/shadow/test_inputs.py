from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from types import MappingProxyType
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import EventPhase, EventType, SignalType, TrustLabel
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInputError,
    ShadowInvariantError,
    ShadowStateError,
)
from saliencegate.shadow.inputs import (
    SHADOW_PROJECTION_MATRIX,
    ShadowActionInput,
    ShadowControllerErrorInput,
    ShadowEventRef,
    ShadowFinishInput,
    ShadowInputKind,
    ShadowObservationInput,
    ShadowObservationSource,
    ShadowProjectionSpec,
    ShadowStartInput,
    ShadowTestResultInput,
    ShadowToolResultInput,
    derive_shadow_event_id,
    derive_shadow_source_event_digest,
    project_shadow_input,
)
from saliencegate.signals.fingerprints import TestFailureEvidence as _TestFailureEvidence

RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_RUN_ID = UUID("22222222-2222-4222-8222-222222222222")
ACTION_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
OCCURRED_AT = datetime(2026, 7, 16, 10, 0, tzinfo=UTC)
ENVIRONMENT_DIGEST = "a" * 64
SOURCE_ADAPTER = "example-adapter/v1"


class _DatetimeSubclass(datetime):
    pass


class _UUIDSubclass(UUID):
    pass


class _StringSubclass(str):
    pass


def action_ref(*, run_id: UUID = RUN_ID, sequence: int = 2) -> ShadowEventRef:
    return ShadowEventRef(run_id=run_id, event_id=ACTION_ID, sequence=sequence)


def common_fields(
    *,
    source_event_id: str = "event-1",
    occurred_at: datetime = OCCURRED_AT,
) -> dict[str, object]:
    return {
        "source_event_id": source_event_id,
        "occurred_at": occurred_at,
    }


def test_public_errors_are_argument_free_value_free_families() -> None:
    cases = (
        (ShadowInputError, "shadow input is invalid"),
        (ShadowConfigurationError, "shadow configuration is invalid"),
        (ShadowStateError, "shadow state is invalid"),
        (ShadowInvariantError, "shadow invariant is invalid"),
    )

    for error_type, message in cases:
        error = error_type()
        assert str(error) == message
        assert not vars(error)
        assert error.__cause__ is None
        assert error.__context__ is None
        with pytest.raises(TypeError):
            error_type("caller secret")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("model", "kind"),
    (
        (ShadowStartInput(**common_fields()), ShadowInputKind.START),
        (
            ShadowActionInput(
                **common_fields(),
                argv=("pytest", "-q"),
                working_directory="/project",
                environment_digest=ENVIRONMENT_DIGEST,
            ),
            ShadowInputKind.ACTION,
        ),
        (
            ShadowToolResultInput(
                **common_fields(),
                action=action_ref(),
                status="failed",
                exit_status=1,
            ),
            ShadowInputKind.TOOL_RESULT,
        ),
        (
            ShadowTestResultInput(
                **common_fields(),
                action=action_ref(),
                framework="pytest",
                status="passed",
                failures=(),
            ),
            ShadowInputKind.TEST_RESULT,
        ),
        (
            ShadowObservationInput(
                **common_fields(),
                source=ShadowObservationSource.TASK_INPUT,
                payload={"task": "bounded"},
            ),
            ShadowInputKind.OBSERVATION,
        ),
        (
            ShadowControllerErrorInput(
                **common_fields(),
                error_code="repository_unavailable",
            ),
            ShadowInputKind.CONTROLLER_ERROR,
        ),
        (ShadowFinishInput(**common_fields()), ShadowInputKind.FINISH),
    ),
)
def test_all_input_records_are_strict_frozen_and_versioned(
    model: object,
    kind: ShadowInputKind,
) -> None:
    assert model.schema_version == "shadow-input/v1"
    assert model.kind == kind.value

    with pytest.raises(ValidationError):
        type(model).model_validate({**model.model_dump(), "unexpected": True})  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        model.source_event_id = "replacement"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "source_event_id",
    (
        "",
        "a" * 129,
        "contains whitespace",
        "path/segment",
        "namespace:value",
        "ümlaut",
        _StringSubclass("subclass"),
    ),
)
def test_source_event_id_is_an_exact_bounded_opaque_identifier(source_event_id: str) -> None:
    with pytest.raises(ValidationError):
        ShadowStartInput(**common_fields(source_event_id=source_event_id))


@pytest.mark.parametrize(
    "occurred_at",
    (
        datetime(2026, 7, 16, 10, 0),
        datetime(2026, 7, 16, 10, 0, tzinfo=timezone(timedelta(hours=1))),
        "2026-07-16T10:00:00Z",
        _DatetimeSubclass(2026, 7, 16, 10, 0, tzinfo=UTC),
    ),
)
def test_occurred_at_requires_an_exact_utc_datetime(occurred_at: object) -> None:
    with pytest.raises(ValidationError):
        ShadowStartInput(**common_fields(occurred_at=occurred_at))  # type: ignore[arg-type]


def test_action_requires_exactly_one_bounded_shell_form() -> None:
    argv_input = ShadowActionInput(
        **common_fields(),
        argv=("pytest", "-q"),
        working_directory="/project",
        environment_digest=ENVIRONMENT_DIGEST,
    )
    command_input = ShadowActionInput(
        **common_fields(),
        command="pytest -q",
        working_directory="/project",
        environment_digest=ENVIRONMENT_DIGEST,
    )

    assert argv_input.argv == ("pytest", "-q")
    assert argv_input.command is None
    assert command_input.command == "pytest -q"
    assert command_input.argv is None
    assert "pytest" not in repr(argv_input)
    assert "/project" not in repr(argv_input)

    invalid = (
        {},
        {"command": "pytest", "argv": ("pytest",)},
        {"argv": ["pytest"]},
        {"argv": (_StringSubclass("pytest"),)},
        {"command": "pytest", "environment_digest": _StringSubclass(ENVIRONMENT_DIGEST)},
        {"command": "pytest", "environment_digest": "A" * 64},
        {"command": "pytest", "environment_digest": "a" * 63},
    )
    for fields in invalid:
        model_fields: dict[str, object] = {
            "working_directory": "/project",
            "environment_digest": ENVIRONMENT_DIGEST,
        }
        model_fields.update(fields)
        with pytest.raises(ValidationError):
            ShadowActionInput(
                **common_fields(),
                **model_fields,
            )


def test_event_reference_is_exact_frozen_and_contains_no_source_id() -> None:
    reference = action_ref()

    assert reference.schema_version == "shadow-event-ref/v1"
    assert reference.run_id == RUN_ID
    assert reference.event_id == ACTION_ID
    assert reference.sequence == 2
    assert "source" not in reference.model_dump()
    assert str(RUN_ID) not in repr(reference)
    assert str(ACTION_ID) not in repr(reference)

    invalid = (
        {"run_id": UUID(int=0), "event_id": ACTION_ID, "sequence": 2},
        {"run_id": _UUIDSubclass(str(RUN_ID)), "event_id": ACTION_ID, "sequence": 2},
        {"run_id": RUN_ID, "event_id": ACTION_ID, "sequence": 0},
        {"run_id": RUN_ID, "event_id": ACTION_ID, "sequence": True},
        {"run_id": str(RUN_ID), "event_id": ACTION_ID, "sequence": 2},
    )
    for fields in invalid:
        with pytest.raises(ValidationError):
            ShadowEventRef(**fields)  # type: ignore[arg-type]


def test_tool_result_copies_parent_and_enforces_structured_evidence() -> None:
    parent = action_ref()
    tool_result = ShadowToolResultInput(
        **common_fields(),
        action=parent,
        status="failed",
        exit_status=1,
        exception_type="AssertionError",
    )

    assert tool_result.action == parent
    assert tool_result.action is not parent
    assert "AssertionError" not in repr(tool_result)

    with pytest.raises(ValidationError):
        ShadowToolResultInput(
            **common_fields(),
            action=parent,
            status="succeeded",
            exit_status=1,
        )
    with pytest.raises(ValidationError):
        ShadowToolResultInput(**common_fields(), action=parent)
    with pytest.raises(ValidationError):
        ShadowToolResultInput(
            **common_fields(),
            action=parent,
            status=_StringSubclass("failed"),
            exit_status=1,
        )


def test_test_result_copies_and_validates_failure_envelopes() -> None:
    raw_failure = {
        "schema_version": "1.0",
        "test_id": "tests/test_demo.py::test_example",
        "failure_type": "AssertionError",
        "signature": "expected one",
    }
    failures = (raw_failure,)
    test_result = ShadowTestResultInput(
        **common_fields(),
        action=action_ref(),
        framework="pytest",
        status="failed",
        failures=failures,
    )
    raw_failure["test_id"] = "mutated"

    assert type(test_result.failures[0]) is _TestFailureEvidence
    assert test_result.failures[0].test_id == "tests/test_demo.py::test_example"
    assert "expected one" not in repr(test_result)

    with pytest.raises(ValidationError):
        ShadowTestResultInput(
            **common_fields(),
            action=action_ref(),
            framework="pytest",
            status="passed",
            failures=failures,
        )
    with pytest.raises(ValidationError):
        ShadowTestResultInput(
            **common_fields(),
            action=action_ref(),
            framework="pytest",
            status="failed",
            failures=(
                {
                    "schema_version": "1.0",
                    "test_id": _StringSubclass("tests/test_demo.py::test_example"),
                },
            ),
        )


def test_test_failure_copy_never_dispatches_through_the_caller_instance() -> None:
    secret = "caller-controlled-secret"
    serializer_called = False
    failure = _TestFailureEvidence(
        schema_version="1.0",
        test_id="tests/test_demo.py::test_example",
        failure_type="AssertionError",
    )

    def poisoned_model_dump(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError(secret)

    class PoisonedSerializer:
        def to_python(self, *args: object, **kwargs: object) -> object:
            nonlocal serializer_called
            del args, kwargs
            serializer_called = True
            raise AssertionError(secret)

    object.__setattr__(failure, "model_dump", poisoned_model_dump)
    object.__setattr__(failure, "__pydantic_serializer__", PoisonedSerializer())

    result = ShadowTestResultInput(
        **common_fields(),
        action=action_ref(),
        framework="pytest",
        status="failed",
        failures=(failure,),
    )

    assert type(result.failures[0]) is _TestFailureEvidence
    assert result.failures[0].test_id == "tests/test_demo.py::test_example"
    assert "model_dump" not in result.failures[0].__dict__
    assert "__pydantic_serializer__" not in result.failures[0].__dict__
    assert serializer_called is False
    assert secret not in repr(result)


def test_test_failure_aggregate_is_bounded_before_nested_serialization() -> None:
    serializer_called = False
    failure = _TestFailureEvidence(
        schema_version="1.0",
        test_id="tests/test_demo.py::test_example",
        failure_type="AssertionError",
        signature="x" * (128 * 1_024),
    )

    class PoisonedSerializer:
        def to_python(self, *args: object, **kwargs: object) -> object:
            nonlocal serializer_called
            del args, kwargs
            serializer_called = True
            raise AssertionError("aggregate-secret")

    object.__setattr__(failure, "__pydantic_serializer__", PoisonedSerializer())

    with pytest.raises(ValidationError) as caught:
        ShadowTestResultInput(
            **common_fields(),
            action=action_ref(),
            framework="pytest",
            status="failed",
            failures=(failure,) * 3,
        )

    assert serializer_called is False
    assert "aggregate-secret" not in str(caught.value)


@pytest.mark.parametrize(
    "source",
    tuple(ShadowObservationSource),
)
def test_observation_accepts_only_approved_untrusted_sources_and_copies_payload(
    source: ShadowObservationSource,
) -> None:
    caller_payload = {"nested": {"items": [1, 2]}}
    observation = ShadowObservationInput(
        **common_fields(),
        source=source,
        payload=caller_payload,
    )
    caller_payload["nested"] = {"items": [99]}

    assert observation.payload == {"nested": {"items": (1, 2)}}
    assert isinstance(observation.payload, MappingProxyType)
    assert "nested" not in repr(observation)
    with pytest.raises(TypeError):
        observation.payload["new"] = True  # type: ignore[index]


@pytest.mark.parametrize(
    "reserved_key",
    (
        "shadow_run",
        "shadow_run_end",
        "action",
        "tool_outcome",
        "test_report",
        "controller_error",
    ),
)
def test_observation_rejects_reserved_top_level_namespaces(reserved_key: str) -> None:
    with pytest.raises(ValidationError):
        ShadowObservationInput(
            **common_fields(),
            source=ShadowObservationSource.TOOL_OUTPUT,
            payload={reserved_key: {}},
        )


def test_observation_rejects_unapproved_source_and_unbounded_payload() -> None:
    with pytest.raises(ValidationError):
        ShadowObservationInput(
            **common_fields(),
            source="trusted_controller",  # type: ignore[arg-type]
            payload={},
        )
    unbounded: dict[str, object] = {}
    cursor = unbounded
    for _ in range(65):
        nested: dict[str, object] = {}
        cursor["nested"] = nested
        cursor = nested
    with pytest.raises(ValidationError):
        ShadowObservationInput(
            **common_fields(),
            source=ShadowObservationSource.MODEL_OUTPUT,
            payload=unbounded,
        )


def test_controller_error_is_one_component_identifier_without_free_text() -> None:
    controller_error = ShadowControllerErrorInput(
        **common_fields(),
        error_code="repository_unavailable",
    )

    assert controller_error.error_code == "repository_unavailable"
    assert "repository_unavailable" not in repr(controller_error)
    with pytest.raises(ValidationError):
        ShadowControllerErrorInput(
            **common_fields(),
            error_code="contains whitespace",
        )
    with pytest.raises(ValidationError):
        ShadowControllerErrorInput(
            **common_fields(),
            error_code="repository_unavailable",
            details="caller exception text",  # type: ignore[call-arg]
        )


def test_projection_matrix_freezes_every_normative_row() -> None:
    assert isinstance(SHADOW_PROJECTION_MATRIX, MappingProxyType)
    assert tuple(SHADOW_PROJECTION_MATRIX) == tuple(ShadowInputKind)
    assert {
        ShadowInputKind.START: ShadowProjectionSpec(
            event_type=EventType.RUN_START,
            phase=EventPhase.INITIALIZATION,
            trust_label=TrustLabel.TRUSTED_CONTROLLER,
            payload_namespace="shadow_run",
            parent="none",
            applicable_detectors=(),
        ),
        ShadowInputKind.ACTION: ShadowProjectionSpec(
            event_type=EventType.ACTION_PROPOSAL,
            phase=EventPhase.PRE_ACTION,
            trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            payload_namespace="action",
            parent="none",
            applicable_detectors=(SignalType.REPEATED_ACTION,),
        ),
        ShadowInputKind.TOOL_RESULT: ShadowProjectionSpec(
            event_type=EventType.TOOL_COMPLETION,
            phase=EventPhase.POST_ACTION,
            trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
            payload_namespace="tool_outcome",
            parent="action",
            applicable_detectors=(SignalType.TOOL_ERROR, SignalType.REPEATED_FAILURE),
        ),
        ShadowInputKind.TEST_RESULT: ShadowProjectionSpec(
            event_type=EventType.OBSERVATION,
            phase=EventPhase.POST_ACTION,
            trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
            payload_namespace="test_report",
            parent="action",
            applicable_detectors=(SignalType.TEST_FAILURE, SignalType.REPEATED_FAILURE),
        ),
        ShadowInputKind.OBSERVATION: ShadowProjectionSpec(
            event_type=EventType.OBSERVATION,
            phase=EventPhase.POST_ACTION,
            trust_label=TrustLabel.UNTRUSTED_TASK_INPUT,
            payload_namespace="observation",
            parent="none",
            applicable_detectors=(),
        ),
        ShadowInputKind.CONTROLLER_ERROR: ShadowProjectionSpec(
            event_type=EventType.CONTROLLER_ERROR,
            phase=EventPhase.INTERNAL,
            trust_label=TrustLabel.TRUSTED_CONTROLLER,
            payload_namespace="controller_error",
            parent="none",
            applicable_detectors=(SignalType.TOOL_ERROR,),
        ),
        ShadowInputKind.FINISH: ShadowProjectionSpec(
            event_type=EventType.RUN_END,
            phase=EventPhase.TERMINAL,
            trust_label=TrustLabel.TRUSTED_CONTROLLER,
            payload_namespace="shadow_run_end",
            parent="none",
            applicable_detectors=(),
        ),
    } == SHADOW_PROJECTION_MATRIX

    with pytest.raises(TypeError):
        SHADOW_PROJECTION_MATRIX[ShadowInputKind.START] = SHADOW_PROJECTION_MATRIX[  # type: ignore[index]
            ShadowInputKind.START
        ]


def test_observation_source_overrides_only_the_frozen_untrusted_trust_label() -> None:
    expected = {
        ShadowObservationSource.TASK_INPUT: TrustLabel.UNTRUSTED_TASK_INPUT,
        ShadowObservationSource.TOOL_OUTPUT: TrustLabel.UNTRUSTED_TOOL_OUTPUT,
        ShadowObservationSource.MODEL_OUTPUT: TrustLabel.UNTRUSTED_MODEL_OUTPUT,
        ShadowObservationSource.EXTERNAL_MEMORY: TrustLabel.UNTRUSTED_EXTERNAL_MEMORY,
    }

    for source, trust_label in expected.items():
        projected = project_shadow_input(
            ShadowObservationInput(
                **common_fields(),
                source=source,
                payload={"bounded": True},
            ),
            run_id=RUN_ID,
            source_adapter=SOURCE_ADAPTER,
        )
        assert projected.trust_label is trust_label


def test_identity_helpers_freeze_domains_framing_and_uuid_bits() -> None:
    event_id = derive_shadow_event_id(RUN_ID, "action-1")

    assert event_id == UUID("c90d6563-0116-4a6f-8a37-e9d3e9c18c4d")
    assert type(event_id) is UUID
    assert event_id.version == 4
    assert event_id.variant == "specified in RFC 4122"
    assert derive_shadow_source_event_digest(RUN_ID, "action-1") == (
        "a41fc21e8245078f7bd8fda841bfa5803b502a81e1628c074dfcee7669dacf94"
    )
    assert derive_shadow_event_id(RUN_ID, "ab-c") != derive_shadow_event_id(RUN_ID, "a-bc")
    assert derive_shadow_event_id(RUN_ID, "action-1") != derive_shadow_event_id(
        OTHER_RUN_ID,
        "action-1",
    )


@pytest.mark.parametrize(
    ("run_id", "source_event_id"),
    (
        (UUID(int=0), "event-1"),
        (_UUIDSubclass(str(RUN_ID)), "event-1"),
        (RUN_ID, "contains whitespace"),
        (RUN_ID, _StringSubclass("event-1")),
    ),
)
def test_identity_helpers_map_invalid_public_values_to_sanitized_error(
    run_id: UUID,
    source_event_id: str,
) -> None:
    for helper in (derive_shadow_event_id, derive_shadow_source_event_digest):
        with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$") as caught:
            helper(run_id, source_event_id)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_projection_builds_exact_normalized_drafts() -> None:
    cases = (
        (
            ShadowStartInput(**common_fields(source_event_id="start")),
            {"shadow_run": {"schema_version": "shadow-run/v1", "capture_scope": "unknown"}},
            (),
            TrustLabel.TRUSTED_CONTROLLER,
            EventType.RUN_START,
            EventPhase.INITIALIZATION,
        ),
        (
            ShadowActionInput(
                **common_fields(source_event_id="action"),
                argv=("pytest", "-q"),
                working_directory="/project",
                environment_digest=ENVIRONMENT_DIGEST,
            ),
            {
                "action": {
                    "schema_version": "1.0",
                    "kind": "shell",
                    "command": None,
                    "argv": ["pytest", "-q"],
                    "working_directory": "/project",
                    "environment_digest": ENVIRONMENT_DIGEST,
                }
            },
            (),
            TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            EventType.ACTION_PROPOSAL,
            EventPhase.PRE_ACTION,
        ),
        (
            ShadowToolResultInput(
                **common_fields(source_event_id="tool"),
                action=action_ref(),
                status="failed",
                exit_status=1,
                exception_type="AssertionError",
            ),
            {
                "tool_outcome": {
                    "schema_version": "1.0",
                    "status": "failed",
                    "exit_status": 1,
                    "exception_type": "AssertionError",
                    "error_code": None,
                    "failure_signature": None,
                }
            },
            (ACTION_ID,),
            TrustLabel.UNTRUSTED_TOOL_OUTPUT,
            EventType.TOOL_COMPLETION,
            EventPhase.POST_ACTION,
        ),
        (
            ShadowTestResultInput(
                **common_fields(source_event_id="test"),
                action=action_ref(),
                framework="pytest",
                status="failed",
                failures=(
                    {
                        "schema_version": "1.0",
                        "test_id": "tests/test_demo.py::test_example",
                        "failure_type": "AssertionError",
                        "signature": None,
                    },
                ),
            ),
            {
                "test_report": {
                    "schema_version": "1.0",
                    "framework": "pytest",
                    "status": "failed",
                    "failures": [
                        {
                            "schema_version": "1.0",
                            "test_id": "tests/test_demo.py::test_example",
                            "failure_type": "AssertionError",
                            "signature": None,
                        }
                    ],
                }
            },
            (ACTION_ID,),
            TrustLabel.UNTRUSTED_TOOL_OUTPUT,
            EventType.OBSERVATION,
            EventPhase.POST_ACTION,
        ),
        (
            ShadowObservationInput(
                **common_fields(source_event_id="observation"),
                source=ShadowObservationSource.MODEL_OUTPUT,
                payload={"bounded": [1, 2]},
            ),
            {"observation": {"bounded": [1, 2]}},
            (),
            TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            EventType.OBSERVATION,
            EventPhase.POST_ACTION,
        ),
        (
            ShadowControllerErrorInput(
                **common_fields(source_event_id="controller-error"),
                error_code="repository_unavailable",
            ),
            {
                "controller_error": {
                    "schema_version": "controller_error/v1",
                    "error_code": "repository_unavailable",
                }
            },
            (),
            TrustLabel.TRUSTED_CONTROLLER,
            EventType.CONTROLLER_ERROR,
            EventPhase.INTERNAL,
        ),
        (
            ShadowFinishInput(**common_fields(source_event_id="finish")),
            {
                "shadow_run_end": {
                    "schema_version": "shadow-run-end/v1",
                    "start_event_id": str(ACTION_ID),
                }
            },
            (),
            TrustLabel.TRUSTED_CONTROLLER,
            EventType.RUN_END,
            EventPhase.TERMINAL,
        ),
    )

    for value, expected_payload, parent_ids, trust_label, event_type, phase in cases:
        kwargs: dict[str, object] = {}
        if type(value) is ShadowStartInput:
            kwargs["start_payload"] = expected_payload["shadow_run"]
        elif type(value) is ShadowFinishInput:
            kwargs["finish_payload"] = expected_payload["shadow_run_end"]
        projected = project_shadow_input(
            value,
            run_id=RUN_ID,
            source_adapter=SOURCE_ADAPTER,
            **kwargs,
        )

        assert projected.run_id == RUN_ID
        assert projected.source_event_id == value.source_event_id
        assert projected.timestamp == OCCURRED_AT
        assert projected.event_type is event_type
        assert projected.phase is phase
        assert projected.model_dump(mode="json")["payload"] == expected_payload
        assert projected.parent_ids == parent_ids
        assert projected.source_adapter == SOURCE_ADAPTER
        assert projected.trust_label is trust_label


def test_projection_requires_exact_marker_payload_and_rejects_misplaced_payloads() -> None:
    start = ShadowStartInput(**common_fields())
    finish = ShadowFinishInput(**common_fields())
    action = ShadowActionInput(
        **common_fields(),
        command="pytest -q",
        working_directory="/project",
        environment_digest=ENVIRONMENT_DIGEST,
    )

    invalid_calls = (
        (start, {}),
        (start, {"start_payload": "not-a-mapping"}),
        (finish, {}),
        (finish, {"finish_payload": "not-a-mapping"}),
        (action, {"start_payload": {}}),
        (action, {"finish_payload": {}}),
    )
    for value, kwargs in invalid_calls:
        with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$") as caught:
            project_shadow_input(
                value,
                run_id=RUN_ID,
                source_adapter=SOURCE_ADAPTER,
                **kwargs,  # type: ignore[arg-type]
            )
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "value",
    (
        ShadowToolResultInput(
            **common_fields(),
            action=action_ref(run_id=OTHER_RUN_ID),
            status="failed",
            exit_status=1,
        ),
        ShadowTestResultInput(
            **common_fields(),
            action=action_ref(run_id=OTHER_RUN_ID),
            framework="pytest",
            status="passed",
            failures=(),
        ),
    ),
)
def test_projection_rejects_cross_run_result_parent_without_echoing_identity(
    value: object,
) -> None:
    with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$") as caught:
        project_shadow_input(
            value,
            run_id=RUN_ID,
            source_adapter=SOURCE_ADAPTER,
        )
    assert str(OTHER_RUN_ID) not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_projection_revalidates_forged_inputs_and_never_echoes_content() -> None:
    valid = ShadowActionInput(
        **common_fields(),
        command="caller-secret-command",
        working_directory="/caller-secret-directory",
        environment_digest=ENVIRONMENT_DIGEST,
    )
    forged = valid.model_copy(update={"environment_digest": "invalid-secret-digest"})

    with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$") as caught:
        project_shadow_input(
            forged,
            run_id=RUN_ID,
            source_adapter=SOURCE_ADAPTER,
        )
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_inputs_module_exports_only_the_deliberate_contract() -> None:
    from saliencegate.shadow import inputs

    assert inputs.__all__ == [
        "SHADOW_PROJECTION_MATRIX",
        "ShadowActionInput",
        "ShadowControllerErrorInput",
        "ShadowEventRef",
        "ShadowFinishInput",
        "ShadowInputKind",
        "ShadowInputRecord",
        "ShadowObservationInput",
        "ShadowObservationSource",
        "ShadowProjectionSpec",
        "ShadowStartInput",
        "ShadowTestResultInput",
        "ShadowToolResultInput",
        "derive_shadow_event_id",
        "derive_shadow_source_event_digest",
        "project_shadow_input",
    ]
