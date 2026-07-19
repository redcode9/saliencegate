from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from pydantic import ValidationError

from saliencegate.domain import EventPhase, EventType, SignalType, TraceEvent
from saliencegate.signals.base import (
    AbstentionReason,
    DetectionContext,
    DetectionInputError,
    DetectionStatus,
    DeterministicSignalExtractor,
)
from saliencegate.signals.repetition import (
    RepeatedActionDetector,
    RepeatedFailureDetector,
    RepetitionConfig,
)

EventFactory = Callable[..., TraceEvent]
OPAQUE_ACTION_DIGEST = "b" * 64
OPAQUE_WORKSPACE_DIGEST = "c" * 64
OPAQUE_ENVIRONMENT_DIGEST = "d" * 64


def context(*events: TraceEvent) -> DetectionContext:
    return DetectionContext(run_id=events[-1].run_id, events=events)


def action(
    event_factory: EventFactory,
    sequence: int,
    command: str = "pytest --quiet tests/test_api.py",
) -> TraceEvent:
    return event_factory(
        sequence,
        event_type=EventType.ACTION_PROPOSAL,
        phase=EventPhase.PRE_ACTION,
        payload={
            "action": {
                "schema_version": "1.0",
                "kind": "shell",
                "command": command,
                "working_directory": "/workspace",
                "environment_digest": "a" * 64,
            }
        },
    )


def opaque_action(
    event_factory: EventFactory,
    sequence: int,
    *,
    identity_authority: str = "exact",
    action_digest: str = OPAQUE_ACTION_DIGEST,
    workspace_digest: str = OPAQUE_WORKSPACE_DIGEST,
    environment_digest: str = OPAQUE_ENVIRONMENT_DIGEST,
) -> TraceEvent:
    return event_factory(
        sequence,
        event_type=EventType.ACTION_PROPOSAL,
        phase=EventPhase.PRE_ACTION,
        payload={
            "action_identity": {
                "schema_version": "1.0",
                "kind": "opaque",
                "action_digest": action_digest,
                "workspace_digest": workspace_digest,
                "environment_digest": environment_digest,
                "identity_authority": identity_authority,
            }
        },
    )


def failure(
    event_factory: EventFactory,
    sequence: int,
    parent: TraceEvent,
    *,
    signature: str = "assertion mismatch",
) -> TraceEvent:
    return event_factory(
        sequence,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "failed",
                "exit_status": 1,
                "exception_type": "AssertionError",
                "failure_signature": signature,
            }
        },
        parent_ids=(parent.event_id,),
    )


def success(
    event_factory: EventFactory,
    sequence: int,
    parent: TraceEvent,
) -> TraceEvent:
    return event_factory(
        sequence,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "succeeded",
                "exit_status": 0,
            }
        },
        parent_ids=(parent.event_id,),
    )


def failed_test_event(
    event_factory: EventFactory,
    sequence: int,
    parent: TraceEvent,
    *,
    signature: str = "expected 1, got 2",
) -> TraceEvent:
    return event_factory(
        sequence,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "test_report": {
                "schema_version": "1.0",
                "framework": "pytest",
                "status": "failed",
                "failures": [
                    {
                        "schema_version": "1.0",
                        "test_id": "tests/test_api.py::test_value",
                        "failure_type": "AssertionError",
                        "signature": signature,
                    }
                ],
            }
        },
        parent_ids=(parent.event_id,),
    )


@pytest.mark.parametrize("window_events", [1, 10_001, 2.0, True])
def test_repetition_config_is_strict_and_bounded(window_events: object) -> None:
    with pytest.raises(ValidationError):
        RepetitionConfig(window_events=window_events)  # type: ignore[arg-type]


def test_repetition_config_is_frozen_and_detector_revalidates_it() -> None:
    config = RepetitionConfig(window_events=2)
    with pytest.raises(ValidationError):
        config.window_events = 3

    forged = RepetitionConfig.model_construct(window_events=1)
    with pytest.raises(DetectionInputError, match="input failed validation"):
        RepeatedActionDetector(forged)
    oversized_forge = RepetitionConfig.model_construct(window_events="secret" * 20_000)
    with pytest.raises(DetectionInputError) as error:
        RepeatedFailureDetector(oversized_forge)
    assert "secret" not in str(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    with pytest.raises(DetectionInputError, match="input failed validation"):
        RepeatedFailureDetector(cast(RepetitionConfig, {"window_events": 4}))

    detector = RepeatedActionDetector(RepetitionConfig(window_events=2))
    with pytest.raises(AttributeError):
        detector._config = RepetitionConfig(window_events=3)
    with pytest.raises(AttributeError):
        detector._detector_version = "tampered/v1"


def test_detector_versions_are_stable_and_bound_to_resolved_config() -> None:
    two = RepetitionConfig(window_events=2)
    three = RepetitionConfig(window_events=3)

    assert (
        RepeatedActionDetector(two).detector_version == RepeatedActionDetector(two).detector_version
    )
    assert (
        RepeatedActionDetector(two).detector_version
        != RepeatedActionDetector(three).detector_version
    )
    assert (
        RepeatedActionDetector(two).detector_version
        != RepeatedFailureDetector(two).detector_version
    )
    assert RepeatedActionDetector(two).config == two
    assert RepeatedFailureDetector(two).config == two


def test_repeated_action_uses_normalized_fingerprint_and_most_recent_match(
    event_factory: EventFactory,
) -> None:
    first = action(event_factory, 1, "pytest tests/test_api.py --quiet")
    second = action(event_factory, 2, "pytest\u2003--quiet\u2003tests/test_api.py")
    current = action(event_factory, 3, "pytest --quiet tests/test_api.py")

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=3)).evaluate(
        context(first, second, current)
    )

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.signal_type is SignalType.REPEATED_ACTION
    assert outcome.strength == 1.0
    assert outcome.evidence_event_ids == (first.event_id, current.event_id)


def test_repeated_action_detects_matching_exact_opaque_identities(
    event_factory: EventFactory,
) -> None:
    prior = opaque_action(event_factory, 1)
    current = opaque_action(event_factory, 2)

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(
        context(prior, current)
    )

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.signal_type is SignalType.REPEATED_ACTION
    assert outcome.evidence_event_ids == (prior.event_id, current.event_id)


@pytest.mark.parametrize(
    "changed_digest",
    (
        {"action_digest": "e" * 64},
        {"workspace_digest": "e" * 64},
        {"environment_digest": "e" * 64},
    ),
)
def test_repeated_action_compares_every_exact_opaque_digest(
    event_factory: EventFactory,
    changed_digest: dict[str, str],
) -> None:
    prior = opaque_action(event_factory, 1)
    current = opaque_action(event_factory, 2, **changed_digest)

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(
        context(prior, current)
    )

    assert outcome.status is DetectionStatus.NO_MATCH
    assert outcome.related_event_ids == (current.event_id,)


@pytest.mark.parametrize("identity_authority", ("coarse", "unavailable"))
def test_repeated_action_abstains_when_current_opaque_identity_is_not_exact(
    event_factory: EventFactory,
    identity_authority: str,
) -> None:
    prior = opaque_action(event_factory, 1)
    current = opaque_action(
        event_factory,
        2,
        identity_authority=identity_authority,
    )
    exact_current = opaque_action(event_factory, 2)

    detector = RepeatedActionDetector(RepetitionConfig(window_events=2))
    exact_outcome = detector.evaluate(context(prior, exact_current))
    outcome = detector.evaluate(context(prior, current))

    assert exact_outcome.status is DetectionStatus.DETECTED
    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.STRUCTURED_EVIDENCE_MISSING
    assert outcome.related_event_ids == (current.event_id,)


@pytest.mark.parametrize("identity_authority", ("coarse", "unavailable"))
def test_repeated_action_abstains_when_only_prior_opaque_identity_is_not_exact(
    event_factory: EventFactory,
    identity_authority: str,
) -> None:
    prior = opaque_action(
        event_factory,
        1,
        identity_authority=identity_authority,
    )
    current = opaque_action(event_factory, 2)

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(
        context(prior, current)
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.STRUCTURED_EVIDENCE_MISSING
    assert outcome.related_event_ids == (prior.event_id, current.event_id)


def test_non_exact_opaque_identity_does_not_suppress_an_exact_match(
    event_factory: EventFactory,
) -> None:
    exact_prior = opaque_action(event_factory, 1)
    unavailable = opaque_action(
        event_factory,
        2,
        identity_authority="unavailable",
    )
    current = opaque_action(event_factory, 3)

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=3)).evaluate(
        context(exact_prior, unavailable, current)
    )

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.evidence_event_ids == (exact_prior.event_id, current.event_id)


def test_repeated_action_does_not_equate_opaque_and_shell_evidence(
    event_factory: EventFactory,
) -> None:
    prior = action(event_factory, 1, OPAQUE_ACTION_DIGEST)
    current = opaque_action(event_factory, 2)

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(
        context(prior, current)
    )

    assert outcome.status is DetectionStatus.NO_MATCH


def test_repeated_action_window_includes_current_and_excludes_older_prefix(
    event_factory: EventFactory,
) -> None:
    prior = action(event_factory, 1)
    observation = event_factory(2)
    current = action(event_factory, 3)
    detection_context = context(prior, observation, current)

    excluded = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(detection_context)
    included = RepeatedActionDetector(RepetitionConfig(window_events=3)).evaluate(detection_context)

    assert excluded.status is DetectionStatus.NO_MATCH
    assert included.status is DetectionStatus.DETECTED
    assert included.evidence_event_ids == (prior.event_id, current.event_id)


def test_repeated_action_does_not_equate_changed_unsafe_flags(
    event_factory: EventFactory,
) -> None:
    prior = action(event_factory, 1, "rm -f build.cache")
    current = action(event_factory, 2, "rm -rf build.cache")

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(
        context(prior, current)
    )

    assert outcome.status is DetectionStatus.NO_MATCH


def test_repeated_action_abstains_on_unusable_structured_evidence(
    event_factory: EventFactory,
) -> None:
    malformed = event_factory(
        1,
        event_type=EventType.ACTION_PROPOSAL,
        payload={
            "action": {
                "schema_version": "1.0",
                "kind": "shell",
                "working_directory": "/workspace",
                "environment_digest": "a" * 64,
            }
        },
    )
    current = action(event_factory, 2, "pytest tests/other.py")
    malformed_current = event_factory(
        2,
        event_type=EventType.ACTION_PROPOSAL,
        phase=EventPhase.PRE_ACTION,
        payload={
            "action": {
                "schema_version": "1.0",
                "kind": "shell",
                "working_directory": "/workspace",
                "environment_digest": "a" * 64,
            }
        },
    )

    prior_outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(
        context(malformed, current)
    )
    current_outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(
        context(event_factory(1), malformed_current)
    )

    assert prior_outcome.status is DetectionStatus.ABSTAINED
    assert prior_outcome.abstention_reason is AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    assert current_outcome.status is DetectionStatus.ABSTAINED
    assert current_outcome.abstention_reason is AbstentionReason.STRUCTURED_EVIDENCE_INVALID


def test_repeated_action_requires_an_applicable_current_event(
    event_factory: EventFactory,
) -> None:
    current = event_factory(1, payload={"message": "pytest --quiet tests/test_api.py"})

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(context(current))

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.EVENT_NOT_APPLICABLE


def test_repeated_action_abstains_when_interception_is_already_too_late(
    event_factory: EventFactory,
) -> None:
    prior = action(event_factory, 1)
    current = action(event_factory, 2).model_copy(update={"phase": EventPhase.POST_ACTION})

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(
        context(prior, current)
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.PRE_ACTION_INTERCEPTION_UNAVAILABLE
    assert outcome.related_event_ids == (current.event_id,)


def test_repeated_action_with_no_prior_event_reports_insufficient_history(
    event_factory: EventFactory,
) -> None:
    current = action(event_factory, 1)

    outcome = RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(context(current))

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.INSUFFICIENT_HISTORY


def test_repeated_failure_requires_same_action_and_failure_fingerprints(
    event_factory: EventFactory,
) -> None:
    first_action = action(event_factory, 1, "pytest tests/test_api.py --quiet")
    first_failure = failure(event_factory, 2, first_action)
    current_action = action(event_factory, 3, "pytest --quiet tests/test_api.py")
    current_failure = failure(event_factory, 4, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=4)).evaluate(
        context(first_action, first_failure, current_action, current_failure)
    )

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.signal_type is SignalType.REPEATED_FAILURE
    assert outcome.strength == 1.0
    assert outcome.evidence_event_ids == (
        first_action.event_id,
        first_failure.event_id,
        current_action.event_id,
        current_failure.event_id,
    )


def test_repeated_failure_current_success_is_an_auditable_no_match(
    event_factory: EventFactory,
) -> None:
    current_action = action(event_factory, 1)
    current_success = success(event_factory, 2, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=2)).evaluate(
        context(current_action, current_success)
    )

    assert outcome.status is DetectionStatus.NO_MATCH
    assert outcome.related_event_ids == (current_success.event_id,)


def test_repeated_failure_with_one_event_reports_insufficient_history(
    event_factory: EventFactory,
) -> None:
    orphan_action = action(event_factory, 99)
    current_failure = failure(event_factory, 1, orphan_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=2)).evaluate(
        context(current_failure)
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.INSUFFICIENT_HISTORY


def test_repeated_failure_supports_versioned_test_reports(
    event_factory: EventFactory,
) -> None:
    first_action = action(event_factory, 1)
    first_failure = failed_test_event(event_factory, 2, first_action)
    current_action = action(event_factory, 3)
    current_failure = failed_test_event(event_factory, 4, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=4)).evaluate(
        context(first_action, first_failure, current_action, current_failure)
    )

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.evidence_event_ids == (
        first_action.event_id,
        first_failure.event_id,
        current_action.event_id,
        current_failure.event_id,
    )


def test_repeated_failure_uses_the_most_recent_unresolved_attempt(
    event_factory: EventFactory,
) -> None:
    action_one = action(event_factory, 1)
    failure_one = failure(event_factory, 2, action_one)
    action_two = action(event_factory, 3)
    failure_two = failure(event_factory, 4, action_two)
    action_three = action(event_factory, 5)
    failure_three = failure(event_factory, 6, action_three)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=6)).evaluate(
        context(
            action_one,
            failure_one,
            action_two,
            failure_two,
            action_three,
            failure_three,
        )
    )

    assert outcome.evidence_event_ids == (
        action_two.event_id,
        failure_two.event_id,
        action_three.event_id,
        failure_three.event_id,
    )


@pytest.mark.parametrize(
    ("current_command", "current_signature"),
    [
        ("pytest tests/test_other.py", "assertion mismatch"),
        ("pytest --quiet tests/test_api.py", "different assertion"),
    ],
)
def test_near_but_different_failure_or_action_is_not_a_repeat(
    event_factory: EventFactory,
    current_command: str,
    current_signature: str,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    current_action = action(event_factory, 3, current_command)
    current_failure = failure(
        event_factory,
        4,
        current_action,
        signature=current_signature,
    )

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=4)).evaluate(
        context(prior_action, prior_failure, current_action, current_failure)
    )

    assert outcome.status is DetectionStatus.NO_MATCH


def test_structured_success_of_same_action_resolves_the_prior_failure(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    resolved_action = action(event_factory, 3)
    resolved_success = success(event_factory, 4, resolved_action)
    current_action = action(event_factory, 5)
    current_failure = failure(event_factory, 6, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=6)).evaluate(
        context(
            prior_action,
            prior_failure,
            resolved_action,
            resolved_success,
            current_action,
            current_failure,
        )
    )

    assert outcome.status is DetectionStatus.NO_MATCH


def test_success_for_the_current_action_makes_later_failure_ambiguous(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    current_action = action(event_factory, 3)
    contradictory_success = success(event_factory, 4, current_action)
    current_failure = failure(event_factory, 5, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=5)).evaluate(
        context(
            prior_action,
            prior_failure,
            current_action,
            contradictory_success,
            current_failure,
        )
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    assert outcome.related_event_ids == (
        contradictory_success.event_id,
        current_failure.event_id,
    )


def test_parallel_same_fingerprint_success_does_not_resolve_current_attempt(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    current_action = action(event_factory, 3)
    parallel_action = action(event_factory, 4)
    parallel_success = success(event_factory, 5, parallel_action)
    current_failure = failure(event_factory, 6, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=6)).evaluate(
        context(
            prior_action,
            prior_failure,
            current_action,
            parallel_action,
            parallel_success,
            current_failure,
        )
    )

    assert outcome.status is DetectionStatus.DETECTED
    assert outcome.evidence_event_ids == (
        prior_action.event_id,
        prior_failure.event_id,
        current_action.event_id,
        current_failure.event_id,
    )


def test_delayed_success_for_an_already_failed_action_is_not_a_resolution(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    contradictory_success = success(event_factory, 3, prior_action)
    current_action = action(event_factory, 4)
    current_failure = failure(event_factory, 5, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=5)).evaluate(
        context(
            prior_action,
            prior_failure,
            contradictory_success,
            current_action,
            current_failure,
        )
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.STRUCTURED_EVIDENCE_INVALID
    assert outcome.related_event_ids == (
        contradictory_success.event_id,
        current_failure.event_id,
    )


def test_unlinked_success_cannot_resolve_a_prior_failure(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    unlinked_success = event_factory(
        3,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "succeeded",
                "exit_status": 0,
            }
        },
    )
    current_action = action(event_factory, 4)
    current_failure = failure(event_factory, 5, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=5)).evaluate(
        context(
            prior_action,
            prior_failure,
            unlinked_success,
            current_action,
            current_failure,
        )
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.PARENT_ACTION_MISSING
    assert outcome.related_event_ids == (
        unlinked_success.event_id,
        current_failure.event_id,
    )


def test_unlinked_success_during_current_action_is_audited_precisely(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    current_action = action(event_factory, 3)
    unlinked_success = event_factory(
        4,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "succeeded",
                "exit_status": 0,
            }
        },
    )
    current_failure = failure(event_factory, 5, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=5)).evaluate(
        context(
            prior_action,
            prior_failure,
            current_action,
            unlinked_success,
            current_failure,
        )
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.PARENT_ACTION_MISSING
    assert outcome.related_event_ids == (
        unlinked_success.event_id,
        current_failure.event_id,
    )


def test_success_of_a_different_action_does_not_resolve_the_failure(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    other_action = action(event_factory, 3, "python -m compileall src")
    other_success = success(event_factory, 4, other_action)
    current_action = action(event_factory, 5)
    current_failure = failure(event_factory, 6, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=6)).evaluate(
        context(
            prior_action,
            prior_failure,
            other_action,
            other_success,
            current_action,
            current_failure,
        )
    )

    assert outcome.status is DetectionStatus.DETECTED


def test_repeated_failure_window_excludes_an_older_failure(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    current_action = action(event_factory, 3)
    current_failure = failure(event_factory, 4, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=2)).evaluate(
        context(prior_action, prior_failure, current_action, current_failure)
    )

    assert outcome.status is DetectionStatus.NO_MATCH


def test_repeated_failure_abstains_on_missing_or_ambiguous_parent_action(
    event_factory: EventFactory,
) -> None:
    first_action = action(event_factory, 1)
    second_action = action(event_factory, 2)
    ambiguous_failure = event_factory(
        3,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "failed",
                "exit_status": 1,
                "failure_signature": "assertion mismatch",
            }
        },
        parent_ids=(first_action.event_id, second_action.event_id),
    )

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=3)).evaluate(
        context(first_action, second_action, ambiguous_failure)
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.AMBIGUOUS_PARENT_ACTION


def test_repeated_failure_requires_a_parent_that_is_actually_an_action(
    event_factory: EventFactory,
) -> None:
    observation = event_factory(1)
    current_failure = event_factory(
        2,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "failed",
                "exit_status": 1,
            }
        },
        parent_ids=(observation.event_id,),
    )

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=2)).evaluate(
        context(observation, current_failure)
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.PARENT_ACTION_MISSING


def test_repeated_failure_abstains_when_matching_prior_linkage_is_ambiguous(
    event_factory: EventFactory,
) -> None:
    unrelated = event_factory(1)
    prior_failure = event_factory(
        2,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "failed",
                "exit_status": 1,
                "exception_type": "AssertionError",
                "failure_signature": "assertion mismatch",
            }
        },
    )
    current_action = action(event_factory, 3)
    current_failure = failure(event_factory, 4, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=4)).evaluate(
        context(unrelated, prior_failure, current_action, current_failure)
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.PARENT_ACTION_MISSING


def test_repeated_failure_fails_closed_when_any_parent_is_outside_window(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    current_action = action(event_factory, 3)
    unknown_parent = event_factory(99)
    current_failure = event_factory(
        4,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "failed",
                "exit_status": 1,
                "exception_type": "AssertionError",
                "failure_signature": "assertion mismatch",
            }
        },
        parent_ids=(current_action.event_id, unknown_parent.event_id),
    )

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=4)).evaluate(
        context(prior_action, prior_failure, current_action, current_failure)
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.PARENT_ACTION_MISSING


def test_repeated_failure_never_uses_or_cites_an_action_outside_its_window(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    current_action = action(event_factory, 3)
    current_failure = failure(event_factory, 4, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=3)).evaluate(
        context(prior_action, prior_failure, current_action, current_failure)
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.PARENT_ACTION_MISSING


def test_repeated_failure_does_not_infer_failure_from_free_text(
    event_factory: EventFactory,
) -> None:
    current = event_factory(
        1,
        event_type=EventType.OBSERVATION,
        payload={"message": "AssertionError: assertion mismatch"},
    )

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=2)).evaluate(context(current))

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.STRUCTURED_EVIDENCE_MISSING


def test_repeated_failure_rejects_competing_structured_outcomes(
    event_factory: EventFactory,
) -> None:
    current = event_factory(
        1,
        event_type=EventType.TOOL_COMPLETION,
        payload={
            "tool_outcome": {
                "schema_version": "1.0",
                "status": "failed",
                "exit_status": 1,
            },
            "test_report": {
                "schema_version": "1.0",
                "framework": "pytest",
                "status": "passed",
                "failures": [],
            },
        },
    )

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=2)).evaluate(context(current))

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.STRUCTURED_EVIDENCE_INVALID


def test_repeated_failure_fails_closed_on_malformed_intervening_outcome(
    event_factory: EventFactory,
) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    malformed = event_factory(
        3,
        event_type=EventType.TOOL_COMPLETION,
        payload={"tool_outcome": {"status": "succeeded", "exit_status": 0}},
    )
    current_action = action(event_factory, 4)
    current_failure = failure(event_factory, 5, current_action)

    outcome = RepeatedFailureDetector(RepetitionConfig(window_events=5)).evaluate(
        context(prior_action, prior_failure, malformed, current_action, current_failure)
    )

    assert outcome.status is DetectionStatus.ABSTAINED
    assert outcome.abstention_reason is AbstentionReason.STRUCTURED_EVIDENCE_INVALID


def test_repeated_failure_signal_is_replay_stable(event_factory: EventFactory) -> None:
    prior_action = action(event_factory, 1)
    prior_failure = failure(event_factory, 2, prior_action)
    current_action = action(event_factory, 3)
    current_failure = failure(event_factory, 4, current_action)
    detection_context = context(prior_action, prior_failure, current_action, current_failure)
    detector = RepeatedFailureDetector(RepetitionConfig(window_events=4))
    extractor = DeterministicSignalExtractor((detector,))

    first = extractor.extract(detection_context)
    second = extractor.extract(detection_context)

    assert first == second
    assert len(first) == 1
    assert first[0].detector_version == detector.detector_version
    assert first[0].evidence_event_ids == (
        prior_action.event_id,
        prior_failure.event_id,
        current_action.event_id,
        current_failure.event_id,
    )


def test_repetition_detectors_revalidate_the_exact_context_boundary(
    event_factory: EventFactory,
) -> None:
    current = action(event_factory, 1)

    with pytest.raises(DetectionInputError, match="input failed validation"):
        RepeatedActionDetector(RepetitionConfig(window_events=2)).evaluate(
            cast(DetectionContext, {"run_id": current.run_id, "events": (current,)})
        )
