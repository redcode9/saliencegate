from __future__ import annotations

from dataclasses import replace

import pytest
from tests.shadow.test_analyzer import _memory_session
from tests.shadow.test_trace import ENVIRONMENT_DIGEST, build_trace

import saliencegate.shadow.analyzer as analyzer_module
import saliencegate.signals.base as signal_base_module
from saliencegate.domain import SignalType, canonical_json
from saliencegate.shadow import ShadowSession
from saliencegate.shadow.errors import ShadowInvariantError
from saliencegate.shadow.evaluation import evaluate_shadow_heuristic
from saliencegate.shadow.inputs import derive_shadow_source_event_digest
from saliencegate.shadow.observation import (
    _admit_shadow_observation_sequence,
    _build_shadow_observation_trusted,
    build_shadow_observation,
    derive_shadow_detection_context_digest,
    derive_shadow_event_prefix_digest,
    derive_shadow_extraction_report_digest,
    derive_shadow_feature_snapshot_digest,
    derive_shadow_observation_digest,
    derive_shadow_redacted_event_digest,
    select_detection_context,
)
from saliencegate.shadow.trace import ShadowTrace
from saliencegate.signals.base import (
    _admit_detection_sequence,
    _DetectionSequenceProof,
    _extract_trusted_report,
    _longest_trusted_detection_context,
)

_PRIVATE_SENTINEL = "fixture-secret-trusted-observation"
_DIGEST_FIELDS = (
    "source_event_digest",
    "event_prefix_digest",
    "detection_context_digest",
    "redacted_event_digest",
    "detector_profile_digest",
    "evaluator_configuration_digest",
    "extraction_report_digest",
    "feature_snapshot_digest",
    "observation_digest",
)


def _mixed_trace() -> ShadowTrace:
    command = f"pytest -q tests/{_PRIVATE_SENTINEL}.py"
    records: list[dict[str, object]] = [
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_start",
            "source_event_id": "mixed-start",
            "occurred_at": "2026-07-17T09:00:00Z",
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "action",
            "source_event_id": "mixed-action-1",
            "occurred_at": "2026-07-17T09:00:01Z",
            "command": command,
            "working_directory": "/private/project",
            "environment_digest": ENVIRONMENT_DIGEST,
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "tool_result",
            "source_event_id": "mixed-tool-1",
            "occurred_at": "2026-07-17T09:00:02Z",
            "action_source_event_id": "mixed-action-1",
            "status": "failed",
            "exit_status": 1,
            "error_code": "TEST_FAILURE",
            "failure_signature": "mixed-repeat-failure",
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "action",
            "source_event_id": "mixed-action-2",
            "occurred_at": "2026-07-17T09:00:03Z",
            "command": command,
            "working_directory": "/private/project",
            "environment_digest": ENVIRONMENT_DIGEST,
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "tool_result",
            "source_event_id": "mixed-tool-2",
            "occurred_at": "2026-07-17T09:00:04Z",
            "action_source_event_id": "mixed-action-2",
            "status": "failed",
            "exit_status": 1,
            "error_code": "TEST_FAILURE",
            "failure_signature": "mixed-repeat-failure",
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "test_result",
            "source_event_id": "mixed-test-1",
            "occurred_at": "2026-07-17T09:00:05Z",
            "action_source_event_id": "mixed-action-2",
            "framework": "pytest",
            "status": "failed",
            "failures": [
                {
                    "schema_version": "1.0",
                    "test_id": "tests/unit.py::test_mixed",
                    "failure_type": "AssertionError",
                    "signature": "mixed-repeat-failure",
                }
            ],
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "controller_error",
            "source_event_id": "mixed-controller-1",
            "occurred_at": "2026-07-17T09:00:06Z",
            "error_code": "controller_timeout",
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_end",
            "source_event_id": "mixed-finish",
            "occurred_at": "2026-07-17T09:00:07Z",
        },
    ]
    return build_trace(records)


def _prepare(
    session: ShadowSession,
    trace: ShadowTrace,
) -> analyzer_module._PreparedAnalysis:
    prepared = analyzer_module._prepare_analysis(session, trace)
    assert type(prepared) is analyzer_module._PreparedAnalysis
    return prepared


def _assert_prefix_parity(
    session: ShadowSession,
    prepared: analyzer_module._PreparedAnalysis,
    sequence: _DetectionSequenceProof,
    end_ordinal: int,
) -> set[SignalType]:
    item = prepared.events[end_ordinal - 1]
    prefix = sequence.events[:end_ordinal]
    trusted_context = _longest_trusted_detection_context(sequence, end_ordinal)
    trusted_extraction = _extract_trusted_report(session._extractor, trusted_context)
    admission = _admit_shadow_observation_sequence(
        sequence,
        config=session._config,
        redaction_policy_tag=session._redaction_policy_tag,
    )
    trusted_observation = _build_shadow_observation_trusted(
        admission,
        trusted_extraction,
        input_kind=item.row.input_kind,
        source_event_digest=item.row.source_event_digest,
        cli_input_ordinal=item.row.input_ordinal,
    )

    public_context = select_detection_context(prefix)
    public_report = session._extract_report(public_context)
    public_feature_digest = derive_shadow_feature_snapshot_digest(
        prefix=prefix,
        context=public_context,
        report=public_report,
        config=session._config,
    )
    public_heuristic = evaluate_shadow_heuristic(
        public_report,
        input_kind=item.row.input_kind,
        config=session._config,
        feature_snapshot_digest=public_feature_digest,
    )
    public_observation = build_shadow_observation(
        prefix=prefix,
        context=public_context,
        report=public_report,
        config=session._config,
        input_kind=item.row.input_kind,
        heuristic=public_heuristic,
        source_event_digest=item.row.source_event_digest,
        redaction_policy_tag=session._redaction_policy_tag,
        cli_input_ordinal=item.row.input_ordinal,
    )

    assert trusted_context.start_index == len(prefix) - len(public_context.events)
    assert trusted_context.end_ordinal == end_ordinal
    assert trusted_context.context == public_context
    assert canonical_json(trusted_context.context) == canonical_json(public_context)
    assert trusted_extraction.report == public_report == item.extraction_report
    assert canonical_json(trusted_extraction.report) == canonical_json(public_report)
    assert trusted_extraction.report.model_dump_json(
        warnings=False
    ) == public_report.model_dump_json(warnings=False)
    assert trusted_observation == public_observation == item.observation
    assert canonical_json(trusted_observation) == canonical_json(public_observation)
    assert trusted_observation.model_dump_json(
        warnings=False
    ) == public_observation.model_dump_json(warnings=False)
    assert all(
        getattr(trusted_observation, field_name) == getattr(public_observation, field_name)
        for field_name in _DIGEST_FIELDS
    )
    assert trusted_observation.source_event_digest == derive_shadow_source_event_digest(
        prefix[-1].run_id,
        prefix[-1].source_event_id,
    )
    assert trusted_observation.event_prefix_digest == derive_shadow_event_prefix_digest(prefix)
    assert trusted_observation.detection_context_digest == derive_shadow_detection_context_digest(
        public_context
    )
    assert trusted_observation.redacted_event_digest == derive_shadow_redacted_event_digest(
        prefix[-1]
    )
    assert trusted_observation.extraction_report_digest == derive_shadow_extraction_report_digest(
        public_report
    )
    assert trusted_observation.feature_snapshot_digest == public_feature_digest
    assert trusted_observation.observation_digest == derive_shadow_observation_digest(
        public_observation
    )
    assert trusted_observation.redaction_policy_tag == session._redaction_policy_tag
    assert trusted_observation.detector_profile_digest == session._config.detector_profile_digest
    assert (
        trusted_observation.evaluator_configuration_digest
        == session._config.evaluator_configuration_digest
    )
    assert trusted_observation.heuristic_evaluations == (public_heuristic,)
    return {signal.signal_type for signal in trusted_extraction.report.signals}


def _assert_sanitized(error: ShadowInvariantError) -> None:
    assert _PRIVATE_SENTINEL not in str(error)
    assert _PRIVATE_SENTINEL not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_trusted_analyzer_observations_match_the_public_slow_path_for_every_prefix() -> None:
    trace = _mixed_trace()
    session = _memory_session(trace)
    prepared = _prepare(session, trace)
    sequence = _admit_detection_sequence(prepared.expected_events)
    detected_types: set[SignalType] = set()

    for end_ordinal in range(1, len(sequence.events) + 1):
        detected_types.update(_assert_prefix_parity(session, prepared, sequence, end_ordinal))

    assert len(sequence.events) == len(trace.records) == 8
    assert detected_types == set(session._config.supported_signal_types)


def test_trusted_and_public_paths_match_at_a_real_longest_suffix_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _mixed_trace()
    session = _memory_session(trace)
    baseline = _prepare(session, trace)
    baseline_sequence = _admit_detection_sequence(baseline.expected_events)
    costs = baseline_sequence._event_costs
    bounded_cost = max(max(costs), sum(costs[-3:]))
    assert bounded_cost < sum(costs)
    monkeypatch.setattr(
        signal_base_module,
        "_MAX_CONTEXT_SIZE_UPPER_BOUND",
        bounded_cost,
    )

    prepared = _prepare(session, trace)
    sequence = _admit_detection_sequence(prepared.expected_events)
    end_ordinal = len(sequence.events)
    trusted_context = _longest_trusted_detection_context(sequence, end_ordinal)

    _assert_prefix_parity(session, prepared, sequence, end_ordinal)

    assert trusted_context.start_index > 0
    assert len(trusted_context.context.events) < len(sequence.events)
    selected_cost = (
        sequence._prefix_costs[end_ordinal] - sequence._prefix_costs[trusted_context.start_index]
    )
    extended_cost = (
        sequence._prefix_costs[end_ordinal]
        - sequence._prefix_costs[trusted_context.start_index - 1]
    )
    assert selected_cost <= bounded_cost < extended_cost
    assert prepared.events[-1].observation.context_truncated is True


def test_observation_admission_rejects_sequence_token_tampering_sanitized() -> None:
    trace = _mixed_trace()
    session = _memory_session(trace)
    prepared = _prepare(session, trace)
    sequence = _admit_detection_sequence(prepared.expected_events)
    damaged = replace(sequence, _token=object())

    with pytest.raises(ShadowInvariantError) as captured:
        _admit_shadow_observation_sequence(
            damaged,
            config=session._config,
            redaction_policy_tag=session._redaction_policy_tag,
        )

    _assert_sanitized(captured.value)


@pytest.mark.parametrize(
    "tampering",
    (
        "admission_token",
        "admission_config",
        "admission_tag",
        "extraction_token",
        "cross_sequence",
    ),
)
def test_trusted_observation_builder_rejects_tampering_sanitized(tampering: str) -> None:
    trace = _mixed_trace()
    session = _memory_session(trace)
    prepared = _prepare(session, trace)
    sequence = _admit_detection_sequence(prepared.expected_events)
    admission = _admit_shadow_observation_sequence(
        sequence,
        config=session._config,
        redaction_policy_tag=session._redaction_policy_tag,
    )
    item = prepared.events[-1]
    trusted_context = _longest_trusted_detection_context(sequence, len(sequence.events))
    extraction = _extract_trusted_report(session._extractor, trusted_context)
    if tampering == "admission_token":
        admission = replace(admission, _token=object())
    elif tampering == "admission_config":
        copied_config = type(admission.config).model_validate_json(
            admission.config.model_dump_json(warnings=False)
        )
        admission = replace(admission, config=copied_config)
    elif tampering == "admission_tag":
        copied_tag = type(admission.redaction_policy_tag).model_validate_json(
            admission.redaction_policy_tag.model_dump_json(warnings=False)
        )
        admission = replace(admission, redaction_policy_tag=copied_tag)
    elif tampering == "extraction_token":
        extraction = replace(extraction, _token=object())
    else:
        second_sequence = _admit_detection_sequence(prepared.expected_events)
        second_context = _longest_trusted_detection_context(
            second_sequence,
            len(second_sequence.events),
        )
        extraction = _extract_trusted_report(session._extractor, second_context)

    with pytest.raises(ShadowInvariantError) as captured:
        _build_shadow_observation_trusted(
            admission,
            extraction,
            input_kind=item.row.input_kind,
            source_event_digest=item.row.source_event_digest,
            cli_input_ordinal=item.row.input_ordinal,
        )

    _assert_sanitized(captured.value)
