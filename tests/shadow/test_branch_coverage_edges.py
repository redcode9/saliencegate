"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from uuid import UUID

import pytest
from tests.shadow.conftest import TraceEventFactory
from tests.shadow.test_atif_coverage_edges import _trace as build_atif_trace
from tests.shadow.test_observation import make_case, no_match_report, replace_event
from tests.shadow.test_trace import RUN_ID, build_trace, complete_records

import saliencegate.shadow.atif as atif_module
import saliencegate.shadow.observation as observation_module
import saliencegate.shadow.session as session_module
import saliencegate.shadow.trace as trace_module
from saliencegate.domain import TraceEvent, canonical_json
from saliencegate.security import InstallationKey
from saliencegate.security.redaction import RedactionPolicy
from saliencegate.shadow.atif import ATIFProfile, ShadowEnvironmentBinding
from saliencegate.shadow.config import ShadowConfig
from saliencegate.shadow.errors import (
    ShadowInputError,
    ShadowTraceInputError,
)
from saliencegate.shadow.inputs import ShadowStartInput, derive_shadow_event_id
from saliencegate.signals import DetectionContext, DetectorEvaluation, ExtractionReport

_KEY = InstallationKey(b"m" * 32)
_ENVIRONMENT_DIGEST = "a" * 64
_TIMESTAMP = "2026-07-17T09:00:00Z"


def _prepare_options(**overrides: object) -> session_module._SessionOptions:
    arguments: dict[str, object] = {
        "run_id": UUID("99999999-9999-4999-8999-999999999999"),
        "config": ShadowConfig.reference(),
        "installation_key": _KEY,
        "redaction_policy": RedactionPolicy(),
        "capture_scope": "unknown",
        "task_scope_digest": None,
        "lineage_scope_digest": None,
        "capture_manifest_digest": None,
        "source_adapter": "coverage-shadow/v1",
    }
    arguments.update(overrides)
    return session_module._prepare_options(**arguments)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"run_id": UUID(int=0)}, "run identity"),
        ({"capture_scope": "everything"}, "capture scope"),
        ({"source_adapter": "SALIENCEGATE.REPOSITORY"}, "reserved"),
        ({"task_scope_digest": "g" * 64}, "digest"),
    ),
)
def test_session_option_boundaries_reject_each_invalid_identity(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare_options(**overrides)


def test_session_policy_preflight_reaches_the_metadata_identity_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = iter((True, True, False))
    monkeypatch.setattr(
        session_module,
        "_marker_is_redaction_identity",
        lambda *_args: next(decisions),
    )

    with pytest.raises(ValueError, match="metadata"):
        _prepare_options()


def test_session_input_copy_rejects_unknown_and_drifted_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ShadowInputError):
        session_module._copy_input(object())

    supplied = ShadowStartInput(
        source_event_id="coverage-start",
        occurred_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    drifted = ShadowStartInput(
        source_event_id="coverage-drift",
        occurred_at=datetime(2026, 7, 17, tzinfo=UTC),
    )

    def drift_copy(_cls: type[ShadowStartInput], _value: object) -> ShadowStartInput:
        return drifted

    monkeypatch.setattr(ShadowStartInput, "model_validate", classmethod(drift_copy))
    with pytest.raises(ShadowInputError):
        session_module._copy_input(supplied)


def test_observation_prefix_rejects_nonmonotonic_and_duplicate_events(
    trace_event_factory: TraceEventFactory,
) -> None:
    first = trace_event_factory(1)
    second = trace_event_factory(2)
    nonmonotonic = replace_event(second, timestamp=first.timestamp - timedelta(microseconds=1))
    duplicate = replace_event(second, source_event_id=first.source_event_id)

    with pytest.raises(ValueError, match="timestamps are not monotonic"):
        observation_module._copy_event_prefix((first, nonmonotonic))
    with pytest.raises(ValueError, match="identities are not unique"):
        observation_module._copy_event_prefix((first, duplicate))


def test_observation_exact_copy_rejects_defensive_copy_drift(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = trace_event_factory(1)
    drifted = trace_event_factory(2)

    def drift_copy(_cls: type[TraceEvent], _value: object) -> TraceEvent:
        return drifted

    monkeypatch.setattr(TraceEvent, "model_validate_json", classmethod(drift_copy))
    with pytest.raises(ValueError, match="defensive validation"):
        observation_module._copy_exact_model(TraceEvent, first)


def test_observation_feature_validation_rejects_context_and_report_mismatches(
    trace_event_factory: TraceEventFactory,
) -> None:
    case = make_case(trace_event_factory)
    context_not_ending_at_current = DetectionContext(
        run_id=case.prefix[0].run_id,
        events=(case.prefix[0],),
    )
    with pytest.raises(ValueError, match="does not end"):
        observation_module._validate_feature_inputs(
            prefix=case.prefix,
            context=context_not_ending_at_current,
            report=case.report,
            config=case.config,
        )

    first, current = case.prefix
    changed_first = replace_event(first, payload={"changed": True})
    nonsuffix_context = DetectionContext(
        run_id=current.run_id,
        events=(changed_first, current),
    )
    with pytest.raises(ValueError, match="not a suffix"):
        observation_module._validate_feature_inputs(
            prefix=case.prefix,
            context=nonsuffix_context,
            report=case.report,
            config=case.config,
        )

    other_current = replace_event(
        current,
        source_event_id="coverage-other-current",
        event_id=derive_shadow_event_id(current.run_id, "coverage-other-current"),
    )
    other_prefix = (first, other_current)
    other_report = no_match_report(other_prefix, case.config)
    with pytest.raises(ValueError, match="does not identify"):
        observation_module._validate_feature_inputs(
            prefix=case.prefix,
            context=case.context,
            report=other_report,
            config=case.config,
        )


def test_observation_feature_validation_rejects_detector_shape_drift(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_case(trace_event_factory)
    missing_evaluation = ExtractionReport(
        run_id=case.report.run_id,
        current_event_id=case.report.current_event_id,
        current_event_timestamp=case.report.current_event_timestamp,
        evaluations=case.report.evaluations[:-1],
        signals=(),
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            observation_module,
            "_copy_extraction_report",
            lambda value: value,
        )
        with pytest.raises(ValueError, match="does not cover"):
            observation_module._validate_feature_inputs(
                prefix=case.prefix,
                context=case.context,
                report=missing_evaluation,
                config=case.config,
            )

    evaluations = list(case.report.evaluations)
    first = evaluations[0]
    evaluations[0] = DetectorEvaluation(
        signal_type=first.signal_type,
        detector_version="coverage-version/v2",
        outcome=first.outcome,
    )
    wrong_version = ExtractionReport(
        run_id=case.report.run_id,
        current_event_id=case.report.current_event_id,
        current_event_timestamp=case.report.current_event_timestamp,
        evaluations=tuple(evaluations),
        signals=(),
    )
    with pytest.raises(ValueError, match="versions do not match"):
        observation_module._validate_feature_inputs(
            prefix=case.prefix,
            context=case.context,
            report=wrong_version,
            config=case.config,
        )


def _environment() -> ShadowEnvironmentBinding:
    return ShadowEnvironmentBinding(
        default_working_directory="/synthetic/coverage",
        environment_digest=_ENVIRONMENT_DIGEST,
    )


def _action_context(
    profile: ATIFProfile,
) -> tuple[
    atif_module._ATIFProfileContract,
    ShadowEnvironmentBinding,
    dict[str, object],
    MappingProxyType[str, object],
]:
    environment = _environment()
    source_event_id = "atif-s00000001-c0001-action"
    if profile is ATIFProfile.HARBOR_TERMINUS_2_V1:
        semantics = dict(atif_module._TERMINUS_EXECUTION_SEMANTICS)
        digest = atif_module._terminus_environment_digest(environment.environment_digest)
    else:
        semantics = {}
        digest = atif_module._codex_environment_digest(environment.environment_digest, semantics)
    context: dict[str, object] = {
        "source_event_id": source_event_id,
        "working_directory": environment.default_working_directory,
        "execution_semantics": semantics,
        "environment_digest": digest,
    }
    action = MappingProxyType(
        {
            "schema_version": "shadow-input/v1",
            "kind": "action",
            "source_event_id": source_event_id,
            "occurred_at": "2000-01-01T00:00:00.000001Z",
            "command": "printf coverage",
            "working_directory": environment.default_working_directory,
            "environment_digest": digest,
        }
    )
    return atif_module._PROFILE_CONTRACTS[profile], environment, context, action


def test_atif_action_context_rejects_shape_and_common_identity_mismatches() -> None:
    contract, environment, context, action = _action_context(ATIFProfile.HARBOR_CODEX_V1)

    assert not atif_module._context_matches_action(
        contract=contract,
        environment=environment,
        context=[],
        action_record=action,
    )
    assert not atif_module._context_matches_action(
        contract=contract,
        environment=environment,
        context=context,
        action_record=MappingProxyType({**action, "unexpected": True}),
    )
    assert not atif_module._context_matches_action(
        contract=contract,
        environment=environment,
        context={**context, "source_event_id": "not-an-action-coordinate"},
        action_record=action,
    )


def test_atif_terminus_action_context_rejects_semantics_and_directory_drift() -> None:
    contract, environment, context, action = _action_context(ATIFProfile.HARBOR_TERMINUS_2_V1)

    assert not atif_module._context_matches_action(
        contract=contract,
        environment=environment,
        context={**context, "execution_semantics": {}},
        action_record=action,
    )

    changed_directory = "/synthetic/elsewhere"
    assert not atif_module._context_matches_action(
        contract=contract,
        environment=environment,
        context={**context, "working_directory": changed_directory},
        action_record=MappingProxyType({**action, "working_directory": changed_directory}),
    )


def test_atif_codex_action_context_rejects_unknown_and_ill_typed_semantics() -> None:
    contract, environment, context, action = _action_context(ATIFProfile.HARBOR_CODEX_V1)

    for semantics in ({"unknown": True}, {"login": 1}, {"shell": object()}):
        assert not atif_module._context_matches_action(
            contract=contract,
            environment=environment,
            context={**context, "execution_semantics": semantics},
            action_record=action,
        )


def test_atif_step_collection_helpers_reject_each_non_list_shape() -> None:
    with pytest.raises(ShadowTraceInputError):
        atif_module._tool_calls({"tool_calls": {}}, ordinal=1)
    with pytest.raises(ShadowTraceInputError):
        atif_module._results({"observation": []}, ordinal=2)
    assert atif_module._results({"observation": {}}, ordinal=3) == []
    with pytest.raises(ShadowTraceInputError):
        atif_module._results({"observation": {"results": ()}}, ordinal=4)


def test_atif_call_classifiers_reject_missing_exact_fields() -> None:
    environment = _environment()

    with pytest.raises(ShadowTraceInputError):
        atif_module._classify_terminus_call(
            {"function_name": 1},
            step_ordinal=1,
            call_ordinal=1,
            tool_call_id="call",
            timestamp=None,
            environment=environment,
        )
    with pytest.raises(ShadowTraceInputError):
        atif_module._classify_terminus_call(
            {"function_name": "bash_command", "arguments": {"keystrokes": "pwd\n"}},
            step_ordinal=1,
            call_ordinal=1,
            tool_call_id=None,
            timestamp=None,
            environment=environment,
        )
    with pytest.raises(ShadowTraceInputError):
        atif_module._classify_codex_call(
            {"function_name": 1},
            step_ordinal=1,
            call_ordinal=1,
            tool_call_id="call",
            timestamp=None,
            environment=environment,
        )
    with pytest.raises(ShadowTraceInputError):
        atif_module._classify_codex_call(
            {"function_name": "exec_command", "arguments": {"cmd": "pwd"}},
            step_ordinal=1,
            call_ordinal=1,
            tool_call_id=None,
            timestamp=None,
            environment=environment,
        )


def test_atif_optional_arguments_and_exit_metadata_reject_alias_shapes() -> None:
    with pytest.raises(ShadowTraceInputError):
        atif_module._validate_codex_optional_arguments({"login": 1}, step=1, call=1)
    assert (
        atif_module._codex_exit_status(
            {"extra": []},
            tool_call_id="call",
            total_calls=1,
            step_ordinal=1,
            call_ordinal=1,
        )
        is None
    )


def _number(value: int) -> atif_module._JSONNumber:
    return atif_module._JSONNumber(str(value), True)


def _codex_root(step: object, **overrides: object) -> dict[str, object]:
    root: dict[str, object] = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "codex"},
        "steps": [step],
    }
    root.update(overrides)
    return root


def _codex_step(
    *,
    call: object | None = None,
    results: list[object] | None = None,
    source: object = "agent",
    extra: object | None = None,
) -> dict[str, object]:
    selected_call = (
        {
            "tool_call_id": "coverage-call",
            "function_name": "exec_command",
            "arguments": {"cmd": "printf coverage"},
        }
        if call is None
        else call
    )
    step: dict[str, object] = {
        "step_id": _number(1),
        "source": source,
        "tool_calls": [selected_call],
    }
    if results is not None:
        step["observation"] = {"results": results}
    if extra is not None:
        step["extra"] = extra
    return step


@pytest.mark.parametrize(
    "root",
    (
        _codex_root({}, agent=[]),
        _codex_root(_codex_step(), subagent_trajectories=object()),
        _codex_root(1),
        _codex_root(_codex_step(call=1)),
        _codex_root(
            _codex_step(
                call={
                    "tool_call_id": 1,
                    "function_name": "exec_command",
                    "arguments": {"cmd": "pwd"},
                }
            )
        ),
        _codex_root(
            _codex_step(
                call={"tool_call_id": "call", "function_name": 1},
                source="user",
            )
        ),
        _codex_root(
            _codex_step(
                call={"tool_call_id": "call", "function_name": "other"},
                source="user",
            )
        ),
        _codex_root(_codex_step(results=[1])),
        _codex_root(_codex_step(results=[{"source_call_id": 1}])),
    ),
)
def test_atif_mapping_plan_rejects_each_ambiguous_container_or_reference(
    root: dict[str, object],
) -> None:
    with pytest.raises(ShadowTraceInputError):
        atif_module._plan_mapping(
            root,
            contract=atif_module._PROFILE_CONTRACTS[ATIFProfile.HARBOR_CODEX_V1],
            environment=_environment(),
        )


def test_atif_mapping_plan_rejects_missing_tool_id_and_duplicate_outcomes() -> None:
    missing_id = _codex_root(
        _codex_step(call={"function_name": "exec_command", "arguments": {"cmd": "pwd"}})
    )
    with pytest.raises(ShadowTraceInputError):
        atif_module._plan_mapping(
            missing_id,
            contract=atif_module._PROFILE_CONTRACTS[ATIFProfile.HARBOR_CODEX_V1],
            environment=_environment(),
        )

    result = {"source_call_id": "coverage-call"}
    duplicate_outcomes = _codex_root(
        _codex_step(
            results=[result, dict(result)],
            extra={"tool_metadata": {"exit_code": _number(0)}},
        )
    )
    with pytest.raises(ShadowTraceInputError) as captured:
        atif_module._plan_mapping(
            duplicate_outcomes,
            contract=atif_module._PROFILE_CONTRACTS[ATIFProfile.HARBOR_CODEX_V1],
            environment=_environment(),
        )
    assert captured.value.reason_code == "invalid_outcome_metadata"


def _wire_action(**overrides: object) -> dict[str, object]:
    action: dict[str, object] = {
        "schema_version": "shadow-input/v1",
        "kind": "action",
        "source_event_id": "coverage-action",
        "occurred_at": _TIMESTAMP,
        "argv": ["printf", "coverage"],
        "working_directory": "/synthetic/coverage",
        "environment_digest": _ENVIRONMENT_DIGEST,
    }
    action.update(overrides)
    return action


def test_trace_timestamp_parent_and_wire_shape_guards_reject_aliases() -> None:
    with pytest.raises(ShadowTraceInputError):
        trace_module._parse_timestamp("2026-13-01T00:00:00Z", ordinal=1)
    with pytest.raises(ValueError, match="parent is invalid"):
        trace_module._parent_ref(1, {})
    with pytest.raises(ValueError, match="kind is invalid"):
        trace_module._validate_wire_record(
            {"schema_version": "shadow-input/v1", "kind": 1},
            run_id=RUN_ID,
            known={},
            ordinal=1,
        )
    with pytest.raises(ValueError, match="identifier is invalid"):
        trace_module._validate_wire_record(
            {
                "schema_version": "shadow-input/v1",
                "kind": "run_start",
                "source_event_id": 1,
                "occurred_at": _TIMESTAMP,
            },
            run_id=RUN_ID,
            known={},
            ordinal=1,
        )


def test_trace_wire_collections_require_lists_of_exact_values() -> None:
    for argv in (("pwd",), [1]):
        with pytest.raises(ValueError, match="arguments are invalid"):
            trace_module._validate_wire_record(
                _wire_action(argv=argv),
                run_id=RUN_ID,
                known={},
                ordinal=1,
            )

    kind, *_rest = trace_module._validate_wire_record(
        _wire_action(),
        run_id=RUN_ID,
        known={},
        ordinal=1,
    )
    assert kind.value == "action"

    with pytest.raises(ValueError, match="failures are invalid"):
        trace_module._validate_wire_record(
            {
                "schema_version": "shadow-input/v1",
                "kind": "test_result",
                "source_event_id": "coverage-test",
                "occurred_at": _TIMESTAMP,
                "action_source_event_id": "coverage-action",
                "framework": "pytest",
                "status": "failed",
                "failures": (),
            },
            run_id=RUN_ID,
            known={},
            ordinal=1,
        )
    with pytest.raises(ValueError, match="source is invalid"):
        trace_module._validate_wire_record(
            {
                "schema_version": "shadow-input/v1",
                "kind": "observation",
                "source_event_id": "coverage-observation",
                "occurred_at": _TIMESTAMP,
                "source": 1,
                "payload": {},
            },
            run_id=RUN_ID,
            known={},
            ordinal=1,
        )


def test_trace_canonicalizer_rejects_empty_frozen_and_nonfinal_finish_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ShadowTraceInputError):
        trace_module._canonical_records(
            [],
            run_id=RUN_ID,
            capture_scope="unknown",
            timestamp_mode="record_declared",
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(trace_module, "_freeze_json", lambda _value: ())
        with pytest.raises(ShadowTraceInputError):
            trace_module._canonical_records(
                [complete_records()[0]],
                run_id=RUN_ID,
                capture_scope="unknown",
                timestamp_mode="record_declared",
            )

    start = complete_records()[0]
    finish = {**complete_records()[-1], "occurred_at": "2026-07-17T09:00:01Z"}
    action = {**complete_records()[1], "occurred_at": "2026-07-17T09:00:02Z"}
    with pytest.raises(ShadowTraceInputError):
        trace_module._canonical_records(
            [start, finish, action],
            run_id=RUN_ID,
            capture_scope="unknown",
            timestamp_mode="record_declared",
        )


def test_trace_exact_json_and_aggregate_source_limits_take_late_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(trace_module._ExactJSONLimitError):
        trace_module._copy_exact_json(
            [1, 1],
            max_bytes=4,
            max_depth=5,
            max_container_items=5,
            max_nodes=10,
            max_string_bytes=10,
        )

    monkeypatch.setattr(trace_module, "MAX_SHADOW_TRACE_BYTES", 5)
    monkeypatch.setattr(
        trace_module,
        "_canonical_records",
        lambda *_args, **_kwargs: ((MappingProxyType({}),), (b"12345",), (), 0),
    )
    with pytest.raises(ShadowTraceInputError) as captured:
        build_trace([])
    assert captured.value.reason_code == "input_limit_exceeded"


def _new_trace_arguments() -> tuple[trace_module.ShadowTrace, dict[str, object]]:
    trace = build_trace()
    return trace, {
        "run_id": trace.run_id,
        "binding": trace.binding,
        "diagnostics": trace.diagnostics,
        "records": trace.records,
        "record_bytes": trace._wire_record_bytes(),
        "adapter_descriptor_bytes": trace._descriptor_preimage(),
        "adapter_configuration_bytes": trace._configuration_preimage(),
    }


def test_trace_sealer_rejects_profile_and_configuration_preimage_drift() -> None:
    trace, arguments = _new_trace_arguments()
    for field, changed in (
        ("adapter_descriptor_bytes", trace._descriptor_preimage() + b" "),
        ("adapter_configuration_bytes", trace._configuration_preimage() + b" "),
    ):
        current = dict(arguments)
        current[field] = changed
        with pytest.raises(ValueError, match="digest does not match"):
            trace_module._new_shadow_trace(**current)  # type: ignore[arg-type]


def test_trace_exactness_reaches_discriminant_and_recanonicalization_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discriminant = build_trace()
    discriminant.binding.__dict__["source_format"] = "atif"
    assert not discriminant._is_exact()

    recanonicalized = build_trace()
    monkeypatch.setattr(
        trace_module,
        "_canonical_records",
        lambda *_args, **_kwargs: (
            recanonicalized.records,
            (b"drift",) * len(recanonicalized.records),
            recanonicalized.diagnostics.input_kind_counts,
            recanonicalized.diagnostics.repeated_source_identifier_count,
        ),
    )
    assert not recanonicalized._is_exact()


def test_atif_trace_exactness_reaches_the_profile_seal_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_atif_trace()
    monkeypatch.setattr(atif_module, "_matches_sealed_trace_contract", lambda **_kwargs: False)

    assert not trace._is_exact()


def test_configuration_parser_and_manifest_loader_preserve_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt_preflight(_self: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(atif_module._JSONPreflight, "run", interrupt_preflight)
    with pytest.raises(KeyboardInterrupt):
        atif_module._configuration_object(canonical_json({"coverage": True}))

    with monkeypatch.context() as scoped:
        scoped.setattr(
            atif_module.resources,
            "files",
            lambda _package: (_ for _ in ()).throw(SystemExit(17)),
        )
        with pytest.raises(SystemExit, match="17"):
            atif_module._load_manifest_bytes()
