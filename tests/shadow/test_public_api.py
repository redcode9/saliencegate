from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import saliencegate.shadow as public_shadow
from saliencegate.shadow.adapters import ShadowTraceAdapter
from saliencegate.shadow.analyzer import ShadowAnalyzer, analyze_atif_bytes
from saliencegate.shadow.atif import (
    ATIFProfile,
    ATIFShadowAdapter,
    ShadowEnvironmentBinding,
)
from saliencegate.shadow.config import ShadowConfig
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInputError,
    ShadowInvariantError,
    ShadowStateError,
    ShadowTraceInputError,
)
from saliencegate.shadow.inputs import ShadowActionIdentityInput, ShadowEventRef
from saliencegate.shadow.observation import ShadowEventResult, ShadowObservation
from saliencegate.shadow.report import ShadowRunReport, build_shadow_run_report
from saliencegate.shadow.session import ShadowSession
from saliencegate.shadow.trace import (
    ShadowTrace,
    ShadowTraceBinding,
    ShadowTraceDiagnostics,
)
from saliencegate.shadow.trace_report import (
    ShadowTraceReport,
    decode_shadow_trace_report,
    encode_shadow_trace_report,
    verify_shadow_trace_source,
)


def assert_signature_contract(
    target: Callable[..., object],
    *,
    positional: tuple[str, ...] = (),
    keyword_only: tuple[str, ...] = (),
    defaults: dict[str, object] | None = None,
) -> None:
    parameters = inspect.signature(target).parameters
    expected_names = positional + keyword_only

    assert tuple(parameters) == expected_names
    assert tuple(parameter.kind for parameter in parameters.values()) == (
        (inspect.Parameter.POSITIONAL_OR_KEYWORD,) * len(positional)
        + (inspect.Parameter.KEYWORD_ONLY,) * len(keyword_only)
    )
    assert {
        name: parameter.default
        for name, parameter in parameters.items()
        if parameter.default is not inspect.Parameter.empty
    } == ({} if defaults is None else defaults)


def test_shadow_root_exports_the_public_contract() -> None:
    assert public_shadow.__all__ == [
        "ATIFProfile",
        "ATIFShadowAdapter",
        "ShadowActionIdentityInput",
        "ShadowAnalyzer",
        "ShadowConfig",
        "ShadowConfigurationError",
        "ShadowEnvironmentBinding",
        "ShadowEventRef",
        "ShadowEventResult",
        "ShadowInputError",
        "ShadowInvariantError",
        "ShadowObservation",
        "ShadowRunReport",
        "ShadowSession",
        "ShadowStateError",
        "ShadowTrace",
        "ShadowTraceAdapter",
        "ShadowTraceBinding",
        "ShadowTraceDiagnostics",
        "ShadowTraceInputError",
        "ShadowTraceReport",
        "analyze_atif_bytes",
        "build_shadow_run_report",
        "decode_shadow_trace_report",
        "encode_shadow_trace_report",
        "verify_shadow_trace_source",
    ]
    assert public_shadow.ATIFProfile is ATIFProfile
    assert public_shadow.ATIFShadowAdapter is ATIFShadowAdapter
    assert public_shadow.ShadowAnalyzer is ShadowAnalyzer
    assert public_shadow.ShadowActionIdentityInput is ShadowActionIdentityInput
    assert public_shadow.ShadowConfig is ShadowConfig
    assert public_shadow.ShadowConfigurationError is ShadowConfigurationError
    assert public_shadow.ShadowEventRef is ShadowEventRef
    assert public_shadow.ShadowEventResult is ShadowEventResult
    assert public_shadow.ShadowEnvironmentBinding is ShadowEnvironmentBinding
    assert public_shadow.ShadowInputError is ShadowInputError
    assert public_shadow.ShadowInvariantError is ShadowInvariantError
    assert public_shadow.ShadowObservation is ShadowObservation
    assert public_shadow.ShadowRunReport is ShadowRunReport
    assert public_shadow.ShadowSession is ShadowSession
    assert public_shadow.ShadowStateError is ShadowStateError
    assert public_shadow.ShadowTrace is ShadowTrace
    assert public_shadow.ShadowTraceAdapter is ShadowTraceAdapter
    assert public_shadow.ShadowTraceBinding is ShadowTraceBinding
    assert public_shadow.ShadowTraceDiagnostics is ShadowTraceDiagnostics
    assert public_shadow.ShadowTraceInputError is ShadowTraceInputError
    assert public_shadow.ShadowTraceReport is ShadowTraceReport
    assert public_shadow.analyze_atif_bytes is analyze_atif_bytes
    assert public_shadow.build_shadow_run_report is build_shadow_run_report
    assert public_shadow.decode_shadow_trace_report is decode_shadow_trace_report
    assert public_shadow.encode_shadow_trace_report is encode_shadow_trace_report
    assert public_shadow.verify_shadow_trace_source is verify_shadow_trace_source
    assert not hasattr(public_shadow, "ShadowReportRow")


def test_shadow_regular_callable_signatures_are_semantically_frozen() -> None:
    assert_signature_contract(
        ATIFShadowAdapter,
        keyword_only=("profile", "environment"),
    )
    assert_signature_contract(
        ATIFShadowAdapter.adapt_bytes,
        positional=("self", "source"),
        keyword_only=(
            "run_id",
            "task_scope_digest",
            "lineage_scope_digest",
            "capture_manifest_digest",
        ),
        defaults={
            "task_scope_digest": None,
            "lineage_scope_digest": None,
            "capture_manifest_digest": None,
        },
    )
    assert_signature_contract(ShadowAnalyzer, positional=("session",))
    assert_signature_contract(
        ShadowAnalyzer.analyze,
        positional=("self", "trace"),
    )
    assert_signature_contract(ShadowConfig.reference)
    for error_type in (
        ShadowConfigurationError,
        ShadowInputError,
        ShadowInvariantError,
        ShadowStateError,
    ):
        assert_signature_contract(error_type)
    assert_signature_contract(
        ShadowTraceInputError,
        positional=("reason_code",),
        keyword_only=("step_ordinal", "call_ordinal", "result_ordinal"),
        defaults={
            "step_ordinal": None,
            "call_ordinal": None,
            "result_ordinal": None,
        },
    )
    assert_signature_contract(ShadowSession)
    assert_signature_contract(ShadowTrace)
    assert_signature_contract(
        ShadowTrace.adapter_profile_digest,
        positional=("adapter_descriptor",),
    )
    assert_signature_contract(
        ShadowTrace.from_records,
        positional=("records",),
        keyword_only=(
            "run_id",
            "adapter_profile_id",
            "adapter_descriptor",
            "source_bytes",
            "source_format",
            "source_schema_version",
            "timestamp_mode",
            "capture_scope",
            "task_scope_digest",
            "lineage_scope_digest",
            "capture_manifest_digest",
        ),
        defaults={
            "source_bytes": None,
            "source_format": "shadow-records",
            "source_schema_version": "shadow-input/v1",
            "timestamp_mode": "record_declared",
            "capture_scope": "unknown",
            "task_scope_digest": None,
            "lineage_scope_digest": None,
            "capture_manifest_digest": None,
        },
    )
    assert_signature_contract(
        analyze_atif_bytes,
        positional=("source_bytes",),
        keyword_only=(
            "run_id",
            "profile",
            "environment",
            "installation_key",
            "redaction_policy",
            "task_scope_digest",
            "lineage_scope_digest",
            "capture_manifest_digest",
        ),
        defaults={
            "installation_key": None,
            "redaction_policy": None,
            "task_scope_digest": None,
            "lineage_scope_digest": None,
            "capture_manifest_digest": None,
        },
    )
    assert_signature_contract(
        build_shadow_run_report,
        keyword_only=(
            "run_id",
            "initial_ledger_entry_count",
            "initial_ledger_chain_tag",
            "initial_ledger_projection_tag",
            "initial_ledger_head_tag",
            "input_byte_digest",
            "normalized_input_digest",
            "redaction_policy_tag",
            "detector_profile_digest",
            "capture_scope",
            "task_scope_digest",
            "lineage_scope_digest",
            "capture_manifest_digest",
            "rows",
            "observations",
        ),
        defaults={
            "task_scope_digest": None,
            "lineage_scope_digest": None,
            "capture_manifest_digest": None,
        },
    )
    assert_signature_contract(decode_shadow_trace_report, positional=("data",))
    assert_signature_contract(encode_shadow_trace_report, positional=("report",))
    assert_signature_contract(
        verify_shadow_trace_source,
        positional=("report", "source"),
        keyword_only=("adapter",),
    )


def test_shadow_session_factory_signatures_are_semantically_frozen() -> None:
    assert_signature_contract(
        ShadowSession.action_identity,
        positional=("self",),
        keyword_only=(
            "source_event_id",
            "occurred_at",
            "action_digest",
            "workspace_digest",
            "environment_digest",
            "identity_authority",
        ),
    )
    assert_signature_contract(
        ShadowSession.in_memory_for_trace,
        keyword_only=(
            "run_id",
            "trace_binding",
            "config",
            "installation_key",
            "redaction_policy",
        ),
        defaults={
            "config": None,
            "installation_key": None,
            "redaction_policy": None,
        },
    )
    assert_signature_contract(
        ShadowSession.sqlite_for_trace,
        positional=("path",),
        keyword_only=(
            "run_id",
            "trace_binding",
            "installation_key",
            "config",
            "redaction_policy",
        ),
        defaults={"config": None, "redaction_policy": None},
    )
    for factory, positional in (
        (ShadowSession.in_memory, ()),
        (ShadowSession.sqlite, ("path",)),
    ):
        assert_signature_contract(
            factory,
            positional=positional,
            keyword_only=(
                "run_id",
                "config",
                "installation_key",
                "redaction_policy",
                "capture_scope",
                "task_scope_digest",
                "lineage_scope_digest",
                "capture_manifest_digest",
                "source_adapter",
            ),
            defaults={
                "config": None,
                "installation_key": None,
                "redaction_policy": None,
                "capture_scope": "unknown",
                "task_scope_digest": None,
                "lineage_scope_digest": None,
                "capture_manifest_digest": None,
                "source_adapter": "saliencegate-shadow/v1",
            },
        )


def test_exported_shadow_model_fields_and_schema_defaults_are_frozen() -> None:
    expected = {
        ShadowActionIdentityInput: (
            "shadow-input/v1",
            """schema_version source_event_id occurred_at kind action_digest workspace_digest
            environment_digest identity_authority""",
        ),
        ShadowConfig: (
            "shadow-config/v1",
            """schema_version detectors supported_signal_types unsupported_signal_types
            applicability evaluator_id indeterminate_reasons evaluator_configuration_digest
            detector_profile_digest""",
        ),
        ShadowEnvironmentBinding: (
            "shadow-environment-binding/v1",
            "schema_version default_working_directory environment_digest",
        ),
        ShadowEventRef: (
            "shadow-event-ref/v1",
            "schema_version run_id event_id sequence",
        ),
        ShadowEventResult: (
            "shadow-event-result/v1",
            "schema_version ref observation",
        ),
        ShadowObservation: (
            "shadow-observation/v1",
            """schema_version run_id event_id source_event_digest sequence event_prefix_digest
            context_first_sequence context_last_sequence context_event_count context_truncated
            detection_context_digest redacted_event_digest redaction_policy_tag
            detector_profile_digest evaluator_configuration_digest extraction_report_digest
            feature_snapshot_digest supported_signal_types unsupported_signal_types
            detector_evaluations detected_signals heuristic_evaluations cli_input_ordinal
            execution_mode evidence_level task_outcome_evidence intervention_outcome_evidence
            confirmatory calibrated calibration_eligible decision_authority
            representativeness_supported task_efficacy_supported counterfactual_effect_supported
            model_calls budget_reservations cycles_created memory_revisions interventions
            delivery_authorizations deliveries intervention_outcomes observation_digest""",
        ),
        ShadowRunReport: (
            "shadow-run-report/v1",
            """schema_version run_id initial_ledger_entry_count initial_ledger_chain_tag
            initial_ledger_projection_tag initial_ledger_head_tag input_byte_digest
            normalized_input_digest redaction_policy_tag detector_profile_digest capture_scope
            task_scope_digest lineage_scope_digest capture_manifest_digest split_metadata_complete
            input_row_count unique_input_event_count retry_row_count appended_event_count
            preexisting_event_count rejected_row_count evaluated_unique_event_count
            observation_count rows observations supported_signal_types unsupported_signal_types
            detector_outcome_counts abstention_reason_counts heuristic_disposition_counts
            applicable_detector_evaluation_count
            evidence_sufficient_applicable_detector_evaluation_count signal_cooccurrence_counts
            event_type_counts phase_counts first_flagged_event_sequence execution_mode
            evidence_level task_outcome_evidence intervention_outcome_evidence confirmatory
            calibrated calibration_eligible decision_authority representativeness_supported
            task_efficacy_supported counterfactual_effect_supported model_calls
            budget_reservations cycles_created memory_revisions interventions
            delivery_authorizations deliveries intervention_outcomes report_digest""",
        ),
        ShadowTraceBinding: (
            "shadow-trace-binding/v1",
            """schema_version source_format source_schema_version source_digest_kind
            source_byte_count source_byte_digest adapter_profile_id adapter_profile_digest
            adapter_configuration_digest source_adapter identity_mode timestamp_mode capture_scope
            task_scope_digest lineage_scope_digest capture_manifest_digest binding_digest""",
        ),
        ShadowTraceReport: (
            "shadow-trace-report/v1",
            """schema_version run_id binding binding_digest diagnostics diagnostics_digest
            mapped_record_digest normalized_input_digest shadow_report report_digest""",
        ),
    }

    for model, (schema_version, field_names) in expected.items():
        assert tuple(model.model_fields) == tuple(field_names.split())
        assert model.model_fields["schema_version"].default == schema_version


def test_shadow_trace_adapter_protocol_members_are_frozen() -> None:
    members = {name for name in ShadowTraceAdapter.__dict__ if not name.startswith("_")}

    assert members == {"adapt_bytes", "profile_digest", "profile_id"}
    assert isinstance(ShadowTraceAdapter.profile_id, property)
    assert isinstance(ShadowTraceAdapter.profile_digest, property)
    assert_signature_contract(
        ShadowTraceAdapter.adapt_bytes,
        positional=("self", "source"),
        keyword_only=(
            "run_id",
            "task_scope_digest",
            "lineage_scope_digest",
            "capture_manifest_digest",
        ),
        defaults={
            "task_scope_digest": None,
            "lineage_scope_digest": None,
            "capture_manifest_digest": None,
        },
    )


def test_root_package_does_not_import_or_reexport_shadow() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import saliencegate, sys; "
                "assert 'saliencegate.shadow' not in sys.modules; "
                "assert not hasattr(saliencegate, 'ShadowConfig')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_shadow_contract_has_no_forbidden_direct_imports() -> None:
    forbidden = (
        "anthropic",
        "harbor",
        "httpx",
        "openai",
        "openai_harmony",
        "saliencegate.commands",
        "saliencegate.intervention",
        "saliencegate.memory",
        "saliencegate.models",
        "saliencegate.ports.model_calls",
        "saliencegate.ports.models",
        "saliencegate.ports.two_phase",
        "saliencegate.repository",
        "saliencegate.runtime",
    )
    imported_by_path: dict[Path, set[str]] = {}
    for path in sorted(Path("src/saliencegate/shadow").glob("*.py")):
        imported: set[str] = set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        imported_by_path[path] = imported

    violations: list[tuple[Path, str]] = []
    for path, imported in imported_by_path.items():
        for name in imported:
            for prefix in forbidden:
                if name == prefix or name.startswith(f"{prefix}."):
                    if path.name == "session.py" and prefix == "saliencegate.repository":
                        continue
                    violations.append((path, name))

    assert violations == []


def test_shadow_import_remains_provider_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import saliencegate.shadow, sys; "
                "assert 'anthropic' not in sys.modules; "
                "assert 'harbor' not in sys.modules; "
                "assert 'httpx' not in sys.modules; "
                "assert 'openai' not in sys.modules; "
                "assert 'openai_harmony' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_public_example_uses_only_the_core_observational_surface() -> None:
    source = Path("examples/shadow_asyncio.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    assert "saliencegate.shadow" in imported
    assert all(
        name not in source
        for name in (
            "saliencegate.models",
            "saliencegate.memory",
            "saliencegate.runtime",
            "saliencegate.commands",
            "httpx",
            "openai_harmony",
            "os.environ",
            "getenv(",
            "socket",
        )
    )
