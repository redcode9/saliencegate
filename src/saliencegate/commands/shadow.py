from __future__ import annotations

import asyncio
import os
import stat
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated, Literal, Self, TypeAlias, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import PayloadDigest, SignalType, canonical_json
from saliencegate.domain.records import UUID4, Sha256Digest
from saliencegate.ports.repository import RepositoryError
from saliencegate.security import (
    AtomicFilePublication,
    InstallationKey,
    RedactionPolicy,
    SecureFileBoundError,
    SecureFileError,
    SecureFileUnsupportedError,
    StableFileAuthorization,
    StableFileRead,
    StableReadPolicy,
    authorize_private_sqlite_path,
    inspect_private_file_location,
    load_or_create_installation_key,
    read_stable_file,
)
from saliencegate.shadow import (
    ATIFProfile,
    ATIFShadowAdapter,
    ShadowConfig,
    ShadowConfigurationError,
    ShadowEnvironmentBinding,
    ShadowInputError,
    ShadowInvariantError,
    ShadowSession,
    ShadowStateError,
    ShadowTraceInputError,
    ShadowTraceReport,
    decode_shadow_trace_report,
    encode_shadow_trace_report,
)
from saliencegate.shadow.analyzer import (
    _analyze_legacy_preflighted,
    _analyze_prepared,
    _prepare_analysis,
    _preview_prepared,
)
from saliencegate.shadow.evaluation import ShadowHeuristicDisposition
from saliencegate.shadow.io import (
    MAX_SHADOW_REPORT_BYTES,
    CaptureScope,
    PreflightedShadowTrace,
    ShadowReportBinding,
    ShadowTraceReportBinding,
    authorize_shadow_report_publication,
    authorize_shadow_trace_report_publication,
    decode_shadow_run_report,
    encode_shadow_run_report,
    read_shadow_trace,
    shadow_report_binding,
    validate_published_shadow_report,
    validate_published_shadow_trace_report,
    validate_shadow_report_replacement,
    validate_shadow_trace_report_binding,
)
from saliencegate.shadow.report import ShadowRunReport
from saliencegate.shadow.session import _redaction_policy_tag
from saliencegate.shadow.trace import (
    MAX_SHADOW_TRACE_BYTES,
    ATIFShadowDiagnostics,
    ResultDisposition,
    ShadowTrace,
    TimestampMode,
    ToolCallDisposition,
)
from saliencegate.shadow.trace_report import MAX_SHADOW_TRACE_REPORT_BYTES
from saliencegate.signals import AbstentionReason, DetectionStatus

SHADOW_COMMAND_REPORT_SCHEMA_VERSION: Literal["shadow-command-report/v1"] = (
    "shadow-command-report/v1"
)
ATIF_SHADOW_COMMAND_REPORT_SCHEMA_VERSION: Literal["shadow-atif-command-report/v1"] = (
    "shadow-atif-command-report/v1"
)
_DEFAULT_REPOSITORY = ":memory:"
_DEFAULT_SOURCE_ADAPTER = "saliencegate-shadow/v1"
_SQLITE_NAMES = ("", "-wal", "-shm", "-journal")
_SUPPORTED_SIGNAL_TYPES = (
    SignalType.REPEATED_ACTION,
    SignalType.REPEATED_FAILURE,
    SignalType.TEST_FAILURE,
    SignalType.TOOL_ERROR,
)
_UNSUPPORTED_SIGNAL_TYPES = (
    SignalType.CONFLICT,
    SignalType.CONTEXT_SHIFT,
    SignalType.IRREVERSIBLE_ACTION,
    SignalType.STAGNATION,
    SignalType.STALE_CONSTRAINT,
)
_HEURISTIC_DISPOSITIONS = tuple(sorted(ShadowHeuristicDisposition, key=lambda item: item.value))
_DETECTION_STATUSES = tuple(sorted(DetectionStatus, key=lambda item: item.value))
_ABSTENTION_REASONS = tuple(sorted(AbstentionReason, key=lambda item: item.value))
_TOOL_CALL_DISPOSITIONS: tuple[ToolCallDisposition, ...] = (
    "mapped_action",
    "ignored_unsupported_function",
    "ignored_continuation",
    "ignored_non_command_wait",
    "ignored_unsubmitted_keystrokes",
    "ignored_unresolved_terminal_submission",
    "ignored_copied_context",
)
_RESULT_DISPOSITIONS: tuple[ResultDisposition, ...] = (
    "mapped_structured_outcome",
    "ignored_evidence_absent",
    "ignored_ambiguous_parent",
    "ignored_no_parent",
    "ignored_unsupported_parent",
    "ignored_copied_context",
)

_NonNegativeCount = Annotated[int, Field(ge=0, le=(1 << 63) - 1)]
_DispositionCount: TypeAlias = tuple[ShadowHeuristicDisposition, _NonNegativeCount]
_ToolDispositionCount: TypeAlias = tuple[ToolCallDisposition, _NonNegativeCount]
_ResultDispositionCount: TypeAlias = tuple[ResultDisposition, _NonNegativeCount]
_DetectorOutcomeCount: TypeAlias = tuple[SignalType, DetectionStatus, _NonNegativeCount]
_AbstentionReasonCount: TypeAlias = tuple[SignalType, AbstentionReason, _NonNegativeCount]
_ProfileEvidenceAvailability: TypeAlias = Literal["conditional", "none"]
_ProfileDetectorEvidence: TypeAlias = tuple[SignalType, _ProfileEvidenceAvailability]
_StructuredOutcomeCoverage: TypeAlias = tuple[_NonNegativeCount, _NonNegativeCount]
_ATIFProfileId: TypeAlias = Literal["harbor-terminus-2/v1", "harbor-codex/v1"]


class ShadowCommandInputError(ValueError):
    """A value-free invalid command input, output, or alias failure."""

    def __init__(self) -> None:
        super().__init__("shadow input or output is invalid")


class ShadowCommandConfigurationError(RuntimeError):
    """A value-free local key, repository, or platform failure."""

    def __init__(self) -> None:
        super().__init__("shadow configuration is invalid")


class ShadowCommandIntegrityError(RuntimeError):
    """A value-free replacement or published-report integrity failure."""

    def __init__(self) -> None:
        super().__init__("shadow report integrity check failed")


class ShadowCommandReport(BaseModel):
    """The one path-free command summary printed after successful publication."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["shadow-command-report/v1"] = SHADOW_COMMAND_REPORT_SCHEMA_VERSION
    status: Literal["ok"] = "ok"
    run_id: UUID4
    input_byte_digest: Sha256Digest
    normalized_input_digest: Sha256Digest
    detector_profile_digest: Sha256Digest
    supported_signal_types: tuple[SignalType, ...]
    unsupported_signal_types: tuple[SignalType, ...]
    unique_input_event_count: _NonNegativeCount
    retry_row_count: _NonNegativeCount
    heuristic_disposition_counts: tuple[_DispositionCount, ...]
    report_digest: Sha256Digest
    execution_mode: Literal["shadow"] = "shadow"
    evidence_level: Literal["descriptive_observational"] = "descriptive_observational"
    task_outcome_evidence: Literal["none"] = "none"
    intervention_outcome_evidence: Literal["none"] = "none"
    confirmatory: Literal[False] = False
    calibrated: Literal[False] = False
    calibration_eligible: Literal[False] = False
    decision_authority: Literal[False] = False
    representativeness_supported: Literal[False] = False
    task_efficacy_supported: Literal[False] = False
    counterfactual_effect_supported: Literal[False] = False
    model_calls: Literal[0] = 0
    budget_reservations: Literal[0] = 0
    cycles_created: Literal[0] = 0
    memory_revisions: Literal[0] = 0
    interventions: Literal[0] = 0
    delivery_authorizations: Literal[0] = 0
    deliveries: Literal[0] = 0
    intervention_outcomes: Literal[0] = 0

    @field_validator("schema_version", "status", mode="before")
    @classmethod
    def require_exact_fixed_text(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("command report text is invalid")
        return value

    @field_validator(
        "input_byte_digest",
        "normalized_input_digest",
        "detector_profile_digest",
        "report_digest",
        mode="before",
    )
    @classmethod
    def require_exact_digest_text(cls, value: object) -> object:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("command report digest is invalid")
        return value

    @field_validator("supported_signal_types", "unsupported_signal_types", mode="before")
    @classmethod
    def require_exact_signal_type_tuple(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not SignalType for item in value):
            raise ValueError("command report signal types are invalid")
        return value

    @field_validator("heuristic_disposition_counts", mode="before")
    @classmethod
    def require_exact_disposition_counts(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("command report disposition counts are invalid")
        for item in value:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not ShadowHeuristicDisposition
                or type(item[1]) is not int
            ):
                raise ValueError("command report disposition counts are invalid")
        return value

    @model_validator(mode="after")
    def fields_match_the_frozen_command_contract(self) -> Self:
        if self.supported_signal_types != _SUPPORTED_SIGNAL_TYPES:
            raise ValueError("command report supported types are invalid")
        if self.unsupported_signal_types != _UNSUPPORTED_SIGNAL_TYPES:
            raise ValueError("command report unsupported types are invalid")
        if tuple(item[0] for item in self.heuristic_disposition_counts) != (
            _HEURISTIC_DISPOSITIONS
        ):
            raise ValueError("command report dispositions are not canonical")
        if sum(item[1] for item in self.heuristic_disposition_counts) != (
            self.unique_input_event_count
        ):
            raise ValueError("command report disposition denominator is invalid")
        return self


class ATIFShadowCommandReport(BaseModel):
    """A path-free ATIF mapping and Shadow analysis summary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["shadow-atif-command-report/v1"] = (
        ATIF_SHADOW_COMMAND_REPORT_SCHEMA_VERSION
    )
    status: Literal["ok"] = "ok"
    run_id: UUID4
    adapter_profile_id: _ATIFProfileId
    source_schema_version: Annotated[str, Field(min_length=1, max_length=128)]
    timestamp_mode: TimestampMode
    root_segment_only: Literal[True] = True
    continued_trajectory_ref_present: bool
    embedded_subagent_trajectory_count: _NonNegativeCount
    complete_execution_session_coverage: Literal[False] = False
    producer_authentication: Literal["none"] = "none"
    outcome_evidence_authority: Literal["none", "producer_claimed_structured"]
    profile_audit_manifest_digest: Sha256Digest
    total_step_count: _NonNegativeCount
    ignored_message_step_count: _NonNegativeCount
    total_tool_call_count: _NonNegativeCount
    tool_call_disposition_counts: tuple[_ToolDispositionCount, ...]
    total_observation_result_count: _NonNegativeCount
    result_disposition_counts: tuple[_ResultDispositionCount, ...]
    mapped_shadow_record_count: _NonNegativeCount
    structured_outcome_coverage: _StructuredOutcomeCoverage
    evaluated_unique_event_count: _NonNegativeCount
    heuristic_disposition_counts: tuple[_DispositionCount, ...]
    supported_signal_types: tuple[SignalType, ...]
    unsupported_signal_types: tuple[SignalType, ...]
    profile_detector_evidence: tuple[_ProfileDetectorEvidence, ...]
    detector_outcome_counts: tuple[_DetectorOutcomeCount, ...]
    abstention_reason_counts: tuple[_AbstentionReasonCount, ...]
    applicable_detector_evaluation_count: _NonNegativeCount
    evidence_sufficient_applicable_detector_evaluation_count: _NonNegativeCount
    report_digest: Sha256Digest
    execution_mode: Literal["shadow"] = "shadow"
    evidence_level: Literal["descriptive_observational"] = "descriptive_observational"
    task_outcome_evidence: Literal["none"] = "none"
    intervention_outcome_evidence: Literal["none"] = "none"
    confirmatory: Literal[False] = False
    calibrated: Literal[False] = False
    calibration_eligible: Literal[False] = False
    decision_authority: Literal[False] = False
    representativeness_supported: Literal[False] = False
    task_efficacy_supported: Literal[False] = False
    counterfactual_effect_supported: Literal[False] = False
    model_calls: Literal[0] = 0
    budget_reservations: Literal[0] = 0
    cycles_created: Literal[0] = 0
    memory_revisions: Literal[0] = 0
    interventions: Literal[0] = 0
    delivery_authorizations: Literal[0] = 0
    deliveries: Literal[0] = 0
    intervention_outcomes: Literal[0] = 0

    @field_validator(
        "schema_version",
        "status",
        "adapter_profile_id",
        "source_schema_version",
        "timestamp_mode",
        "producer_authentication",
        "outcome_evidence_authority",
        mode="before",
    )
    @classmethod
    def require_exact_fixed_text(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("ATIF command report text is invalid")
        return value

    @field_validator(
        "profile_audit_manifest_digest",
        "report_digest",
        mode="before",
    )
    @classmethod
    def require_exact_digest_text(cls, value: object) -> object:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("ATIF command report digest is invalid")
        return value

    @field_validator(
        "root_segment_only",
        "continued_trajectory_ref_present",
        "complete_execution_session_coverage",
        mode="before",
    )
    @classmethod
    def require_exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("ATIF command report boolean is invalid")
        return value

    @field_validator(
        "tool_call_disposition_counts",
        "result_disposition_counts",
        "heuristic_disposition_counts",
        "profile_detector_evidence",
        "detector_outcome_counts",
        "abstention_reason_counts",
        mode="before",
    )
    @classmethod
    def require_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not tuple for item in value):
            raise ValueError("ATIF command report counts are invalid")
        return value

    @field_validator("structured_outcome_coverage", mode="before")
    @classmethod
    def require_exact_coverage_tuple(cls, value: object) -> object:
        if type(value) is not tuple or len(value) != 2:
            raise ValueError("ATIF command report coverage is invalid")
        return value

    @field_validator("supported_signal_types", "unsupported_signal_types", mode="before")
    @classmethod
    def require_exact_signal_type_tuple(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not SignalType for item in value):
            raise ValueError("ATIF command report signal types are invalid")
        return value

    @model_validator(mode="after")
    def fields_match_the_atif_command_contract(self) -> Self:
        tool_counts = dict(self.tool_call_disposition_counts)
        result_counts = dict(self.result_disposition_counts)
        detector_counts = {
            (signal_type, status): count
            for signal_type, status, count in self.detector_outcome_counts
        }
        abstention_counts = {
            (signal_type, reason): count
            for signal_type, reason, count in self.abstention_reason_counts
        }
        expected_profile_evidence: tuple[_ProfileDetectorEvidence, ...]
        if self.adapter_profile_id == ATIFProfile.HARBOR_TERMINUS_2_V1.value:
            expected_profile_evidence = (
                (SignalType.REPEATED_ACTION, "conditional"),
                (SignalType.REPEATED_FAILURE, "none"),
                (SignalType.TEST_FAILURE, "none"),
                (SignalType.TOOL_ERROR, "none"),
            )
            profile_matches = (
                self.source_schema_version in {"ATIF-v1.6", "ATIF-v1.7"}
                and self.outcome_evidence_authority == "none"
            )
        else:
            expected_profile_evidence = (
                (SignalType.REPEATED_ACTION, "conditional"),
                (SignalType.REPEATED_FAILURE, "conditional"),
                (SignalType.TEST_FAILURE, "none"),
                (SignalType.TOOL_ERROR, "conditional"),
            )
            profile_matches = (
                self.source_schema_version == "ATIF-v1.7"
                and self.outcome_evidence_authority == "producer_claimed_structured"
            )
        expected_detector_keys = tuple(
            (signal_type, status)
            for signal_type in _SUPPORTED_SIGNAL_TYPES
            for status in _DETECTION_STATUSES
        )
        expected_abstention_keys = tuple(
            (signal_type, reason)
            for signal_type in _SUPPORTED_SIGNAL_TYPES
            for reason in _ABSTENTION_REASONS
        )
        if (
            self.supported_signal_types != _SUPPORTED_SIGNAL_TYPES
            or self.unsupported_signal_types != _UNSUPPORTED_SIGNAL_TYPES
            or tuple(item[0] for item in self.tool_call_disposition_counts)
            != _TOOL_CALL_DISPOSITIONS
            or tuple(item[0] for item in self.result_disposition_counts) != _RESULT_DISPOSITIONS
            or sum(item[1] for item in self.tool_call_disposition_counts)
            != self.total_tool_call_count
            or sum(item[1] for item in self.result_disposition_counts)
            != self.total_observation_result_count
            or self.ignored_message_step_count > self.total_step_count
            or self.mapped_shadow_record_count
            != 2 + tool_counts["mapped_action"] + result_counts["mapped_structured_outcome"]
            or self.structured_outcome_coverage
            != (
                result_counts["mapped_structured_outcome"],
                tool_counts["mapped_action"],
            )
            or tuple(item[0] for item in self.heuristic_disposition_counts)
            != _HEURISTIC_DISPOSITIONS
            or sum(item[1] for item in self.heuristic_disposition_counts)
            != self.evaluated_unique_event_count
            or self.profile_detector_evidence != expected_profile_evidence
            or not profile_matches
            or tuple((item[0], item[1]) for item in self.detector_outcome_counts)
            != expected_detector_keys
            or tuple((item[0], item[1]) for item in self.abstention_reason_counts)
            != expected_abstention_keys
            or self.evidence_sufficient_applicable_detector_evaluation_count
            > self.applicable_detector_evaluation_count
        ):
            raise ValueError("ATIF command report contract is inconsistent")
        for signal_type in _SUPPORTED_SIGNAL_TYPES:
            if sum(detector_counts[(signal_type, status)] for status in _DETECTION_STATUSES) != (
                self.evaluated_unique_event_count
            ):
                raise ValueError("ATIF command report detector denominator is invalid")
            if (
                sum(abstention_counts[(signal_type, reason)] for reason in _ABSTENTION_REASONS)
                != detector_counts[(signal_type, DetectionStatus.ABSTAINED)]
            ):
                raise ValueError("ATIF command report abstention denominator is invalid")
        return self


@dataclass(frozen=True, slots=True)
class _LocationPlan:
    output: StableFileAuthorization
    sqlite_slots: tuple[StableFileAuthorization, ...]
    sqlite_path: str | None


@dataclass(frozen=True, slots=True, repr=False)
class _ATIFOutputPlan:
    publication: AtomicFilePublication
    replacement_data: bytes | None


def _copy_path(value: str | os.PathLike[str]) -> str:
    try:
        copied = os.fspath(value)
    except (OSError, TypeError, ValueError):
        raise ShadowCommandInputError() from None
    if type(copied) is not str or not copied:
        raise ShadowCommandInputError()
    return copied


def _copy_repository_path(value: str | os.PathLike[str]) -> str | None:
    copied = _copy_path(value)
    return None if copied == _DEFAULT_REPOSITORY else copied


def _locations_alias(authorizations: tuple[StableFileAuthorization, ...]) -> bool:
    return any(left.aliases(right) for left, right in combinations(authorizations, 2))


def _inspect_locations(
    source: StableFileAuthorization,
    *,
    output_path: str,
    repository_path: str | None,
) -> _LocationPlan:
    output: StableFileAuthorization | None = None
    sqlite_slots: tuple[StableFileAuthorization, ...] = ()
    output_failed = False
    sqlite_failed = False
    try:
        output = inspect_private_file_location(output_path)
    except SecureFileUnsupportedError:
        raise ShadowCommandConfigurationError() from None
    except Exception:
        output_failed = True
    if output_failed or output is None:
        raise ShadowCommandInputError()
    if repository_path is not None:
        inspected: list[StableFileAuthorization] = []
        try:
            for suffix in _SQLITE_NAMES:
                inspected.append(inspect_private_file_location(f"{repository_path}{suffix}"))
            sqlite_slots = tuple(inspected)
        except Exception:
            sqlite_failed = True
        if sqlite_failed:
            raise ShadowCommandConfigurationError()
    aliases = False
    try:
        aliases = _locations_alias((source, output, *sqlite_slots))
    except Exception:
        aliases = True
    if aliases:
        raise ShadowCommandInputError()
    return _LocationPlan(
        output=output,
        sqlite_slots=sqlite_slots,
        sqlite_path=repository_path,
    )


def _desired_report_binding(
    trace: PreflightedShadowTrace,
    *,
    redaction_policy_tag: PayloadDigest,
    config: ShadowConfig,
    capture_scope: CaptureScope,
    task_scope_digest: str | None,
    lineage_scope_digest: str | None,
    capture_manifest_digest: str | None,
) -> ShadowReportBinding:
    return ShadowReportBinding(
        run_id=trace.run_id,
        input_byte_digest=trace.input_byte_digest,
        normalized_input_digest=trace.normalized_input_digest,
        redaction_policy_tag=redaction_policy_tag,
        detector_profile_digest=config.detector_profile_digest,
        capture_scope=capture_scope,
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
    )


def _existing_output_is_valid(
    location: StableFileAuthorization,
    binding: ShadowReportBinding,
) -> bool:
    if location._target_identity is None:
        return True
    stable = read_stable_file(
        location.path,
        maximum_bytes=MAX_SHADOW_REPORT_BYTES,
        policy=StableReadPolicy.PRIVATE_OWNER,
    )
    location.revalidate()
    return validate_shadow_report_replacement(stable.data, binding)


def _authorize_output(
    location: StableFileAuthorization,
    *,
    binding: ShadowReportBinding,
    replace: bool,
) -> AtomicFilePublication:
    existing_invalid = False
    precheck_failed = False
    if replace and location._target_identity is not None:
        try:
            existing_invalid = not _existing_output_is_valid(location, binding)
        except SecureFileBoundError:
            existing_invalid = True
        except Exception:
            precheck_failed = True
    if existing_invalid:
        raise ShadowCommandIntegrityError()
    if precheck_failed:
        raise ShadowCommandInputError()
    publication: AtomicFilePublication | None = None
    unsupported = False
    failed = False
    try:
        publication = authorize_shadow_report_publication(
            location.path,
            replacement_binding=binding if replace else None,
        )
        location.revalidate()
    except SecureFileUnsupportedError:
        unsupported = True
    except Exception:
        failed = True
    if unsupported:
        raise ShadowCommandConfigurationError()
    if failed or publication is None:
        raise ShadowCommandInputError()
    return publication


def _authorize_sqlite(
    trace: PreflightedShadowTrace,
    locations: _LocationPlan,
    publication: AtomicFilePublication,
) -> StableFileAuthorization | None:
    if locations.sqlite_path is None:
        return None
    precheck_failed = False
    try:
        trace.authorization.revalidate()
        locations.output.revalidate()
        publication.authorization.revalidate()
        for slot in locations.sqlite_slots:
            slot.revalidate()
    except Exception:
        precheck_failed = True
    if precheck_failed:
        raise ShadowCommandInputError()
    authorization: StableFileAuthorization | None = None
    failed = False
    try:
        authorization = authorize_private_sqlite_path(locations.sqlite_path)
    except Exception:
        failed = True
    if failed or authorization is None:
        raise ShadowCommandConfigurationError()
    aliases = False
    try:
        if not locations.output.aliases(publication.authorization):
            raise SecureFileError()
        aliases = any(
            candidate.aliases(authorization)
            for candidate in (
                trace.authorization,
                locations.output,
                publication.authorization,
            )
        )
    except Exception:
        aliases = True
    if aliases:
        raise ShadowCommandInputError()
    return authorization


def _new_session(
    *,
    sqlite_authorization: StableFileAuthorization | None,
    run_id: UUID,
    config: ShadowConfig,
    installation_key: InstallationKey,
    redaction_policy: RedactionPolicy,
    capture_scope: CaptureScope,
    task_scope_digest: str | None,
    lineage_scope_digest: str | None,
    capture_manifest_digest: str | None,
    source_adapter: str,
) -> ShadowSession:
    if sqlite_authorization is None:
        return ShadowSession.in_memory(
            run_id=run_id,
            config=config,
            installation_key=installation_key,
            redaction_policy=redaction_policy,
            capture_scope=capture_scope,
            task_scope_digest=task_scope_digest,
            lineage_scope_digest=lineage_scope_digest,
            capture_manifest_digest=capture_manifest_digest,
            source_adapter=source_adapter,
        )
    return ShadowSession._from_sqlite_authorization(
        sqlite_authorization,
        run_id=run_id,
        config=config,
        installation_key=installation_key,
        redaction_policy=redaction_policy,
        capture_scope=capture_scope,
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
        source_adapter=source_adapter,
    )


def _revalidate_before_publication(
    *,
    trace: PreflightedShadowTrace,
    output: StableFileAuthorization,
    publication: AtomicFilePublication,
    sqlite: StableFileAuthorization | None,
) -> None:
    failed = False
    try:
        trace.authorization.revalidate()
        output.revalidate()
        publication.authorization.revalidate()
        if sqlite is not None:
            sqlite.revalidate()
    except Exception:
        failed = True
    if failed:
        raise ShadowCommandInputError()


def _publish_report(
    publication: AtomicFilePublication,
    report: ShadowRunReport,
    encoded: bytes,
) -> None:
    published_seen = False
    published_valid = False
    publish_returned = False
    reopened_valid = False

    def validate_reopened(data: bytes) -> bool:
        nonlocal published_seen, published_valid
        published_seen = True
        published_valid = validate_published_shadow_report(data, report)
        return published_valid

    unsupported = False
    failed = False
    try:
        reopened = publication.publish(
            encoded,
            validate_published=validate_reopened,
        )
        publish_returned = True
        decoded = decode_shadow_run_report(reopened.data)
        reopened_valid = decoded == report and reopened.data == encoded
    except SecureFileUnsupportedError:
        unsupported = True
    except Exception:
        failed = True
    if unsupported:
        raise ShadowCommandConfigurationError()
    if published_seen and not published_valid:
        raise ShadowCommandIntegrityError()
    if failed:
        if publish_returned:
            raise ShadowCommandIntegrityError()
        raise ShadowCommandInputError()
    if not published_seen or not published_valid or not reopened_valid:
        raise ShadowCommandIntegrityError()


def _read_atif_source(path: str) -> StableFileRead:
    try:
        source = read_stable_file(
            path,
            maximum_bytes=MAX_SHADOW_TRACE_BYTES,
            policy=StableReadPolicy.PRIVATE_OWNER,
        )
        identity = source.authorization._target_identity
        if identity is None or stat.S_IMODE(identity.mode) != 0o600:
            raise SecureFileError()
        return source
    except SecureFileBoundError:
        raise ShadowCommandInputError() from None
    except SecureFileUnsupportedError:
        raise ShadowCommandConfigurationError() from None
    except Exception:
        raise ShadowCommandInputError() from None


def _trace_report_matches_exactly(
    data: bytes,
    report: ShadowTraceReport,
    encoded: bytes,
) -> bool:
    return data == encoded and validate_published_shadow_trace_report(data, report)


def _trace_report_matches_preview(
    data: bytes,
    trace: ShadowTrace,
    shadow_report: ShadowRunReport,
) -> bool:
    try:
        report = decode_shadow_trace_report(data)
        return (
            type(trace) is ShadowTrace
            and type(shadow_report) is ShadowRunReport
            and report.run_id == trace.run_id
            and report.binding == trace.binding
            and report.diagnostics == trace.diagnostics
            and report.mapped_record_digest == trace.mapped_record_digest
            and report.normalized_input_digest == shadow_report.normalized_input_digest
            and report.shadow_report == shadow_report
        )
    except Exception:
        return False


def _desired_atif_report_binding(
    trace: ShadowTrace,
    *,
    normalized_input_digest: str,
    redaction_policy_tag: PayloadDigest,
    config: ShadowConfig,
) -> ShadowTraceReportBinding:
    return ShadowTraceReportBinding(
        run_id=trace.run_id,
        trace_binding=trace.binding,
        diagnostics_digest=trace.diagnostics.diagnostics_digest,
        mapped_record_digest=trace.mapped_record_digest,
        normalized_input_digest=normalized_input_digest,
        redaction_policy_tag=redaction_policy_tag,
        detector_profile_digest=config.detector_profile_digest,
    )


def _authorize_atif_output(
    location: StableFileAuthorization,
    *,
    binding: ShadowTraceReportBinding,
    replace: bool,
) -> _ATIFOutputPlan:
    replacement_data: bytes | None = None
    if replace and location._target_identity is not None:
        try:
            stable = read_stable_file(
                location.path,
                maximum_bytes=MAX_SHADOW_TRACE_REPORT_BYTES,
                policy=StableReadPolicy.PRIVATE_OWNER,
            )
            location.revalidate()
            if not validate_shadow_trace_report_binding(stable.data, binding):
                raise ShadowCommandIntegrityError()
            replacement_data = bytes(stable.data)
        except ShadowCommandIntegrityError:
            raise
        except SecureFileBoundError:
            raise ShadowCommandIntegrityError() from None
        except SecureFileUnsupportedError:
            raise ShadowCommandConfigurationError() from None
        except Exception:
            raise ShadowCommandInputError() from None

    try:
        publication = authorize_shadow_trace_report_publication(
            location.path,
            replacement_binding=binding if replace else None,
        )
        location.revalidate()
        return _ATIFOutputPlan(
            publication=publication,
            replacement_data=replacement_data,
        )
    except SecureFileUnsupportedError:
        raise ShadowCommandConfigurationError() from None
    except Exception:
        raise ShadowCommandInputError() from None


def _trace_session_sqlite_authorization(
    session: ShadowSession,
    *,
    repository_path: str | None,
) -> StableFileAuthorization | None:
    if repository_path is None:
        return None
    try:
        repository = session._repository
        authorization = getattr(repository, "_file_authorization", None)
        if type(authorization) is not StableFileAuthorization:
            raise ValueError("trace SQLite authorization is unavailable")
        return authorization
    except Exception:
        raise ShadowInvariantError() from None


def _revalidate_atif_before_publication(
    *,
    source: StableFileAuthorization,
    output: StableFileAuthorization,
    publication: AtomicFilePublication,
    sqlite: StableFileAuthorization | None,
) -> None:
    try:
        source.revalidate()
        output.revalidate()
        publication.authorization.revalidate()
        if not output.aliases(publication.authorization):
            raise SecureFileError()
        if source.aliases(output) or source.aliases(publication.authorization):
            raise SecureFileError()
        if sqlite is not None:
            sqlite.revalidate()
            if (
                source.aliases(sqlite)
                or output.aliases(sqlite)
                or publication.authorization.aliases(sqlite)
            ):
                raise SecureFileError()
    except Exception:
        raise ShadowCommandInputError() from None


def _publish_atif_report(
    publication: AtomicFilePublication,
    report: ShadowTraceReport,
    encoded: bytes,
) -> None:
    published_seen = False
    published_valid = False
    publish_returned = False

    def validate_reopened(data: bytes) -> bool:
        nonlocal published_seen, published_valid
        published_seen = True
        published_valid = _trace_report_matches_exactly(data, report, encoded)
        return published_valid

    try:
        reopened = publication.publish(encoded, validate_published=validate_reopened)
        publish_returned = True
        reopened_valid = _trace_report_matches_exactly(reopened.data, report, encoded)
    except SecureFileUnsupportedError:
        raise ShadowCommandConfigurationError() from None
    except Exception:
        if published_seen and not published_valid:
            raise ShadowCommandIntegrityError() from None
        if publish_returned:
            raise ShadowCommandIntegrityError() from None
        raise ShadowCommandInputError() from None
    if not published_seen or not published_valid or not reopened_valid:
        raise ShadowCommandIntegrityError()


def _profile_detector_evidence(
    profile_id: str,
) -> tuple[_ProfileDetectorEvidence, ...]:
    if profile_id == ATIFProfile.HARBOR_TERMINUS_2_V1.value:
        return (
            (SignalType.REPEATED_ACTION, "conditional"),
            (SignalType.REPEATED_FAILURE, "none"),
            (SignalType.TEST_FAILURE, "none"),
            (SignalType.TOOL_ERROR, "none"),
        )
    if profile_id == ATIFProfile.HARBOR_CODEX_V1.value:
        return (
            (SignalType.REPEATED_ACTION, "conditional"),
            (SignalType.REPEATED_FAILURE, "conditional"),
            (SignalType.TEST_FAILURE, "none"),
            (SignalType.TOOL_ERROR, "conditional"),
        )
    raise ShadowInvariantError()


def _atif_command_report(report: ShadowTraceReport) -> ATIFShadowCommandReport:
    diagnostics = report.diagnostics
    if type(diagnostics) is not ATIFShadowDiagnostics:
        raise ShadowInvariantError()
    tool_counts = dict(diagnostics.tool_call_disposition_counts)
    result_counts = dict(diagnostics.result_disposition_counts)
    nested = report.shadow_report
    profile_id = report.binding.adapter_profile_id
    if profile_id not in {
        ATIFProfile.HARBOR_TERMINUS_2_V1.value,
        ATIFProfile.HARBOR_CODEX_V1.value,
    }:
        raise ShadowInvariantError()
    try:
        return ATIFShadowCommandReport(
            run_id=report.run_id,
            adapter_profile_id=cast(_ATIFProfileId, profile_id),
            source_schema_version=report.binding.source_schema_version,
            timestamp_mode=report.binding.timestamp_mode,
            root_segment_only=diagnostics.root_segment_only,
            continued_trajectory_ref_present=diagnostics.continued_trajectory_ref_present,
            embedded_subagent_trajectory_count=(diagnostics.embedded_subagent_trajectory_count),
            complete_execution_session_coverage=(diagnostics.complete_execution_session_coverage),
            producer_authentication=diagnostics.producer_authentication,
            outcome_evidence_authority=diagnostics.outcome_evidence_authority,
            profile_audit_manifest_digest=diagnostics.profile_audit_manifest_digest,
            total_step_count=diagnostics.total_step_count,
            ignored_message_step_count=diagnostics.ignored_message_step_count,
            total_tool_call_count=diagnostics.total_tool_call_count,
            tool_call_disposition_counts=diagnostics.tool_call_disposition_counts,
            total_observation_result_count=diagnostics.total_observation_result_count,
            result_disposition_counts=diagnostics.result_disposition_counts,
            mapped_shadow_record_count=diagnostics.mapped_shadow_record_count,
            structured_outcome_coverage=(
                result_counts["mapped_structured_outcome"],
                tool_counts["mapped_action"],
            ),
            evaluated_unique_event_count=nested.evaluated_unique_event_count,
            heuristic_disposition_counts=nested.heuristic_disposition_counts,
            supported_signal_types=nested.supported_signal_types,
            unsupported_signal_types=nested.unsupported_signal_types,
            profile_detector_evidence=_profile_detector_evidence(profile_id),
            detector_outcome_counts=nested.detector_outcome_counts,
            abstention_reason_counts=nested.abstention_reason_counts,
            applicable_detector_evaluation_count=(nested.applicable_detector_evaluation_count),
            evidence_sufficient_applicable_detector_evaluation_count=(
                nested.evidence_sufficient_applicable_detector_evaluation_count
            ),
            report_digest=report.report_digest,
        )
    except ShadowInvariantError:
        raise
    except Exception:
        raise ShadowInvariantError() from None


def _command_report(report: ShadowRunReport) -> ShadowCommandReport:
    return ShadowCommandReport(
        run_id=report.run_id,
        input_byte_digest=report.input_byte_digest,
        normalized_input_digest=report.normalized_input_digest,
        detector_profile_digest=report.detector_profile_digest,
        supported_signal_types=report.supported_signal_types,
        unsupported_signal_types=report.unsupported_signal_types,
        unique_input_event_count=report.unique_input_event_count,
        retry_row_count=report.retry_row_count,
        heuristic_disposition_counts=report.heuristic_disposition_counts,
        report_digest=report.report_digest,
    )


async def _run_shadow_analyze(
    trace_path: str | os.PathLike[str],
    *,
    run_id: UUID,
    output_path: str | os.PathLike[str],
    repository_path: str | os.PathLike[str],
    capture_scope: CaptureScope,
    task_scope_digest: str | None,
    lineage_scope_digest: str | None,
    capture_manifest_digest: str | None,
    source_adapter: str,
    replace: bool,
) -> ShadowCommandReport:
    if type(run_id) is not UUID or run_id.version != 4 or type(replace) is not bool:
        raise ShadowCommandInputError()
    copied_trace_path = _copy_path(trace_path)
    copied_output_path = _copy_path(output_path)
    copied_repository_path = _copy_repository_path(repository_path)
    installation_key: InstallationKey | None = None
    key_failed = False
    try:
        installation_key = load_or_create_installation_key()
    except Exception:
        key_failed = True
    if key_failed or installation_key is None:
        raise ShadowCommandConfigurationError()
    redaction_policy = RedactionPolicy()
    config = ShadowConfig.reference()
    policy_tag = _redaction_policy_tag(installation_key, redaction_policy)
    trace = read_shadow_trace(
        copied_trace_path,
        run_id=run_id,
        config=config,
        installation_key=installation_key,
        redaction_policy=redaction_policy,
        redaction_policy_tag=policy_tag,
        capture_scope=capture_scope,
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
        source_adapter=source_adapter,
    )
    locations = _inspect_locations(
        trace.authorization,
        output_path=copied_output_path,
        repository_path=copied_repository_path,
    )
    binding = _desired_report_binding(
        trace,
        redaction_policy_tag=policy_tag,
        config=config,
        capture_scope=capture_scope,
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
    )
    publication = _authorize_output(
        locations.output,
        binding=binding,
        replace=replace,
    )
    sqlite_authorization = _authorize_sqlite(trace, locations, publication)
    session = _new_session(
        sqlite_authorization=sqlite_authorization,
        run_id=run_id,
        config=config,
        installation_key=installation_key,
        redaction_policy=redaction_policy,
        capture_scope=capture_scope,
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
        source_adapter=source_adapter,
    )
    closed = False
    try:
        report = await _analyze_legacy_preflighted(session, trace)
        if shadow_report_binding(report) != binding:
            raise ShadowInvariantError()
        encoded = encode_shadow_run_report(report)
        await session.aclose()
        closed = True
        _revalidate_before_publication(
            trace=trace,
            output=locations.output,
            publication=publication,
            sqlite=sqlite_authorization,
        )
        _publish_report(publication, report, encoded)
        return _command_report(report)
    finally:
        if not closed:
            await session.aclose()


async def run_shadow_analyze(
    trace_path: str | os.PathLike[str],
    *,
    run_id: UUID,
    output_path: str | os.PathLike[str],
    repository_path: str | os.PathLike[str] = _DEFAULT_REPOSITORY,
    capture_scope: CaptureScope = "unknown",
    task_scope_digest: str | None = None,
    lineage_scope_digest: str | None = None,
    capture_manifest_digest: str | None = None,
    source_adapter: str = _DEFAULT_SOURCE_ADAPTER,
    replace: bool = False,
) -> ShadowCommandReport:
    """Analyze one completely preflighted trace and publish exactly one report."""

    result: ShadowCommandReport | None = None
    failure: Exception | None = None
    try:
        result = await _run_shadow_analyze(
            trace_path,
            run_id=run_id,
            output_path=output_path,
            repository_path=repository_path,
            capture_scope=capture_scope,
            task_scope_digest=task_scope_digest,
            lineage_scope_digest=lineage_scope_digest,
            capture_manifest_digest=capture_manifest_digest,
            source_adapter=source_adapter,
            replace=replace,
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as error:
        failure = error
    if result is not None:
        return result
    if isinstance(failure, ShadowCommandIntegrityError):
        raise ShadowCommandIntegrityError()
    if isinstance(failure, ShadowCommandInputError | ShadowInputError):
        raise ShadowCommandInputError()
    if isinstance(
        failure,
        ShadowCommandConfigurationError
        | ShadowConfigurationError
        | ShadowStateError
        | RepositoryError
        | SecureFileUnsupportedError,
    ):
        raise ShadowCommandConfigurationError()
    if isinstance(failure, ShadowInvariantError):
        raise ShadowInvariantError()
    if isinstance(failure, (SecureFileError, OSError)):
        raise ShadowCommandConfigurationError()
    raise ShadowInvariantError()


async def _run_shadow_analyze_atif(
    trace_path: str | os.PathLike[str],
    *,
    profile: ATIFProfile,
    run_id: UUID,
    working_directory: str,
    environment_digest: str,
    output_path: str | os.PathLike[str],
    repository_path: str | os.PathLike[str],
    task_scope_digest: str | None,
    lineage_scope_digest: str | None,
    capture_manifest_digest: str | None,
    replace: bool,
) -> ATIFShadowCommandReport:
    if (
        type(profile) is not ATIFProfile
        or type(run_id) is not UUID
        or run_id.version != 4
        or type(replace) is not bool
    ):
        raise ShadowCommandInputError()
    copied_trace_path = _copy_path(trace_path)
    copied_output_path = _copy_path(output_path)
    copied_repository_path = _copy_repository_path(repository_path)
    try:
        environment = ShadowEnvironmentBinding(
            default_working_directory=working_directory,
            environment_digest=environment_digest,
        )
    except Exception:
        raise ShadowCommandInputError() from None

    source = _read_atif_source(copied_trace_path)
    adapter = ATIFShadowAdapter(profile=profile, environment=environment)
    trace = adapter.adapt_bytes(
        source.data,
        run_id=run_id,
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
    )

    installation_key: InstallationKey | None = None
    key_failed = False
    try:
        installation_key = load_or_create_installation_key()
    except Exception:
        key_failed = True
    if key_failed or installation_key is None:
        raise ShadowCommandConfigurationError()

    redaction_policy = RedactionPolicy()
    config = ShadowConfig.reference()
    policy_tag = _redaction_policy_tag(installation_key, redaction_policy)
    locations = _inspect_locations(
        source.authorization,
        output_path=copied_output_path,
        repository_path=copied_repository_path,
    )

    if copied_repository_path is None:
        session = ShadowSession.in_memory_for_trace(
            run_id=run_id,
            trace_binding=trace.binding,
            config=config,
            installation_key=installation_key,
            redaction_policy=redaction_policy,
        )
    else:
        try:
            source.authorization.revalidate()
            locations.output.revalidate()
            for slot in locations.sqlite_slots:
                slot.revalidate()
        except Exception:
            raise ShadowCommandInputError() from None
        session = ShadowSession._from_sqlite_authorization_for_trace(
            locations.sqlite_slots[0],
            sidecar_authorizations=locations.sqlite_slots[1:],
            run_id=run_id,
            trace_binding=trace.binding,
            config=config,
            installation_key=installation_key,
            redaction_policy=redaction_policy,
        )

    closed = False
    try:
        prepared = _prepare_analysis(session, trace)
        output_plan = _authorize_atif_output(
            locations.output,
            binding=_desired_atif_report_binding(
                trace,
                normalized_input_digest=prepared.normalized_input_digest,
                redaction_policy_tag=policy_tag,
                config=config,
            ),
            replace=replace,
        )
        preview = None
        if output_plan.replacement_data is not None:
            preview = await _preview_prepared(
                session,
                prepared,
                assume_empty=(
                    copied_repository_path is None
                    or locations.sqlite_slots[0]._target_identity is None
                ),
            )
            if not _trace_report_matches_preview(
                output_plan.replacement_data,
                trace,
                preview.shadow_report,
            ):
                raise ShadowCommandIntegrityError()
        report = (
            await _analyze_prepared(session, prepared)
            if preview is None
            else await _analyze_prepared(
                session,
                prepared,
                expected_initial_state=preview.initial_state,
            )
        )
        encoded = encode_shadow_trace_report(report)
        sqlite_authorization = _trace_session_sqlite_authorization(
            session,
            repository_path=copied_repository_path,
        )
        await session.aclose()
        closed = True
        if output_plan.replacement_data is not None and not _trace_report_matches_exactly(
            output_plan.replacement_data,
            report,
            encoded,
        ):
            raise ShadowCommandIntegrityError()
        _revalidate_atif_before_publication(
            source=source.authorization,
            output=locations.output,
            publication=output_plan.publication,
            sqlite=sqlite_authorization,
        )
        _publish_atif_report(output_plan.publication, report, encoded)
        return _atif_command_report(report)
    finally:
        if not closed:
            await session.aclose()


async def run_shadow_analyze_atif(
    trace_path: str | os.PathLike[str],
    *,
    profile: ATIFProfile,
    run_id: UUID,
    working_directory: str,
    environment_digest: str,
    output_path: str | os.PathLike[str],
    repository_path: str | os.PathLike[str] = _DEFAULT_REPOSITORY,
    task_scope_digest: str | None = None,
    lineage_scope_digest: str | None = None,
    capture_manifest_digest: str | None = None,
    replace: bool = False,
) -> ATIFShadowCommandReport:
    """Analyze one stable private ATIF trace and publish its provenance report."""

    result: ATIFShadowCommandReport | None = None
    failure: Exception | None = None
    try:
        result = await _run_shadow_analyze_atif(
            trace_path,
            profile=profile,
            run_id=run_id,
            working_directory=working_directory,
            environment_digest=environment_digest,
            output_path=output_path,
            repository_path=repository_path,
            task_scope_digest=task_scope_digest,
            lineage_scope_digest=lineage_scope_digest,
            capture_manifest_digest=capture_manifest_digest,
            replace=replace,
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as error:
        failure = error
    if result is not None:
        return result
    if isinstance(failure, ShadowCommandIntegrityError):
        raise ShadowCommandIntegrityError()
    if isinstance(
        failure,
        ShadowCommandInputError | ShadowInputError | ShadowTraceInputError,
    ):
        raise ShadowCommandInputError()
    if isinstance(
        failure,
        ShadowCommandConfigurationError
        | ShadowConfigurationError
        | ShadowStateError
        | RepositoryError
        | SecureFileUnsupportedError,
    ):
        raise ShadowCommandConfigurationError()
    if isinstance(failure, ShadowInvariantError):
        raise ShadowInvariantError()
    if isinstance(failure, (SecureFileError, OSError)):
        raise ShadowCommandConfigurationError()
    raise ShadowInvariantError()


def _copy_command_report(value: object) -> ShadowCommandReport:
    try:
        if type(value) is not ShadowCommandReport:
            raise ValueError("command report type is invalid")
        copied_fields = ShadowCommandReport.__pydantic_serializer__.to_python(
            value,
            mode="python",
            warnings=False,
        )
        copied = ShadowCommandReport.model_validate(copied_fields)
        if copied != value:
            raise ValueError("command report copy differs")
        return copied
    except Exception:
        raise ShadowInvariantError() from None


def render_shadow_json(report: ShadowCommandReport) -> str:
    validated = _copy_command_report(report)
    return canonical_json(validated.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_shadow_human(report: ShadowCommandReport) -> str:
    validated = _copy_command_report(report)
    counts = {disposition: count for disposition, count in validated.heuristic_disposition_counts}
    return (
        "Shadow analysis complete\n"
        f"run: {validated.run_id}\n"
        f"evaluated events: {validated.unique_input_event_count}\n"
        f"dispositions: flagged={counts[ShadowHeuristicDisposition.FLAGGED]}, "
        f"not_flagged={counts[ShadowHeuristicDisposition.NOT_FLAGGED]}, "
        f"indeterminate={counts[ShadowHeuristicDisposition.INDETERMINATE]}, "
        f"not_applicable={counts[ShadowHeuristicDisposition.NOT_APPLICABLE]}\n"
        f"supported detectors: {len(validated.supported_signal_types)} of "
        f"{len(validated.supported_signal_types) + len(validated.unsupported_signal_types)}\n"
        f"report digest: {validated.report_digest}\n"
        "evidence: descriptive observational; no decision authority\n"
    )


def _copy_atif_command_report(value: object) -> ATIFShadowCommandReport:
    try:
        if type(value) is not ATIFShadowCommandReport:
            raise ValueError("ATIF command report type is invalid")
        copied_fields = ATIFShadowCommandReport.__pydantic_serializer__.to_python(
            value,
            mode="python",
            warnings=False,
        )
        copied = ATIFShadowCommandReport.model_validate(copied_fields)
        if copied != value:
            raise ValueError("ATIF command report copy differs")
        return copied
    except Exception:
        raise ShadowInvariantError() from None


def render_shadow_atif_json(report: ATIFShadowCommandReport) -> str:
    validated = _copy_atif_command_report(report)
    return canonical_json(validated.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_shadow_atif_human(report: ATIFShadowCommandReport) -> str:
    validated = _copy_atif_command_report(report)
    tool_counts = dict(validated.tool_call_disposition_counts)
    result_counts = dict(validated.result_disposition_counts)
    disposition_counts = dict(validated.heuristic_disposition_counts)
    detector_counts = {
        (signal_type, status): count
        for signal_type, status, count in validated.detector_outcome_counts
    }
    abstention_counts = {
        (signal_type, reason): count
        for signal_type, reason, count in validated.abstention_reason_counts
    }
    coverage_numerator, coverage_denominator = validated.structured_outcome_coverage
    coverage = (
        "not_available (0/0)"
        if coverage_denominator == 0
        else f"{coverage_numerator}/{coverage_denominator}"
    )
    detector_lines = "".join(
        (
            f"detector {signal_type.value}: "
            f"detected={detector_counts[(signal_type, DetectionStatus.DETECTED)]}, "
            f"no_match={detector_counts[(signal_type, DetectionStatus.NO_MATCH)]}, "
            f"abstained={detector_counts[(signal_type, DetectionStatus.ABSTAINED)]}; "
            "abstentions="
            + ",".join(
                f"{reason.value}:{abstention_counts[(signal_type, reason)]}"
                for reason in _ABSTENTION_REASONS
            )
            + "\n"
        )
        for signal_type in validated.supported_signal_types
    )
    return (
        "ATIF Shadow analysis complete\n"
        f"run: {validated.run_id}\n"
        f"profile: {validated.adapter_profile_id}\n"
        f"source schema: {validated.source_schema_version}\n"
        "scope: root_segment_only=true, "
        f"continued_trajectory={str(validated.continued_trajectory_ref_present).lower()}, "
        f"embedded_subagents={validated.embedded_subagent_trajectory_count}, "
        "complete_execution_session_coverage=false\n"
        f"source totals: steps={validated.total_step_count}, "
        f"tool_calls={validated.total_tool_call_count}, "
        f"observation_results={validated.total_observation_result_count}, "
        f"message_only_steps={validated.ignored_message_step_count}\n"
        f"mapped: actions={tool_counts['mapped_action']}, "
        f"structured_outcomes={result_counts['mapped_structured_outcome']}, "
        f"structured_outcome_coverage={coverage}, "
        f"shadow_records={validated.mapped_shadow_record_count}\n"
        "ignored calls: "
        f"unsupported={tool_counts['ignored_unsupported_function']}, "
        f"continuation={tool_counts['ignored_continuation']}, "
        f"non_command_wait={tool_counts['ignored_non_command_wait']}, "
        f"unsubmitted={tool_counts['ignored_unsubmitted_keystrokes']}, "
        f"unresolved_submission={tool_counts['ignored_unresolved_terminal_submission']}, "
        f"copied_context={tool_counts['ignored_copied_context']}\n"
        "ignored results: "
        f"evidence_absent={result_counts['ignored_evidence_absent']}, "
        f"ambiguous_parent={result_counts['ignored_ambiguous_parent']}, "
        f"no_parent={result_counts['ignored_no_parent']}, "
        f"unsupported_parent={result_counts['ignored_unsupported_parent']}, "
        f"copied_context={result_counts['ignored_copied_context']}\n"
        f"evaluated events: {validated.evaluated_unique_event_count}\n"
        f"dispositions: flagged={disposition_counts[ShadowHeuristicDisposition.FLAGGED]}, "
        f"not_flagged={disposition_counts[ShadowHeuristicDisposition.NOT_FLAGGED]}, "
        f"indeterminate={disposition_counts[ShadowHeuristicDisposition.INDETERMINATE]}, "
        f"not_applicable={disposition_counts[ShadowHeuristicDisposition.NOT_APPLICABLE]}\n"
        f"engine detectors: {len(validated.supported_signal_types)} of "
        f"{len(validated.supported_signal_types) + len(validated.unsupported_signal_types)}; "
        + ",".join(item.value for item in validated.supported_signal_types)
        + "\nprofile detector evidence: "
        + ",".join(
            f"{signal_type.value}={availability}"
            for signal_type, availability in validated.profile_detector_evidence
        )
        + "\n"
        + detector_lines
        + f"evidence-sufficient applicable detector evaluations: "
        f"{validated.evidence_sufficient_applicable_detector_evaluation_count}/"
        f"{validated.applicable_detector_evaluation_count}\n"
        f"timestamps: {validated.timestamp_mode}\n"
        f"producer authentication: {validated.producer_authentication}\n"
        f"outcome evidence authority: {validated.outcome_evidence_authority}\n"
        f"compatibility evidence manifest digest: "
        f"{validated.profile_audit_manifest_digest}\n"
        f"report digest: {validated.report_digest}\n"
        "evidence: descriptive observational; no decision authority\n"
    )


__all__ = [
    "ATIF_SHADOW_COMMAND_REPORT_SCHEMA_VERSION",
    "SHADOW_COMMAND_REPORT_SCHEMA_VERSION",
    "ATIFShadowCommandReport",
    "ShadowCommandConfigurationError",
    "ShadowCommandInputError",
    "ShadowCommandIntegrityError",
    "ShadowCommandReport",
    "render_shadow_atif_human",
    "render_shadow_atif_json",
    "render_shadow_human",
    "render_shadow_json",
    "run_shadow_analyze",
    "run_shadow_analyze_atif",
]
