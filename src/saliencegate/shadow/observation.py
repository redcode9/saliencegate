from __future__ import annotations

import hmac
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from itertools import chain, pairwise
from types import MappingProxyType
from typing import Annotated, Literal, Self, TypeVar
from uuid import UUID
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import (
    EventPhase,
    EventType,
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    Signal,
    SignalType,
    TraceEvent,
    TrustLabel,
    canonical_json,
    length_prefixed_sha256,
    trace_event_payload_is_bounded,
)
from saliencegate.domain.records import UUID4, Sha256Digest
from saliencegate.shadow.config import ShadowApplicability, ShadowConfig, ShadowDetectorSpec
from saliencegate.shadow.errors import ShadowInvariantError
from saliencegate.shadow.evaluation import (
    ShadowHeuristicDisposition,
    ShadowHeuristicEvaluation,
    evaluate_shadow_heuristic,
)
from saliencegate.shadow.inputs import (
    SHADOW_PROJECTION_MATRIX,
    ShadowEventRef,
    ShadowInputKind,
    derive_shadow_event_id,
    derive_shadow_source_event_digest,
)
from saliencegate.signals import (
    AbstentionReason,
    DetectionContext,
    DetectionOutcome,
    DetectionStatus,
    DetectorEvaluation,
    ExtractionReport,
)
from saliencegate.signals.base import (
    _detection_sequence_proof_is_exact,
    _DetectionSequenceProof,
    _trusted_extraction_is_exact,
    _TrustedExtraction,
)

SHADOW_OBSERVATION_SCHEMA_VERSION: Literal["shadow-observation/v1"] = "shadow-observation/v1"
SHADOW_EVENT_RESULT_SCHEMA_VERSION: Literal["shadow-event-result/v1"] = "shadow-event-result/v1"

_REDACTED_EVENT_DIGEST_DOMAIN = "saliencegate:shadow:redacted-event:v1"
_EVENT_PREFIX_DIGEST_DOMAIN = "saliencegate:shadow:event-prefix:v1"
_DETECTION_CONTEXT_DIGEST_DOMAIN = "saliencegate:shadow:detection-context:v1"
_EXTRACTION_REPORT_DIGEST_DOMAIN = "saliencegate:shadow:extraction-report:v1"
_FEATURE_SNAPSHOT_DIGEST_DOMAIN = "saliencegate:shadow:feature-snapshot:v1"
_OBSERVATION_DIGEST_DOMAIN = "saliencegate:shadow:observation:v1"
_MAX_SHADOW_EVENTS = 10_000
_MAX_SIGNED_64 = (1 << 63) - 1

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

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_DigestPart = str | bytes
_SELECTION_TOKEN = object()
_TRUSTED_OBSERVATION_SEQUENCE_TOKEN = object()


class _TrustedObservationSeal:
    __slots__ = ("__weakref__",)


_TRUSTED_OBSERVATION_SEQUENCE_SEALS: WeakKeyDictionary[
    _TrustedObservationSeal,
    tuple[object, ...],
] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class _SelectedDetectionContext:
    prefix: tuple[TraceEvent, ...]
    context: DetectionContext
    _token: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _TrustedShadowObservationSequence:
    sequence: _DetectionSequenceProof = field(repr=False)
    config: ShadowConfig = field(repr=False)
    redaction_policy_tag: PayloadDigest = field(repr=False)
    _token: object = field(repr=False, compare=False)
    _seal: _TrustedObservationSeal = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return f"_TrustedShadowObservationSequence(event_count={len(self.sequence.events)})"


class _ShadowObservationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _length_prefixed_sha256_iter(
    parts: Iterable[_DigestPart],
    *,
    domain: _DigestPart,
) -> str:
    """Hash framed parts incrementally without materializing an entire event prefix."""

    digest = sha256()
    for part in chain((domain,), parts):
        encoded = part.encode("utf-8") if isinstance(part, str) else part
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _model_state_is_exact(model_type: type[BaseModel], value: object) -> bool:
    try:
        return (
            type(value) is model_type
            and type(value.__dict__) is dict
            and set(value.__dict__) == set(model_type.model_fields)
            and value.__pydantic_extra__ is None
            and value.__pydantic_private__ is None
        )
    except Exception:
        return False


def _is_uuid4(value: object) -> bool:
    return type(value) is UUID and value.version == 4


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _payload_digest_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(PayloadDigest, value):
        return False
    assert isinstance(value, PayloadDigest)
    try:
        return type(value.algorithm) is PayloadDigestAlgorithm and _is_digest(value.value)
    except Exception:
        return False


def _trace_event_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(TraceEvent, value):
        return False
    assert isinstance(value, TraceEvent)
    try:
        return (
            value.schema_version == "1.0"
            and type(value.schema_version) is str
            and value.record_type == "trace_event"
            and type(value.record_type) is str
            and _is_uuid4(value.event_id)
            and _is_uuid4(value.run_id)
            and type(value.sequence) is int
            and 1 <= value.sequence <= _MAX_SIGNED_64
            and type(value.source_event_id) is str
            and 1 <= len(value.source_event_id) <= 256
            and type(value.timestamp) is datetime
            and type(value.event_type) is EventType
            and type(value.phase) is EventPhase
            and type(value.payload) in (dict, MappingProxyType)
            and trace_event_payload_is_bounded(value.payload)
            and _payload_digest_is_preflight_safe(value.payload_digest)
            and type(value.parent_ids) is tuple
            and len(value.parent_ids) <= 64
            and all(_is_uuid4(parent_id) for parent_id in value.parent_ids)
            and type(value.source_adapter) is str
            and 1 <= len(value.source_adapter) <= 256
            and type(value.trust_label) is TrustLabel
        )
    except Exception:
        return False


def _outcome_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(DetectionOutcome, value):
        return False
    assert isinstance(value, DetectionOutcome)
    try:
        return (
            type(value.signal_type) is SignalType
            and type(value.status) is DetectionStatus
            and (
                value.strength is None
                or (type(value.strength) is float and math.isfinite(value.strength))
            )
            and (value.reason_code is None or type(value.reason_code) is ReasonCode)
            and type(value.evidence_event_ids) is tuple
            and len(value.evidence_event_ids) <= _MAX_SHADOW_EVENTS
            and all(_is_uuid4(item) for item in value.evidence_event_ids)
            and type(value.related_event_ids) is tuple
            and len(value.related_event_ids) <= _MAX_SHADOW_EVENTS
            and all(_is_uuid4(item) for item in value.related_event_ids)
            and (
                value.abstention_reason is None or type(value.abstention_reason) is AbstentionReason
            )
        )
    except Exception:
        return False


def _detector_evaluation_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(DetectorEvaluation, value):
        return False
    assert isinstance(value, DetectorEvaluation)
    try:
        return (
            type(value.signal_type) is SignalType
            and type(value.detector_version) is str
            and 1 <= len(value.detector_version) <= 256
            and _outcome_is_preflight_safe(value.outcome)
        )
    except Exception:
        return False


def _signal_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(Signal, value):
        return False
    assert isinstance(value, Signal)
    try:
        return (
            value.schema_version == "1.0"
            and type(value.schema_version) is str
            and value.record_type == "signal"
            and type(value.record_type) is str
            and _is_uuid4(value.signal_id)
            and _is_uuid4(value.run_id)
            and type(value.created_at) is datetime
            and type(value.signal_type) is SignalType
            and type(value.strength) is float
            and math.isfinite(value.strength)
            and type(value.evidence_event_ids) is tuple
            and 1 <= len(value.evidence_event_ids) <= 64
            and all(_is_uuid4(item) for item in value.evidence_event_ids)
            and type(value.detector_version) is str
            and 1 <= len(value.detector_version) <= 256
            and type(value.reason_code) is ReasonCode
        )
    except Exception:
        return False


def _extraction_report_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(ExtractionReport, value):
        return False
    assert isinstance(value, ExtractionReport)
    try:
        return (
            _is_uuid4(value.run_id)
            and _is_uuid4(value.current_event_id)
            and type(value.current_event_timestamp) is datetime
            and type(value.evaluations) is tuple
            and len(value.evaluations) == len(_SUPPORTED_SIGNAL_TYPES)
            and all(_detector_evaluation_is_preflight_safe(item) for item in value.evaluations)
            and type(value.signals) is tuple
            and len(value.signals) <= len(_SUPPORTED_SIGNAL_TYPES)
            and all(_signal_is_preflight_safe(item) for item in value.signals)
        )
    except Exception:
        return False


def _shadow_config_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(ShadowConfig, value):
        return False
    assert isinstance(value, ShadowConfig)
    try:
        return (
            type(value.schema_version) is str
            and type(value.detectors) is tuple
            and len(value.detectors) == 4
            and all(
                _model_state_is_exact(ShadowDetectorSpec, item)
                and type(item.signal_type) is SignalType
                and type(item.detector_version) is str
                and len(item.detector_version) <= 256
                and (
                    item.repetition_window_events is None
                    or type(item.repetition_window_events) is int
                )
                for item in value.detectors
            )
            and type(value.supported_signal_types) is tuple
            and len(value.supported_signal_types) == 4
            and all(type(item) is SignalType for item in value.supported_signal_types)
            and type(value.unsupported_signal_types) is tuple
            and len(value.unsupported_signal_types) == 5
            and all(type(item) is SignalType for item in value.unsupported_signal_types)
            and type(value.applicability) is tuple
            and len(value.applicability) == 7
            and all(
                _model_state_is_exact(ShadowApplicability, item)
                and type(item.input_kind) is ShadowInputKind
                and type(item.applicable_signal_types) is tuple
                and len(item.applicable_signal_types) <= 4
                and all(
                    type(signal_type) is SignalType for signal_type in item.applicable_signal_types
                )
                for item in value.applicability
            )
            and type(value.evaluator_id) is str
            and len(value.evaluator_id) <= 256
            and type(value.indeterminate_reasons) is tuple
            and len(value.indeterminate_reasons) == 7
            and all(type(item) is AbstentionReason for item in value.indeterminate_reasons)
            and _is_digest(value.evaluator_configuration_digest)
            and _is_digest(value.detector_profile_digest)
        )
    except Exception:
        return False


def _heuristic_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(ShadowHeuristicEvaluation, value):
        return False
    assert isinstance(value, ShadowHeuristicEvaluation)
    try:
        return (
            type(value.schema_version) is str
            and type(value.evaluator_id) is str
            and len(value.evaluator_id) <= 256
            and _is_digest(value.configuration_digest)
            and type(value.scope) is str
            and type(value.disposition) is ShadowHeuristicDisposition
            and type(value.reason_codes) is tuple
            and len(value.reason_codes) <= 7
            and all(type(item) is AbstentionReason for item in value.reason_codes)
            and _is_digest(value.feature_snapshot_digest)
            and type(value.applicable_detector_count) is int
            and type(value.evidence_sufficient_detector_count) is int
            and type(value.incomplete_detector_types) is tuple
            and len(value.incomplete_detector_types) <= 4
            and all(type(item) is SignalType for item in value.incomplete_detector_types)
            and type(value.calibrated) is bool
            and type(value.decision_authority) is bool
        )
    except Exception:
        return False


def _event_ref_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(ShadowEventRef, value):
        return False
    assert isinstance(value, ShadowEventRef)
    try:
        return (
            type(value.schema_version) is str
            and _is_uuid4(value.run_id)
            and _is_uuid4(value.event_id)
            and type(value.sequence) is int
            and 1 <= value.sequence <= _MAX_SIGNED_64
        )
    except Exception:
        return False


def _model_is_preflight_safe(model_type: type[BaseModel], value: object) -> bool:
    if model_type is PayloadDigest:
        return _payload_digest_is_preflight_safe(value)
    if model_type is TraceEvent:
        return _trace_event_is_preflight_safe(value)
    if model_type is DetectorEvaluation:
        return _detector_evaluation_is_preflight_safe(value)
    if model_type is Signal:
        return _signal_is_preflight_safe(value)
    if model_type is ExtractionReport:
        return _extraction_report_is_preflight_safe(value)
    if model_type is ShadowConfig:
        return _shadow_config_is_preflight_safe(value)
    if model_type is ShadowHeuristicEvaluation:
        return _heuristic_is_preflight_safe(value)
    if model_type is ShadowEventRef:
        return _event_ref_is_preflight_safe(value)
    if model_type is ShadowObservation:
        return _shadow_observation_is_preflight_safe(value)
    return False


def _copy_exact_model(model_type: type[_ModelT], value: object) -> _ModelT:
    if not _model_is_preflight_safe(model_type, value):
        raise ValueError("nested shadow record failed preflight validation")
    serialized = model_type.__pydantic_serializer__.to_json(value, warnings=False)
    copied = model_type.model_validate_json(serialized)
    if copied != value:
        raise ValueError("nested shadow record failed defensive validation")
    return copied


def _copy_payload_digest(value: object) -> PayloadDigest:
    copied = _copy_exact_model(PayloadDigest, value)
    if copied.algorithm is not PayloadDigestAlgorithm.HMAC_SHA256:
        raise ValueError("redaction policy identity must use HMAC-SHA-256")
    return copied


def _copy_detector_evaluations(value: object) -> tuple[DetectorEvaluation, ...]:
    if type(value) is not tuple or len(value) != len(_SUPPORTED_SIGNAL_TYPES):
        raise ValueError("shadow observations require every supported detector evaluation")
    return tuple(_copy_exact_model(DetectorEvaluation, item) for item in value)


def _copy_signals(value: object) -> tuple[Signal, ...]:
    if type(value) is not tuple or len(value) > len(_SUPPORTED_SIGNAL_TYPES):
        raise ValueError("shadow observation signals are invalid")
    return tuple(_copy_exact_model(Signal, item) for item in value)


def _copy_heuristic_evaluations(value: object) -> tuple[ShadowHeuristicEvaluation, ...]:
    if type(value) is not tuple or len(value) != 1:
        raise ValueError("shadow observations require exactly one heuristic evaluation")
    return (_copy_exact_model(ShadowHeuristicEvaluation, value[0]),)


def _copy_signal_types(value: object) -> tuple[SignalType, ...]:
    if type(value) is not tuple or not all(type(item) is SignalType for item in value):
        raise ValueError("shadow signal type declarations are invalid")
    return tuple(value)


def _copy_trace_event(value: object) -> TraceEvent:
    return _copy_exact_model(TraceEvent, value)


def _copy_event_prefix(value: object) -> tuple[TraceEvent, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= _MAX_SHADOW_EVENTS:
        raise ValueError("shadow event prefix is invalid")
    copied = tuple(_copy_trace_event(item) for item in value)
    run_id = copied[0].run_id
    if copied[0].sequence != 1 or any(event.run_id != run_id for event in copied):
        raise ValueError("shadow event prefix does not identify one complete run prefix")
    if any(right.sequence != left.sequence + 1 for left, right in pairwise(copied)):
        raise ValueError("shadow event prefix is not contiguous")
    if any(right.timestamp < left.timestamp for left, right in pairwise(copied)):
        raise ValueError("shadow event prefix timestamps are not monotonic")
    event_ids = tuple(event.event_id for event in copied)
    source_ids = tuple(event.source_event_id for event in copied)
    if len(set(event_ids)) != len(event_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("shadow event prefix identities are not unique")
    if any(
        event.event_id != derive_shadow_event_id(event.run_id, event.source_event_id)
        for event in copied
    ):
        raise ValueError("shadow event prefix contains a noncanonical event identity")
    return copied


def _copy_detection_context(value: object) -> DetectionContext:
    if not _model_state_is_exact(DetectionContext, value):
        raise ValueError("shadow detection context failed preflight validation")
    assert isinstance(value, DetectionContext)
    if (
        not _is_uuid4(value.run_id)
        or type(value.events) is not tuple
        or not 1 <= len(value.events) <= _MAX_SHADOW_EVENTS
        or not all(_trace_event_is_preflight_safe(event) for event in value.events)
    ):
        raise ValueError("shadow detection context failed preflight validation")
    bounded = DetectionContext(run_id=value.run_id, events=value.events)
    if bounded != value:
        raise ValueError("shadow detection context failed bounded validation")
    serialized = DetectionContext.__pydantic_serializer__.to_json(
        bounded,
        warnings=False,
    )
    copied = DetectionContext.model_validate_json(serialized)
    if copied != value:
        raise ValueError("shadow detection context failed defensive validation")
    return copied


def _copy_extraction_report(value: object) -> ExtractionReport:
    return _copy_exact_model(ExtractionReport, value)


def _copy_shadow_config(value: object) -> ShadowConfig:
    return _copy_exact_model(ShadowConfig, value)


def _copy_heuristic(value: object) -> ShadowHeuristicEvaluation:
    return _copy_exact_model(ShadowHeuristicEvaluation, value)


def _validate_feature_inputs(
    *,
    prefix: object,
    context: object,
    report: object,
    config: object,
) -> tuple[tuple[TraceEvent, ...], DetectionContext, ExtractionReport, ShadowConfig]:
    copied_prefix = _copy_event_prefix(prefix)
    copied_context = _copy_detection_context(context)
    copied_report = _copy_extraction_report(report)
    copied_config = _copy_shadow_config(config)
    current = copied_prefix[-1]
    if copied_context.run_id != current.run_id or copied_context.current != current:
        raise ValueError("shadow context does not end at the observed event")
    if copied_context.events != copied_prefix[-len(copied_context.events) :]:
        raise ValueError("shadow context is not a suffix of the event prefix")
    if (
        copied_report.run_id != current.run_id
        or copied_report.current_event_id != current.event_id
        or copied_report.current_event_timestamp != current.timestamp
    ):
        raise ValueError("shadow extraction report does not identify the observed event")
    context_order = {event.event_id: ordinal for ordinal, event in enumerate(copied_context.events)}
    for evaluation in copied_report.evaluations:
        outcome = evaluation.outcome
        referenced = (
            outcome.evidence_event_ids
            if outcome.status is DetectionStatus.DETECTED
            else outcome.related_event_ids
        )
        if (
            any(event_id not in context_order for event_id in referenced)
            or tuple(sorted(referenced, key=context_order.__getitem__)) != referenced
        ):
            raise ValueError("shadow detector evidence is outside the selected context")
    if copied_config.supported_signal_types != _SUPPORTED_SIGNAL_TYPES:
        raise ValueError("shadow supported detector set is invalid")
    if copied_config.unsupported_signal_types != _UNSUPPORTED_SIGNAL_TYPES:
        raise ValueError("shadow unsupported detector set is invalid")
    evaluation_types = tuple(item.signal_type for item in copied_report.evaluations)
    if evaluation_types != copied_config.supported_signal_types:
        raise ValueError("shadow extraction report does not cover the detector profile")
    detector_versions = {
        item.signal_type: item.detector_version for item in copied_report.evaluations
    }
    configured_versions = {
        item.signal_type: item.detector_version for item in copied_config.detectors
    }
    if detector_versions != configured_versions:
        raise ValueError("shadow extraction report detector versions do not match the profile")
    return copied_prefix, copied_context, copied_report, copied_config


def _redacted_event_digest(event: TraceEvent) -> str:
    return length_prefixed_sha256(
        canonical_json(event),
        domain=_REDACTED_EVENT_DIGEST_DOMAIN,
    )


def _event_prefix_digest(prefix: tuple[TraceEvent, ...]) -> str:
    parts: Iterable[_DigestPart] = chain(
        (str(prefix[0].run_id), str(prefix[-1].sequence)),
        (canonical_json(event) for event in prefix),
    )
    return _length_prefixed_sha256_iter(
        parts,
        domain=_EVENT_PREFIX_DIGEST_DOMAIN,
    )


def _detection_context_digest(context: DetectionContext) -> str:
    parts: Iterable[_DigestPart] = chain(
        (
            str(context.run_id),
            str(context.events[0].sequence),
            str(context.current.sequence),
        ),
        (canonical_json(event) for event in context.events),
    )
    return _length_prefixed_sha256_iter(
        parts,
        domain=_DETECTION_CONTEXT_DIGEST_DOMAIN,
    )


def _extraction_report_digest(report: ExtractionReport) -> str:
    return length_prefixed_sha256(
        canonical_json(report),
        domain=_EXTRACTION_REPORT_DIGEST_DOMAIN,
    )


def _feature_snapshot_digest(
    *,
    prefix: tuple[TraceEvent, ...],
    context: DetectionContext,
    report: ExtractionReport,
    config: ShadowConfig,
) -> str:
    parts: Iterable[_DigestPart] = chain(
        (
            SHADOW_OBSERVATION_SCHEMA_VERSION,
            str(prefix[0].run_id),
            str(prefix[-1].sequence),
            config.detector_profile_digest,
            config.evaluator_configuration_digest,
            str(context.events[0].sequence),
            str(context.current.sequence),
        ),
        (canonical_json(event) for event in context.events),
        (canonical_json(event) for event in prefix),
        (canonical_json(evaluation) for evaluation in report.evaluations),
    )
    return _length_prefixed_sha256_iter(
        parts,
        domain=_FEATURE_SNAPSHOT_DIGEST_DOMAIN,
    )


def derive_shadow_redacted_event_digest(event: TraceEvent) -> str:
    result: str | None = None
    try:
        result = _redacted_event_digest(_copy_trace_event(event))
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def derive_shadow_event_prefix_digest(prefix: tuple[TraceEvent, ...]) -> str:
    result: str | None = None
    try:
        result = _event_prefix_digest(_copy_event_prefix(prefix))
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def derive_shadow_detection_context_digest(context: DetectionContext) -> str:
    result: str | None = None
    try:
        result = _detection_context_digest(_copy_detection_context(context))
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def derive_shadow_extraction_report_digest(report: ExtractionReport) -> str:
    result: str | None = None
    try:
        result = _extraction_report_digest(_copy_extraction_report(report))
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def derive_shadow_feature_snapshot_digest(
    *,
    prefix: tuple[TraceEvent, ...],
    context: DetectionContext,
    report: ExtractionReport,
    config: ShadowConfig,
) -> str:
    result: str | None = None
    try:
        checked = _validate_feature_inputs(
            prefix=prefix,
            context=context,
            report=report,
            config=config,
        )
        result = _feature_snapshot_digest(
            prefix=checked[0],
            context=checked[1],
            report=checked[2],
            config=checked[3],
        )
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


class _ShadowObservationBody(_ShadowObservationModel):
    schema_version: Literal["shadow-observation/v1"] = SHADOW_OBSERVATION_SCHEMA_VERSION
    run_id: UUID4 = Field(repr=False)
    event_id: UUID4 = Field(repr=False)
    source_event_digest: Sha256Digest = Field(repr=False)
    sequence: Annotated[int, Field(ge=1, le=_MAX_SIGNED_64)] = Field(repr=False)
    event_prefix_digest: Sha256Digest
    context_first_sequence: Annotated[int, Field(ge=1, le=_MAX_SIGNED_64)]
    context_last_sequence: Annotated[int, Field(ge=1, le=_MAX_SIGNED_64)]
    context_event_count: Annotated[int, Field(ge=1, le=_MAX_SHADOW_EVENTS)]
    context_truncated: bool
    detection_context_digest: Sha256Digest
    redacted_event_digest: Sha256Digest
    redaction_policy_tag: PayloadDigest
    detector_profile_digest: Sha256Digest
    evaluator_configuration_digest: Sha256Digest
    extraction_report_digest: Sha256Digest
    feature_snapshot_digest: Sha256Digest
    supported_signal_types: tuple[SignalType, ...]
    unsupported_signal_types: tuple[SignalType, ...]
    detector_evaluations: tuple[DetectorEvaluation, ...]
    detected_signals: tuple[Signal, ...]
    heuristic_evaluations: tuple[ShadowHeuristicEvaluation, ...]
    cli_input_ordinal: Annotated[int, Field(ge=1, le=_MAX_SIGNED_64)] | None = None
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

    @field_validator("redaction_policy_tag")
    @classmethod
    def copy_redaction_policy_tag(cls, value: object) -> PayloadDigest:
        return _copy_payload_digest(value)

    @field_validator("supported_signal_types", "unsupported_signal_types")
    @classmethod
    def copy_signal_type_tuple(cls, value: object) -> tuple[SignalType, ...]:
        return _copy_signal_types(value)

    @field_validator("detector_evaluations")
    @classmethod
    def copy_detector_results(cls, value: object) -> tuple[DetectorEvaluation, ...]:
        return _copy_detector_evaluations(value)

    @field_validator("detected_signals")
    @classmethod
    def copy_detected_signals(cls, value: object) -> tuple[Signal, ...]:
        return _copy_signals(value)

    @field_validator("heuristic_evaluations")
    @classmethod
    def copy_heuristics(cls, value: object) -> tuple[ShadowHeuristicEvaluation, ...]:
        return _copy_heuristic_evaluations(value)

    @model_validator(mode="after")
    def fields_form_one_non_authoritative_observation(self) -> Self:
        if self.context_last_sequence != self.sequence:
            raise ValueError("shadow observation context does not end at the event sequence")
        if self.context_event_count != self.context_last_sequence - self.context_first_sequence + 1:
            raise ValueError("shadow observation context count disagrees with its boundaries")
        if self.context_truncated is not (self.context_first_sequence > 1):
            raise ValueError("shadow observation context truncation flag is inconsistent")
        if self.supported_signal_types != _SUPPORTED_SIGNAL_TYPES:
            raise ValueError("shadow observation supported detector set is invalid")
        if self.unsupported_signal_types != _UNSUPPORTED_SIGNAL_TYPES:
            raise ValueError("shadow observation unsupported detector set is invalid")
        evaluation_types = tuple(item.signal_type for item in self.detector_evaluations)
        if evaluation_types != self.supported_signal_types:
            raise ValueError("shadow observation does not contain the complete detector profile")
        detected = {
            item.signal_type: item
            for item in self.detector_evaluations
            if item.outcome.status is DetectionStatus.DETECTED
        }
        signals = {signal.signal_type: signal for signal in self.detected_signals}
        if (
            len(signals) != len(self.detected_signals)
            or tuple(signal.signal_type for signal in self.detected_signals) != tuple(detected)
            or signals.keys() != detected.keys()
        ):
            raise ValueError("shadow observation signals disagree with detector evaluations")
        for signal_type, signal in signals.items():
            evaluation = detected[signal_type]
            outcome = evaluation.outcome
            if (
                signal.run_id != self.run_id
                or signal.detector_version != evaluation.detector_version
                or signal.strength != outcome.strength
                or signal.reason_code is not outcome.reason_code
                or signal.evidence_event_ids != outcome.evidence_event_ids
                or signal.evidence_event_ids[-1] != self.event_id
            ):
                raise ValueError("shadow observation signal attribution is inconsistent")
        heuristic = self.heuristic_evaluations[0]
        if (
            heuristic.configuration_digest != self.evaluator_configuration_digest
            or heuristic.feature_snapshot_digest != self.feature_snapshot_digest
            or heuristic.calibrated is not False
            or heuristic.decision_authority is not False
        ):
            raise ValueError("shadow observation heuristic identity is inconsistent")
        return self


def _observation_body_digest(value: _ShadowObservationBody) -> str:
    serializer = (
        ShadowObservation.__pydantic_serializer__
        if type(value) is ShadowObservation
        else _ShadowObservationBody.__pydantic_serializer__
    )
    fields = serializer.to_python(
        value,
        mode="json",
        exclude={"observation_digest"},
        warnings=False,
    )
    return length_prefixed_sha256(
        canonical_json(fields),
        domain=_OBSERVATION_DIGEST_DOMAIN,
    )


class ShadowObservation(_ShadowObservationBody):
    """One immutable, payload-free, non-authoritative event observation."""

    observation_digest: Sha256Digest

    @model_validator(mode="after")
    def observation_digest_matches_every_other_field(self) -> Self:
        expected = _observation_body_digest(self)
        if not hmac.compare_digest(self.observation_digest, expected):
            raise ValueError("shadow observation digest does not match its canonical fields")
        return self


def _shadow_observation_is_preflight_safe(value: object) -> bool:
    if not _model_state_is_exact(ShadowObservation, value):
        return False
    assert isinstance(value, ShadowObservation)
    try:
        digest_fields = (
            value.source_event_digest,
            value.event_prefix_digest,
            value.detection_context_digest,
            value.redacted_event_digest,
            value.detector_profile_digest,
            value.evaluator_configuration_digest,
            value.extraction_report_digest,
            value.feature_snapshot_digest,
            value.observation_digest,
        )
        return (
            _is_uuid4(value.run_id)
            and _is_uuid4(value.event_id)
            and all(_is_digest(item) for item in digest_fields)
            and type(value.sequence) is int
            and 1 <= value.sequence <= _MAX_SIGNED_64
            and type(value.context_first_sequence) is int
            and type(value.context_last_sequence) is int
            and type(value.context_event_count) is int
            and type(value.context_truncated) is bool
            and _payload_digest_is_preflight_safe(value.redaction_policy_tag)
            and type(value.supported_signal_types) is tuple
            and len(value.supported_signal_types) == 4
            and all(type(item) is SignalType for item in value.supported_signal_types)
            and type(value.unsupported_signal_types) is tuple
            and len(value.unsupported_signal_types) == 5
            and all(type(item) is SignalType for item in value.unsupported_signal_types)
            and type(value.detector_evaluations) is tuple
            and len(value.detector_evaluations) == 4
            and all(
                _detector_evaluation_is_preflight_safe(item) for item in value.detector_evaluations
            )
            and type(value.detected_signals) is tuple
            and len(value.detected_signals) <= 4
            and all(_signal_is_preflight_safe(item) for item in value.detected_signals)
            and type(value.heuristic_evaluations) is tuple
            and len(value.heuristic_evaluations) == 1
            and _heuristic_is_preflight_safe(value.heuristic_evaluations[0])
            and (
                value.cli_input_ordinal is None
                or (
                    type(value.cli_input_ordinal) is int
                    and 1 <= value.cli_input_ordinal <= _MAX_SIGNED_64
                )
            )
            and type(value.schema_version) is str
            and type(value.execution_mode) is str
            and type(value.evidence_level) is str
            and type(value.task_outcome_evidence) is str
            and type(value.intervention_outcome_evidence) is str
            and type(value.confirmatory) is bool
            and type(value.calibrated) is bool
            and type(value.calibration_eligible) is bool
            and type(value.decision_authority) is bool
            and type(value.representativeness_supported) is bool
            and type(value.task_efficacy_supported) is bool
            and type(value.counterfactual_effect_supported) is bool
            and type(value.model_calls) is int
            and type(value.budget_reservations) is int
            and type(value.cycles_created) is int
            and type(value.memory_revisions) is int
            and type(value.interventions) is int
            and type(value.delivery_authorizations) is int
            and type(value.deliveries) is int
            and type(value.intervention_outcomes) is int
        )
    except Exception:
        return False


class ShadowEventResult(_ShadowObservationModel):
    """The exact persisted event reference paired with its immutable observation."""

    schema_version: Literal["shadow-event-result/v1"] = SHADOW_EVENT_RESULT_SCHEMA_VERSION
    ref: ShadowEventRef
    observation: ShadowObservation

    @field_validator("ref")
    @classmethod
    def copy_event_reference(cls, value: object) -> ShadowEventRef:
        return _copy_exact_model(ShadowEventRef, value)

    @field_validator("observation")
    @classmethod
    def copy_observation(cls, value: object) -> ShadowObservation:
        return _copy_exact_model(ShadowObservation, value)

    @model_validator(mode="after")
    def reference_and_observation_identify_the_same_event(self) -> Self:
        if (
            self.ref.run_id != self.observation.run_id
            or self.ref.event_id != self.observation.event_id
            or self.ref.sequence != self.observation.sequence
        ):
            raise ValueError("shadow event result identities disagree")
        return self


def derive_shadow_observation_digest(observation: ShadowObservation) -> str:
    result: str | None = None
    try:
        checked = _copy_exact_model(ShadowObservation, observation)
        result = _observation_body_digest(checked)
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def _trusted_shadow_observation_sequence_is_exact(value: object) -> bool:
    try:
        if (
            type(value) is not _TrustedShadowObservationSequence
            or type(value._seal) is not _TrustedObservationSeal
        ):
            return False
        sealed = _TRUSTED_OBSERVATION_SEQUENCE_SEALS.get(value._seal)
        return (
            sealed is not None
            and len(sealed) == 3
            and value.sequence is sealed[0]
            and value.config is sealed[1]
            and value.redaction_policy_tag is sealed[2]
            and value._token is _TRUSTED_OBSERVATION_SEQUENCE_TOKEN
            and _detection_sequence_proof_is_exact(value.sequence)
            and _shadow_config_is_preflight_safe(value.config)
            and _payload_digest_is_preflight_safe(value.redaction_policy_tag)
            and value.redaction_policy_tag.algorithm is PayloadDigestAlgorithm.HMAC_SHA256
        )
    except Exception:
        return False


def _admit_shadow_observation_sequence(
    sequence: _DetectionSequenceProof,
    *,
    config: ShadowConfig,
    redaction_policy_tag: PayloadDigest,
) -> _TrustedShadowObservationSequence:
    result: _TrustedShadowObservationSequence | None = None
    try:
        if not _detection_sequence_proof_is_exact(sequence):
            raise ValueError("detection sequence proof is invalid")
        copied_config = _copy_shadow_config(config)
        copied_tag = _copy_payload_digest(redaction_policy_tag)
        events = sequence.events
        if (
            events[0].sequence != 1
            or any(right.timestamp < left.timestamp for left, right in pairwise(events))
            or any(
                event.event_id != derive_shadow_event_id(event.run_id, event.source_event_id)
                for event in events
            )
            or any(
                event.payload_digest.algorithm is not PayloadDigestAlgorithm.HMAC_SHA256
                for event in events
            )
            or copied_config.supported_signal_types != _SUPPORTED_SIGNAL_TYPES
            or copied_config.unsupported_signal_types != _UNSUPPORTED_SIGNAL_TYPES
        ):
            raise ValueError("shadow observation sequence is invalid")
        seal = _TrustedObservationSeal()
        result = _TrustedShadowObservationSequence(
            sequence=sequence,
            config=copied_config,
            redaction_policy_tag=copied_tag,
            _token=_TRUSTED_OBSERVATION_SEQUENCE_TOKEN,
            _seal=seal,
        )
        _TRUSTED_OBSERVATION_SEQUENCE_SEALS[seal] = (
            result.sequence,
            result.config,
            result.redaction_policy_tag,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None or not _trusted_shadow_observation_sequence_is_exact(result):
        raise ShadowInvariantError()
    return result


def _build_shadow_observation_trusted(
    admission: _TrustedShadowObservationSequence,
    extraction: _TrustedExtraction,
    *,
    input_kind: ShadowInputKind,
    source_event_digest: str,
    cli_input_ordinal: int | None = None,
) -> ShadowObservation:
    """Build from analyzer-owned proofs without recopying every historical prefix."""

    result: ShadowObservation | None = None
    try:
        if (
            not _trusted_shadow_observation_sequence_is_exact(admission)
            or not _trusted_extraction_is_exact(extraction)
            or extraction.trusted_context.sequence is not admission.sequence
        ):
            raise ValueError("trusted observation admission is invalid")
        trusted_context = extraction.trusted_context
        sequence = admission.sequence
        config = admission.config
        report = extraction.report
        end_ordinal = trusted_context.end_ordinal
        current = sequence.events[end_ordinal - 1]
        prefix_event_bytes = sequence.event_bytes[:end_ordinal]
        context_event_bytes = sequence.event_bytes[trusted_context.start_index : end_ordinal]
        if (
            trusted_context.context.current is not current
            or not _extraction_report_is_preflight_safe(report)
            or type(input_kind) is not ShadowInputKind
            or not _event_matches_input_kind(current, input_kind)
            or type(source_event_digest) is not str
            or not _is_digest(source_event_digest)
            or not hmac.compare_digest(
                source_event_digest,
                derive_shadow_source_event_digest(
                    current.run_id,
                    current.source_event_id,
                ),
            )
            or (
                cli_input_ordinal is not None
                and (
                    type(cli_input_ordinal) is not int
                    or not 1 <= cli_input_ordinal <= _MAX_SIGNED_64
                )
            )
        ):
            raise ValueError("trusted observation evidence is invalid")
        evaluation_types = tuple(item.signal_type for item in report.evaluations)
        detector_versions = {item.signal_type: item.detector_version for item in report.evaluations}
        configured_versions = {item.signal_type: item.detector_version for item in config.detectors}
        if (
            evaluation_types != config.supported_signal_types
            or detector_versions != configured_versions
        ):
            raise ValueError("trusted detector profile is inconsistent")
        feature_digest = _length_prefixed_sha256_iter(
            chain(
                (
                    SHADOW_OBSERVATION_SCHEMA_VERSION,
                    str(sequence.run_id),
                    str(current.sequence),
                    config.detector_profile_digest,
                    config.evaluator_configuration_digest,
                    str(trusted_context.context.events[0].sequence),
                    str(current.sequence),
                ),
                context_event_bytes,
                prefix_event_bytes,
                (canonical_json(evaluation) for evaluation in report.evaluations),
            ),
            domain=_FEATURE_SNAPSHOT_DIGEST_DOMAIN,
        )
        heuristic = evaluate_shadow_heuristic(
            report,
            input_kind=input_kind,
            config=config,
            feature_snapshot_digest=feature_digest,
        )
        body = _ShadowObservationBody(
            run_id=current.run_id,
            event_id=current.event_id,
            source_event_digest=source_event_digest,
            sequence=current.sequence,
            event_prefix_digest=_length_prefixed_sha256_iter(
                chain(
                    (str(sequence.run_id), str(current.sequence)),
                    prefix_event_bytes,
                ),
                domain=_EVENT_PREFIX_DIGEST_DOMAIN,
            ),
            context_first_sequence=trusted_context.context.events[0].sequence,
            context_last_sequence=current.sequence,
            context_event_count=len(trusted_context.context.events),
            context_truncated=trusted_context.start_index > 0,
            detection_context_digest=_length_prefixed_sha256_iter(
                chain(
                    (
                        str(sequence.run_id),
                        str(trusted_context.context.events[0].sequence),
                        str(current.sequence),
                    ),
                    context_event_bytes,
                ),
                domain=_DETECTION_CONTEXT_DIGEST_DOMAIN,
            ),
            redacted_event_digest=length_prefixed_sha256(
                prefix_event_bytes[-1],
                domain=_REDACTED_EVENT_DIGEST_DOMAIN,
            ),
            redaction_policy_tag=admission.redaction_policy_tag,
            detector_profile_digest=config.detector_profile_digest,
            evaluator_configuration_digest=config.evaluator_configuration_digest,
            extraction_report_digest=length_prefixed_sha256(
                canonical_json(report),
                domain=_EXTRACTION_REPORT_DIGEST_DOMAIN,
            ),
            feature_snapshot_digest=feature_digest,
            supported_signal_types=config.supported_signal_types,
            unsupported_signal_types=config.unsupported_signal_types,
            detector_evaluations=report.evaluations,
            detected_signals=report.signals,
            heuristic_evaluations=(heuristic,),
            cli_input_ordinal=cli_input_ordinal,
        )
        observation = ShadowObservation(
            **body.model_dump(mode="python", warnings=False),
            observation_digest=_observation_body_digest(body),
        )
        result = _copy_exact_model(ShadowObservation, observation)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def _try_detection_context(
    run_id: UUID,
    events: tuple[TraceEvent, ...],
) -> DetectionContext | None:
    try:
        return DetectionContext(run_id=run_id, events=events)
    except Exception:
        return None


def _event_matches_input_kind(event: TraceEvent, input_kind: ShadowInputKind) -> bool:
    try:
        spec = SHADOW_PROJECTION_MATRIX[input_kind]
        if (
            event.event_type is not spec.event_type
            or event.phase is not spec.phase
            or tuple(event.payload) != (spec.payload_namespace,)
        ):
            return False
        if input_kind is ShadowInputKind.OBSERVATION:
            if event.trust_label not in (
                TrustLabel.UNTRUSTED_TASK_INPUT,
                TrustLabel.UNTRUSTED_TOOL_OUTPUT,
                TrustLabel.UNTRUSTED_MODEL_OUTPUT,
                TrustLabel.UNTRUSTED_EXTERNAL_MEMORY,
            ):
                return False
        elif event.trust_label is not spec.trust_label:
            return False
        expected_parent_count = 1 if spec.parent == "action" else 0
        return len(event.parent_ids) == expected_parent_count
    except Exception:
        return False


def _select_detection_context(
    prefix: tuple[TraceEvent, ...],
) -> _SelectedDetectionContext:
    result: _SelectedDetectionContext | None = None
    try:
        copied = _copy_event_prefix(prefix)
        run_id = copied[0].run_id
        singleton = _try_detection_context(run_id, copied[-1:])
        if singleton is None:
            raise ValueError("a repository-admissible event is not singleton-valid")
        low = 0
        high = len(copied) - 1
        while low < high:
            midpoint = (low + high) // 2
            if _try_detection_context(run_id, copied[midpoint:]) is None:
                low = midpoint + 1
            else:
                high = midpoint
        selected = _try_detection_context(run_id, copied[low:])
        if selected is None:
            raise ValueError("selected shadow context failed final validation")
        checked = _copy_detection_context(selected)
        if low > 0 and _try_detection_context(run_id, copied[low - 1 :]) is not None:
            raise ValueError("selected shadow context is not the longest valid suffix")
        result = _SelectedDetectionContext(
            prefix=copied,
            context=checked,
            _token=_SELECTION_TOKEN,
        )
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def select_detection_context(prefix: tuple[TraceEvent, ...]) -> DetectionContext:
    """Select the longest validator-admissible suffix ending at the prefix cutoff."""

    return _select_detection_context(prefix).context


def _build_shadow_observation(
    *,
    prefix: tuple[TraceEvent, ...],
    context: DetectionContext,
    report: ExtractionReport,
    config: ShadowConfig,
    input_kind: ShadowInputKind,
    heuristic: ShadowHeuristicEvaluation,
    source_event_digest: str,
    redaction_policy_tag: PayloadDigest,
    cli_input_ordinal: int | None = None,
    selection: _SelectedDetectionContext | None,
) -> ShadowObservation:
    result: ShadowObservation | None = None
    try:
        copied_prefix, copied_context, copied_report, copied_config = _validate_feature_inputs(
            prefix=prefix,
            context=context,
            report=report,
            config=config,
        )
        if selection is None:
            selected = select_detection_context(copied_prefix)
            if selected != copied_context:
                raise ValueError("shadow context is not the deterministic longest suffix")
        elif (
            type(selection) is not _SelectedDetectionContext
            or selection._token is not _SELECTION_TOKEN
            or selection.prefix != copied_prefix
            or selection.context != copied_context
        ):
            raise ValueError("shadow context selection proof is invalid")
        copied_heuristic = _copy_heuristic(heuristic)
        copied_redaction_policy_tag = _copy_payload_digest(redaction_policy_tag)
        current = copied_prefix[-1]
        if type(input_kind) is not ShadowInputKind or not _event_matches_input_kind(
            current,
            input_kind,
        ):
            raise ValueError("shadow input kind does not match the observed event")
        if (
            type(source_event_digest) is not str
            or len(source_event_digest) != 64
            or any(character not in "0123456789abcdef" for character in source_event_digest)
        ):
            raise ValueError("shadow source-event identity is invalid")
        expected_source_digest = derive_shadow_source_event_digest(
            current.run_id,
            current.source_event_id,
        )
        if not hmac.compare_digest(source_event_digest, expected_source_digest):
            raise ValueError("shadow source-event identity does not match the observed event")
        if cli_input_ordinal is not None and (
            type(cli_input_ordinal) is not int or not 1 <= cli_input_ordinal <= _MAX_SIGNED_64
        ):
            raise ValueError("shadow CLI input ordinal is invalid")
        feature_digest = _feature_snapshot_digest(
            prefix=copied_prefix,
            context=copied_context,
            report=copied_report,
            config=copied_config,
        )
        if (
            copied_heuristic.evaluator_id != copied_config.evaluator_id
            or copied_heuristic.configuration_digest != copied_config.evaluator_configuration_digest
            or copied_heuristic.feature_snapshot_digest != feature_digest
        ):
            raise ValueError("shadow heuristic does not bind the observation inputs")
        expected_heuristic = evaluate_shadow_heuristic(
            copied_report,
            input_kind=input_kind,
            config=copied_config,
            feature_snapshot_digest=feature_digest,
        )
        if copied_heuristic != expected_heuristic:
            raise ValueError("shadow heuristic does not match the frozen evaluator")
        body = _ShadowObservationBody(
            run_id=current.run_id,
            event_id=current.event_id,
            source_event_digest=source_event_digest,
            sequence=current.sequence,
            event_prefix_digest=_event_prefix_digest(copied_prefix),
            context_first_sequence=copied_context.events[0].sequence,
            context_last_sequence=copied_context.current.sequence,
            context_event_count=len(copied_context.events),
            context_truncated=len(copied_context.events) != len(copied_prefix),
            detection_context_digest=_detection_context_digest(copied_context),
            redacted_event_digest=_redacted_event_digest(current),
            redaction_policy_tag=copied_redaction_policy_tag,
            detector_profile_digest=copied_config.detector_profile_digest,
            evaluator_configuration_digest=copied_config.evaluator_configuration_digest,
            extraction_report_digest=_extraction_report_digest(copied_report),
            feature_snapshot_digest=feature_digest,
            supported_signal_types=copied_config.supported_signal_types,
            unsupported_signal_types=copied_config.unsupported_signal_types,
            detector_evaluations=copied_report.evaluations,
            detected_signals=copied_report.signals,
            heuristic_evaluations=(copied_heuristic,),
            cli_input_ordinal=cli_input_ordinal,
        )
        observation = ShadowObservation(
            **body.model_dump(mode="python", warnings=False),
            observation_digest=_observation_body_digest(body),
        )
        result = _copy_exact_model(ShadowObservation, observation)
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def build_shadow_observation(
    *,
    prefix: tuple[TraceEvent, ...],
    context: DetectionContext,
    report: ExtractionReport,
    config: ShadowConfig,
    input_kind: ShadowInputKind,
    heuristic: ShadowHeuristicEvaluation,
    source_event_digest: str,
    redaction_policy_tag: PayloadDigest,
    cli_input_ordinal: int | None = None,
) -> ShadowObservation:
    """Build a self-verifying observation from explicit, already-persisted evidence."""

    return _build_shadow_observation(
        prefix=prefix,
        context=context,
        report=report,
        config=config,
        input_kind=input_kind,
        heuristic=heuristic,
        source_event_digest=source_event_digest,
        redaction_policy_tag=redaction_policy_tag,
        cli_input_ordinal=cli_input_ordinal,
        selection=None,
    )


def _build_shadow_observation_from_selection(
    *,
    selection: _SelectedDetectionContext,
    report: ExtractionReport,
    config: ShadowConfig,
    input_kind: ShadowInputKind,
    heuristic: ShadowHeuristicEvaluation,
    source_event_digest: str,
    redaction_policy_tag: PayloadDigest,
    cli_input_ordinal: int | None = None,
) -> ShadowObservation:
    if type(selection) is not _SelectedDetectionContext or selection._token is not _SELECTION_TOKEN:
        raise ShadowInvariantError()
    return _build_shadow_observation(
        prefix=selection.prefix,
        context=selection.context,
        report=report,
        config=config,
        input_kind=input_kind,
        heuristic=heuristic,
        source_event_digest=source_event_digest,
        redaction_policy_tag=redaction_policy_tag,
        cli_input_ordinal=cli_input_ordinal,
        selection=selection,
    )


__all__ = [
    "ShadowEventResult",
    "ShadowObservation",
    "build_shadow_observation",
    "derive_shadow_detection_context_digest",
    "derive_shadow_event_prefix_digest",
    "derive_shadow_extraction_report_digest",
    "derive_shadow_feature_snapshot_digest",
    "derive_shadow_observation_digest",
    "derive_shadow_redacted_event_digest",
    "select_detection_context",
]
