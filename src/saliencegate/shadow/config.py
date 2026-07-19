from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import SignalType, canonical_json, length_prefixed_sha256
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest
from saliencegate.shadow.errors import ShadowConfigurationError
from saliencegate.shadow.inputs import ShadowInputKind
from saliencegate.signals import (
    AbstentionReason,
    DeterministicSignalExtractor,
    RepeatedActionDetector,
    RepeatedFailureDetector,
    RepetitionConfig,
    TestFailureDetector,
    ToolErrorDetector,
)

_SCHEMA_VERSION: Literal["shadow-config/v1"] = "shadow-config/v1"
_EVALUATOR_ID: Literal["any-detected-signal-baseline/v1"] = "any-detected-signal-baseline/v1"
_DETECTOR_PROFILE_DOMAIN = "saliencegate:shadow:detector-profile:v1"
_EVALUATOR_CONFIGURATION_DOMAIN = "saliencegate:shadow:evaluator-configuration:v1"
_DETECTOR_VERSION = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_REPETITION_WINDOW_EVENTS = 8


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ShadowDetectorSpec(_FrozenModel):
    """One versioned detector in the immutable Shadow v1 profile."""

    signal_type: SignalType
    detector_version: ComponentIdentifier
    repetition_window_events: Annotated[int, Field(ge=2, le=10_000)] | None

    @field_validator("detector_version", mode="before")
    @classmethod
    def exact_detector_version(cls, value: object) -> object:
        if type(value) is not str or _DETECTOR_VERSION.fullmatch(value) is None:
            raise ValueError("detector version is invalid")
        return value

    @field_validator("repetition_window_events", mode="before")
    @classmethod
    def exact_repetition_window(cls, value: object) -> object:
        if value is not None and type(value) is not int:
            raise ValueError("repetition window is invalid")
        return value

    @model_validator(mode="after")
    def repetition_window_matches_detector(self) -> Self:
        is_repetition = self.signal_type in (
            SignalType.REPEATED_ACTION,
            SignalType.REPEATED_FAILURE,
        )
        if is_repetition != (self.repetition_window_events is not None):
            raise ValueError("repetition window does not match detector")
        return self


class ShadowApplicability(_FrozenModel):
    """The supported detector mask for one typed Shadow input."""

    input_kind: ShadowInputKind
    applicable_signal_types: Annotated[tuple[SignalType, ...], Field(max_length=4)]

    @model_validator(mode="after")
    def signal_types_are_unique(self) -> Self:
        if len(set(self.applicable_signal_types)) != len(self.applicable_signal_types):
            raise ValueError("applicability signal types are not unique")
        return self


class ShadowConfig(_FrozenModel):
    """A self-verifying, fully resolved Shadow Mode v1 configuration."""

    schema_version: Literal["shadow-config/v1"] = _SCHEMA_VERSION
    detectors: Annotated[tuple[ShadowDetectorSpec, ...], Field(min_length=4, max_length=4)]
    supported_signal_types: Annotated[tuple[SignalType, ...], Field(min_length=4, max_length=4)]
    unsupported_signal_types: Annotated[tuple[SignalType, ...], Field(min_length=5, max_length=5)]
    applicability: Annotated[tuple[ShadowApplicability, ...], Field(min_length=7, max_length=7)]
    evaluator_id: Literal["any-detected-signal-baseline/v1"]
    indeterminate_reasons: Annotated[
        tuple[AbstentionReason, ...],
        Field(min_length=7, max_length=7),
    ]
    evaluator_configuration_digest: Sha256Digest
    detector_profile_digest: Sha256Digest

    @model_validator(mode="after")
    def matches_reference_profile(self) -> Self:
        if (
            self.detectors != _reference_detector_specs()
            or self.supported_signal_types != _SUPPORTED_SIGNAL_TYPES
            or self.unsupported_signal_types != _UNSUPPORTED_SIGNAL_TYPES
            or self.applicability != _REFERENCE_APPLICABILITY
            or self.evaluator_id != _EVALUATOR_ID
            or self.indeterminate_reasons != _INDETERMINATE_REASONS
        ):
            raise ValueError("shadow configuration does not match the v1 profile")
        expected_evaluator_digest = _evaluator_configuration_digest(
            schema_version=self.schema_version,
            evaluator_id=self.evaluator_id,
            indeterminate_reasons=self.indeterminate_reasons,
            applicability=self.applicability,
        )
        if self.evaluator_configuration_digest != expected_evaluator_digest:
            raise ValueError("shadow evaluator configuration digest does not match")
        expected_profile_digest = _detector_profile_digest(
            schema_version=self.schema_version,
            detectors=self.detectors,
            supported_signal_types=self.supported_signal_types,
            unsupported_signal_types=self.unsupported_signal_types,
            evaluator_configuration_digest=self.evaluator_configuration_digest,
        )
        if self.detector_profile_digest != expected_profile_digest:
            raise ValueError("shadow detector profile digest does not match")
        return self

    @classmethod
    def reference(cls) -> ShadowConfig:
        """Build the immutable reference v1 profile from the installed built-ins."""

        result: ShadowConfig | None = None
        try:
            detectors = _reference_detector_specs()
            evaluator_digest = _evaluator_configuration_digest(
                schema_version=_SCHEMA_VERSION,
                evaluator_id=_EVALUATOR_ID,
                indeterminate_reasons=_INDETERMINATE_REASONS,
                applicability=_REFERENCE_APPLICABILITY,
            )
            profile_digest = _detector_profile_digest(
                schema_version=_SCHEMA_VERSION,
                detectors=detectors,
                supported_signal_types=_SUPPORTED_SIGNAL_TYPES,
                unsupported_signal_types=_UNSUPPORTED_SIGNAL_TYPES,
                evaluator_configuration_digest=evaluator_digest,
            )
            result = ShadowConfig(
                schema_version=_SCHEMA_VERSION,
                detectors=detectors,
                supported_signal_types=_SUPPORTED_SIGNAL_TYPES,
                unsupported_signal_types=_UNSUPPORTED_SIGNAL_TYPES,
                applicability=_REFERENCE_APPLICABILITY,
                evaluator_id=_EVALUATOR_ID,
                indeterminate_reasons=_INDETERMINATE_REASONS,
                evaluator_configuration_digest=evaluator_digest,
                detector_profile_digest=profile_digest,
            )
        except Exception:
            result = None
        if result is None:
            raise ShadowConfigurationError()
        return result


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
_INDETERMINATE_REASONS = (
    AbstentionReason.AMBIGUOUS_PARENT_ACTION,
    AbstentionReason.INSUFFICIENT_HISTORY,
    AbstentionReason.PARENT_ACTION_MISSING,
    AbstentionReason.PRE_ACTION_INTERCEPTION_UNAVAILABLE,
    AbstentionReason.REDACTED_EQUIVALENCE_INPUT,
    AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
    AbstentionReason.STRUCTURED_EVIDENCE_MISSING,
)
_REFERENCE_APPLICABILITY = (
    ShadowApplicability(input_kind=ShadowInputKind.START, applicable_signal_types=()),
    ShadowApplicability(
        input_kind=ShadowInputKind.ACTION,
        applicable_signal_types=(SignalType.REPEATED_ACTION,),
    ),
    ShadowApplicability(
        input_kind=ShadowInputKind.TOOL_RESULT,
        applicable_signal_types=(SignalType.TOOL_ERROR, SignalType.REPEATED_FAILURE),
    ),
    ShadowApplicability(
        input_kind=ShadowInputKind.TEST_RESULT,
        applicable_signal_types=(SignalType.TEST_FAILURE, SignalType.REPEATED_FAILURE),
    ),
    ShadowApplicability(input_kind=ShadowInputKind.OBSERVATION, applicable_signal_types=()),
    ShadowApplicability(
        input_kind=ShadowInputKind.CONTROLLER_ERROR,
        applicable_signal_types=(SignalType.TOOL_ERROR,),
    ),
    ShadowApplicability(input_kind=ShadowInputKind.FINISH, applicable_signal_types=()),
)


def _reference_detector_specs() -> tuple[ShadowDetectorSpec, ...]:
    repetition = RepetitionConfig(window_events=_REPETITION_WINDOW_EVENTS)
    repeated_action = RepeatedActionDetector(repetition)
    repeated_failure = RepeatedFailureDetector(repetition)
    test_failure = TestFailureDetector()
    tool_error = ToolErrorDetector()
    return (
        ShadowDetectorSpec(
            signal_type=repeated_action.signal_type,
            detector_version=repeated_action.detector_version,
            repetition_window_events=_REPETITION_WINDOW_EVENTS,
        ),
        ShadowDetectorSpec(
            signal_type=repeated_failure.signal_type,
            detector_version=repeated_failure.detector_version,
            repetition_window_events=_REPETITION_WINDOW_EVENTS,
        ),
        ShadowDetectorSpec(
            signal_type=test_failure.signal_type,
            detector_version=test_failure.detector_version,
            repetition_window_events=None,
        ),
        ShadowDetectorSpec(
            signal_type=tool_error.signal_type,
            detector_version=tool_error.detector_version,
            repetition_window_events=None,
        ),
    )


def _evaluator_configuration_digest(
    *,
    schema_version: str,
    evaluator_id: str,
    indeterminate_reasons: tuple[AbstentionReason, ...],
    applicability: tuple[ShadowApplicability, ...],
) -> str:
    payload = {
        "schema_version": schema_version,
        "evaluator_id": evaluator_id,
        "indeterminate_reasons": indeterminate_reasons,
        "applicability": applicability,
    }
    return length_prefixed_sha256(
        canonical_json(payload),
        domain=_EVALUATOR_CONFIGURATION_DOMAIN,
    )


def _detector_profile_digest(
    *,
    schema_version: str,
    detectors: tuple[ShadowDetectorSpec, ...],
    supported_signal_types: tuple[SignalType, ...],
    unsupported_signal_types: tuple[SignalType, ...],
    evaluator_configuration_digest: str,
) -> str:
    payload = {
        "schema_version": schema_version,
        "detectors": detectors,
        "supported_signal_types": supported_signal_types,
        "unsupported_signal_types": unsupported_signal_types,
        "evaluator_configuration_digest": evaluator_configuration_digest,
    }
    return length_prefixed_sha256(canonical_json(payload), domain=_DETECTOR_PROFILE_DOMAIN)


def _detector_spec_is_safe(value: object) -> bool:
    return (
        type(value) is ShadowDetectorSpec
        and type(value.signal_type) is SignalType
        and type(value.detector_version) is str
        and _DETECTOR_VERSION.fullmatch(value.detector_version) is not None
        and (value.repetition_window_events is None or type(value.repetition_window_events) is int)
    )


def _applicability_is_safe(value: object) -> bool:
    return (
        type(value) is ShadowApplicability
        and type(value.input_kind) is ShadowInputKind
        and type(value.applicable_signal_types) is tuple
        and all(type(item) is SignalType for item in value.applicable_signal_types)
    )


def _config_is_safe(value: object) -> bool:
    try:
        return (
            type(value) is ShadowConfig
            and type(value.schema_version) is str
            and type(value.detectors) is tuple
            and len(value.detectors) == 4
            and all(_detector_spec_is_safe(item) for item in value.detectors)
            and type(value.supported_signal_types) is tuple
            and len(value.supported_signal_types) == 4
            and all(type(item) is SignalType for item in value.supported_signal_types)
            and type(value.unsupported_signal_types) is tuple
            and len(value.unsupported_signal_types) == 5
            and all(type(item) is SignalType for item in value.unsupported_signal_types)
            and type(value.applicability) is tuple
            and len(value.applicability) == 7
            and all(_applicability_is_safe(item) for item in value.applicability)
            and type(value.evaluator_id) is str
            and type(value.indeterminate_reasons) is tuple
            and len(value.indeterminate_reasons) == 7
            and all(type(item) is AbstentionReason for item in value.indeterminate_reasons)
            and type(value.evaluator_configuration_digest) is str
            and type(value.detector_profile_digest) is str
        )
    except Exception:
        return False


def validate_shadow_config(value: object) -> ShadowConfig:
    """Defensively copy an exact reference config or fail without caller values."""

    validated: ShadowConfig | None = None
    if _config_is_safe(value):
        assert type(value) is ShadowConfig
        try:
            candidate = ShadowConfig.model_validate(
                ShadowConfig.__pydantic_serializer__.to_python(
                    value,
                    mode="python",
                    warnings=False,
                )
            )
            if candidate == value:
                validated = candidate
        except Exception:
            validated = None
    if validated is None:
        raise ShadowConfigurationError()
    return validated


def build_shadow_extractor(config: ShadowConfig) -> DeterministicSignalExtractor:
    """Instantiate the four installed built-ins after exact compatibility checks."""

    validated: ShadowConfig | None = None
    extractor: DeterministicSignalExtractor | None = None
    try:
        validated = validate_shadow_config(config)
    except ShadowConfigurationError:
        validated = None
    if validated is not None:
        try:
            repetition = RepetitionConfig(window_events=_REPETITION_WINDOW_EVENTS)
            detectors = (
                RepeatedActionDetector(repetition),
                RepeatedFailureDetector(repetition),
                TestFailureDetector(),
                ToolErrorDetector(),
            )
            installed = tuple(
                ShadowDetectorSpec(
                    signal_type=detector.signal_type,
                    detector_version=detector.detector_version,
                    repetition_window_events=(
                        _REPETITION_WINDOW_EVENTS
                        if detector.signal_type
                        in (SignalType.REPEATED_ACTION, SignalType.REPEATED_FAILURE)
                        else None
                    ),
                )
                for detector in detectors
            )
            if installed == validated.detectors:
                extractor = DeterministicSignalExtractor(detectors)
        except Exception:
            extractor = None
    if extractor is None:
        raise ShadowConfigurationError()
    return extractor


__all__ = [
    "ShadowApplicability",
    "ShadowConfig",
    "ShadowDetectorSpec",
    "build_shadow_extractor",
    "validate_shadow_config",
]
