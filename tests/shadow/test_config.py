from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from saliencegate.domain import SignalType, canonical_json, length_prefixed_sha256
from saliencegate.shadow.config import (
    ShadowApplicability,
    ShadowConfig,
    ShadowDetectorSpec,
    build_shadow_extractor,
    validate_shadow_config,
)
from saliencegate.shadow.errors import ShadowConfigurationError
from saliencegate.shadow.inputs import ShadowInputKind
from saliencegate.signals import (
    AbstentionReason,
    RepeatedActionDetector,
    RepeatedFailureDetector,
    RepetitionConfig,
    TestFailureDetector,
    ToolErrorDetector,
)


class AliasShadowConfig(ShadowConfig):
    pass


class AliasShadowDetectorSpec(ShadowDetectorSpec):
    pass


def _detector_profile_payload(config: ShadowConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "detectors": config.detectors,
        "supported_signal_types": config.supported_signal_types,
        "unsupported_signal_types": config.unsupported_signal_types,
        "evaluator_configuration_digest": config.evaluator_configuration_digest,
    }


def _evaluator_configuration_payload(config: ShadowConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "evaluator_id": config.evaluator_id,
        "indeterminate_reasons": config.indeterminate_reasons,
        "applicability": config.applicability,
    }


def _with_recomputed_digests(config: ShadowConfig) -> ShadowConfig:
    evaluator_digest = length_prefixed_sha256(
        canonical_json(_evaluator_configuration_payload(config)),
        domain="saliencegate:shadow:evaluator-configuration:v1",
    )
    with_evaluator_digest = config.model_copy(
        update={"evaluator_configuration_digest": evaluator_digest}
    )
    profile_digest = length_prefixed_sha256(
        canonical_json(_detector_profile_payload(with_evaluator_digest)),
        domain="saliencegate:shadow:detector-profile:v1",
    )
    return with_evaluator_digest.model_copy(update={"detector_profile_digest": profile_digest})


def test_reference_config_freezes_the_real_detector_profile() -> None:
    config = ShadowConfig.reference()
    repetition = RepetitionConfig(window_events=8)
    expected_detectors = (
        ShadowDetectorSpec(
            signal_type=SignalType.REPEATED_ACTION,
            detector_version=RepeatedActionDetector(repetition).detector_version,
            repetition_window_events=8,
        ),
        ShadowDetectorSpec(
            signal_type=SignalType.REPEATED_FAILURE,
            detector_version=RepeatedFailureDetector(repetition).detector_version,
            repetition_window_events=8,
        ),
        ShadowDetectorSpec(
            signal_type=SignalType.TEST_FAILURE,
            detector_version=TestFailureDetector().detector_version,
            repetition_window_events=None,
        ),
        ShadowDetectorSpec(
            signal_type=SignalType.TOOL_ERROR,
            detector_version=ToolErrorDetector().detector_version,
            repetition_window_events=None,
        ),
    )

    assert config.schema_version == "shadow-config/v1"
    assert config.detectors == expected_detectors
    assert config.supported_signal_types == tuple(item.signal_type for item in expected_detectors)
    assert config.unsupported_signal_types == (
        SignalType.CONFLICT,
        SignalType.CONTEXT_SHIFT,
        SignalType.IRREVERSIBLE_ACTION,
        SignalType.STAGNATION,
        SignalType.STALE_CONSTRAINT,
    )
    assert config.evaluator_id == "any-detected-signal-baseline/v1"
    assert config.indeterminate_reasons == (
        AbstentionReason.AMBIGUOUS_PARENT_ACTION,
        AbstentionReason.INSUFFICIENT_HISTORY,
        AbstentionReason.PARENT_ACTION_MISSING,
        AbstentionReason.PRE_ACTION_INTERCEPTION_UNAVAILABLE,
        AbstentionReason.REDACTED_EQUIVALENCE_INPUT,
        AbstentionReason.STRUCTURED_EVIDENCE_INVALID,
        AbstentionReason.STRUCTURED_EVIDENCE_MISSING,
    )


def test_reference_config_freezes_the_lifecycle_applicability_matrix() -> None:
    assert ShadowConfig.reference().applicability == (
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


def test_reference_config_digests_are_domain_separated_and_self_verifying() -> None:
    config = ShadowConfig.reference()

    assert config.detector_profile_digest == length_prefixed_sha256(
        canonical_json(_detector_profile_payload(config)),
        domain="saliencegate:shadow:detector-profile:v1",
    )
    assert config.evaluator_configuration_digest == length_prefixed_sha256(
        canonical_json(_evaluator_configuration_payload(config)),
        domain="saliencegate:shadow:evaluator-configuration:v1",
    )
    assert config.detector_profile_digest != config.evaluator_configuration_digest
    assert len(config.detector_profile_digest) == 64
    assert len(config.evaluator_configuration_digest) == 64
    assert (
        config.evaluator_configuration_digest
        == "c518180ae3c472a5497805c23fb4e826b8806232246aaf6c27f1b0253b959e09"
    )
    assert (
        config.detector_profile_digest
        == "1c23bc4bb54d474c447def70fd11f5c5942cb55a8295ed7c8166e3c994bf9d6e"
    )


def test_detector_profile_digest_couples_the_evaluator_configuration() -> None:
    config = ShadowConfig.reference()
    changed_evaluator_digest = "0" * 64
    changed_payload = {
        **_detector_profile_payload(config),
        "evaluator_configuration_digest": changed_evaluator_digest,
    }

    changed_profile_digest = length_prefixed_sha256(
        canonical_json(changed_payload),
        domain="saliencegate:shadow:detector-profile:v1",
    )

    assert changed_profile_digest != config.detector_profile_digest


def test_config_is_strict_frozen_round_trippable_and_defensively_copied() -> None:
    config = ShadowConfig.reference()
    restored = ShadowConfig.model_validate_json(config.model_dump_json())

    assert restored == config
    assert restored is not config
    assert validate_shadow_config(config) == config
    assert validate_shadow_config(config) is not config
    with pytest.raises(ValidationError):
        config.evaluator_id = "replacement/v1"  # type: ignore[assignment,misc]
    with pytest.raises(ValidationError):
        ShadowConfig.model_validate({**config.model_dump(), "unexpected": True})


def test_config_copy_never_dispatches_through_the_caller_instance() -> None:
    config = ShadowConfig.reference()
    serializer_called = False

    def poisoned_model_dump(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("caller-controlled-secret")

    class PoisonedSerializer:
        def to_python(self, *args: object, **kwargs: object) -> object:
            nonlocal serializer_called
            del args, kwargs
            serializer_called = True
            raise AssertionError("caller-controlled-secret")

    object.__setattr__(config, "model_dump", poisoned_model_dump)
    object.__setattr__(config, "__pydantic_serializer__", PoisonedSerializer())

    copied = validate_shadow_config(config)

    assert copied == ShadowConfig.reference()
    assert "model_dump" not in copied.__dict__
    assert "__pydantic_serializer__" not in copied.__dict__
    assert serializer_called is False


def test_validator_rejects_profile_mutations_with_consistently_recomputed_digests() -> None:
    config = ShadowConfig.reference()
    changed_version = config.detectors[0].model_copy(
        update={"detector_version": "repeated-action/v1+review"}
    )
    changed_window = config.detectors[0].model_copy(update={"repetition_window_events": 9})
    mutations = (
        {"detectors": tuple(reversed(config.detectors))},
        {"detectors": (changed_version, *config.detectors[1:])},
        {"detectors": (changed_window, *config.detectors[1:])},
        {
            "supported_signal_types": (
                SignalType.REPEATED_ACTION,
                SignalType.REPEATED_FAILURE,
                SignalType.TEST_FAILURE,
                SignalType.CONFLICT,
            )
        },
        {
            "unsupported_signal_types": (
                SignalType.TOOL_ERROR,
                *config.unsupported_signal_types[1:],
            )
        },
        {"applicability": tuple(reversed(config.applicability))},
        {"indeterminate_reasons": tuple(reversed(config.indeterminate_reasons))},
    )

    for mutation in mutations:
        forged = _with_recomputed_digests(config.model_copy(update=mutation))

        assert forged.detector_profile_digest != config.detector_profile_digest
        with pytest.raises(ShadowConfigurationError):
            validate_shadow_config(forged)


@pytest.mark.parametrize(
    "mutation",
    (
        {"evaluator_configuration_digest": "0" * 64},
        {"detector_profile_digest": "0" * 64},
    ),
)
def test_validator_rejects_direct_digest_mutations(mutation: dict[str, object]) -> None:
    forged = ShadowConfig.reference().model_copy(update=mutation)

    with pytest.raises(ShadowConfigurationError):
        validate_shadow_config(forged)


def test_validator_rejects_mappings_subclasses_and_forged_nested_models_value_free() -> None:
    config = ShadowConfig.reference()
    alias = AliasShadowConfig.model_validate(config.model_dump(mode="python"))
    forged_spec = AliasShadowDetectorSpec.model_validate(
        config.detectors[0].model_dump(mode="python")
    )
    forged_nested = config.model_copy(update={"detectors": (forged_spec, *config.detectors[1:])})
    secret = "configuration-secret"
    forged_value = config.model_copy(
        update={
            "detectors": (
                config.detectors[0].model_copy(update={"detector_version": secret}),
                *config.detectors[1:],
            )
        }
    )

    for candidate in (config.model_dump(), alias, forged_nested, forged_value, object()):
        with pytest.raises(ShadowConfigurationError) as error:
            validate_shadow_config(cast(ShadowConfig, candidate))
        assert secret not in str(error.value)
        assert error.value.__cause__ is None
        assert error.value.__context__ is None


def test_extractor_builder_accepts_only_the_exact_reference_profile() -> None:
    config = ShadowConfig.reference()
    extractor = build_shadow_extractor(config)

    entries = extractor._entries
    assert tuple((item.signal_type, item.detector_version) for item in entries) == tuple(
        (item.signal_type, item.detector_version) for item in config.detectors
    )

    with pytest.raises(ShadowConfigurationError):
        build_shadow_extractor(config.model_copy(update={"detector_profile_digest": "0" * 64}))
